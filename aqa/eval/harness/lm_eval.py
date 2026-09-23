"""lm-evaluation-harness adapter.

Builds and runs the exact ``lm_eval`` invocation the historic
``local_dispatch/phases/evaluate.sh`` used, then parses lm-eval's own results
json.  Generative tasks (chat-template + fewshot-as-multiturn + generation
budget) vs loglikelihood tasks are handled per the recipe's ``type``.
"""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from ...logging_util import AqaError, get_logger
from ..registry import BenchmarkRecipe, BenchmarkResult

_log = get_logger("harness.lm_eval")

# How many trailing log lines to fold into a harness failure so the real
# traceback (not just the "rc=1" wrapper) reaches the fix-loop's error signature.
_FAIL_TAIL_LINES = 120


def _log_tail(path: Path, lines: int = _FAIL_TAIL_LINES) -> str:
    """Return the tail of an lm_eval run log, prefixed for readability.

    The exception the harness raises otherwise carries only a generic
    ``lm_eval failed for … (rc=1)`` wrapper; embedding the log tail lets
    ``extract_signature``/``classify_error`` key knowledge on the *actual*
    error (e.g. an ``AssertionError`` from a kernel) instead of the task list.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    tail = "\n".join(text.splitlines()[-lines:]).strip()
    return f"\n--- {path.name} (last {lines} lines) ---\n{tail}" if tail else ""


class LmEvalHarness:
    name = "lm_eval"

    def __init__(
        self,
        *,
        num_gpus: int = 1,
        batch_size: str = "auto",
        limit: int | None = None,
        gen_max_toks: int = 2048,
        model_args: dict | None = None,
        gen_kwargs: dict | None = None,
        seed: int = 42,
        apply_chat_template: bool | None = None,
        fewshot_as_multiturn: bool | None = None,
    ):
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.limit = limit
        self.gen_max_toks = gen_max_toks
        self.model_args = model_args or {}
        self.gen_kwargs = gen_kwargs or {}
        self.seed = seed
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def available(self) -> bool:
        return shutil.which("lm_eval") is not None

    def _is_generative(self, recipe: BenchmarkRecipe) -> bool:
        if self.apply_chat_template is not None:
            return recipe.type == "generative" or self.apply_chat_template
        return recipe.type == "generative" or recipe.apply_chat_template

    def _batch_size(self, recipe: BenchmarkRecipe) -> str:
        """Per-recipe batch size (``recipes/benchmarks/<task>.yaml``) or the global."""
        return str(recipe.batch_size if recipe.batch_size is not None else self.batch_size)

    def group_key(self, recipe: BenchmarkRecipe) -> tuple:
        """Params that must match for two tasks to share one ``lm_eval`` call."""
        gen = self._is_generative(recipe)
        fewshot_mt = (
            self.fewshot_as_multiturn
            if self.fewshot_as_multiturn is not None
            else recipe.fewshot_as_multiturn or gen
        )
        gk = self.gen_kwargs or recipe.gen_kwargs or ({"max_gen_toks": self.gen_max_toks} if gen else {})
        return (gen, bool(fewshot_mt), recipe.num_fewshot, self._batch_size(recipe),
                tuple(sorted(gk.items())))

    def build_command(
        self, recipes: BenchmarkRecipe | list[BenchmarkRecipe], model: str, backend, out_dir: Path
    ) -> list[str]:
        if isinstance(recipes, BenchmarkRecipe):
            recipes = [recipes]
        lead = recipes[0]
        generative = self._is_generative(lead)
        model_args = backend.build_model_args(
            model, num_gpus=self.num_gpus, gen_max_toks=self.gen_max_toks,
            generative=generative, extra_args=self.model_args,
        )
        tasks = ",".join(r.task_name() for r in recipes)
        cmd = [
            "lm_eval",
            "--model", backend.lm_eval_model_flag,
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", self._batch_size(lead),
            "--output_path", str(out_dir),
            "--log_samples",
            "--seed", str(self.seed),
        ]
        if lead.num_fewshot is not None:
            cmd += ["--num_fewshot", str(lead.num_fewshot)]
        if generative:
            fewshot_mt = (
                self.fewshot_as_multiturn
                if self.fewshot_as_multiturn is not None
                else lead.fewshot_as_multiturn or True
            )
            cmd += ["--apply_chat_template"]
            if fewshot_mt:
                cmd += ["--fewshot_as_multiturn"]
            if backend.lm_eval_model_flag == "hf":
                gen = self.gen_kwargs or lead.gen_kwargs or {"max_gen_toks": self.gen_max_toks}
                cmd += ["--gen_kwargs", ",".join(f"{k}={v}" for k, v in gen.items())]
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        return cmd

    def run(self, recipe: BenchmarkRecipe, model: str, backend, out_dir: Path) -> BenchmarkResult:
        return self.run_group([recipe], model, backend, out_dir)[0]

    def run_group(
        self, recipes: list[BenchmarkRecipe], model: str, backend, out_dir: Path
    ) -> list[BenchmarkResult]:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(recipes, model, backend, out_dir)
        _log.info("[CMD] %s", " ".join(shlex.quote(c) for c in cmd))
        env_extra = backend.env_overrides()
        if not self.available():
            raise AqaError(
                "lm_eval is not installed; install with `pip install autoquant-agent[lm-eval]`"
            )
        import os

        env = os.environ | env_extra
        tag = recipes[0].name if len(recipes) == 1 else "+".join(r.name for r in recipes)
        log_path = out_dir / f"{tag}.eval.log"
        from ...proc import stream_subprocess

        rc = stream_subprocess(cmd, log_path, env=env)
        if rc != 0:
            raise AqaError(
                f"lm_eval failed for {tag} (rc={rc}); see {log_path}"
                + _log_tail(log_path)
            )
        data = self._load_results(out_dir)
        cmd_str = shlex.join(cmd)
        results = [self._result_for(recipe, data) for recipe in recipes]
        for r in results:
            r.command = cmd_str
        return results

    def parse_results(self, recipe: BenchmarkRecipe, out_dir: Path) -> BenchmarkResult:
        return self._result_for(recipe, self._load_results(out_dir))

    def _load_results(self, out_dir: Path) -> dict:
        results_files = sorted(out_dir.rglob("results*.json"))
        if not results_files:
            raise AqaError(f"no lm_eval results json found under {out_dir}")
        return json.loads(results_files[-1].read_text())

    def _result_for(self, recipe: BenchmarkRecipe, data: dict) -> BenchmarkResult:
        task_results = data.get("results", {})
        task_data = task_results.get(recipe.task_name()) or next(iter(task_results.values()), {})
        score, metric = _extract_metric(task_data, recipe)
        n = int(data.get("n-samples", {}).get(recipe.task_name(), {}).get("effective", 0) or 0)
        return BenchmarkResult(
            name=recipe.name, score=score, metric=metric, num_samples=n, raw=task_data
        )


def _extract_metric(task_data: dict, recipe: BenchmarkRecipe) -> tuple[float, str]:
    """Pick the primary accuracy-like metric from an lm_eval task result block."""
    prefer = list(recipe.scoring.prefer)
    if recipe.scoring.metric:
        prefer = [recipe.scoring.metric] + prefer
    # exact_match variants first (generative), then acc_norm, then acc.
    default_order = [
        "exact_match,strict-match", "exact_match,flexible-extract", "exact_match",
        "acc_norm,none", "acc_norm", "acc,none", "acc",
    ]
    for key in prefer + default_order:
        if key in task_data and isinstance(task_data[key], (int, float)):
            return float(task_data[key]), key
    # last resort: first numeric non-stderr metric.
    for key, val in task_data.items():
        if isinstance(val, (int, float)) and "stderr" not in key and "alias" not in key:
            return float(val), key
    return 0.0, "unknown"

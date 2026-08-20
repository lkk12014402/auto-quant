"""Native evaluation harness (migrated from NeMo Evaluator, Apache-2.0).

The core loop -- render prompt -> generate -> score with the scorer library -- is
kept pure and testable via :meth:`NativeHarness.evaluate_samples` (inject a
``generate_fn``).  :meth:`run` wires a real generation function from the backend
(HF transformers) and requires model weights + torch to actually execute.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...logging_util import AqaError, get_logger
from ..dataset import load_samples, render_prompt
from ..registry import BenchmarkRecipe, BenchmarkResult
from ..scoring import BUILTIN_SCORERS, ScorerInput, build_judge_scorer

_log = get_logger("harness.native")

GenerateFn = Callable[[str], str]


def resolve_scorer(
    spec_name: str, spec: Any = None,
) -> Callable[[ScorerInput], dict[str, Any]]:
    """Resolve a scorer by builtin name or ``module.py:func`` escape hatch.

    ``spec`` is the recipe's :class:`ScoringSpec`; it lets configurable scorers
    (currently ``llm_judge``) read their config (``spec.judge``) at build time.
    """
    if spec_name == "llm_judge":
        judge = getattr(spec, "judge", None) if spec is not None else None
        return build_judge_scorer(judge)
    if spec_name == "code_exec":
        from ..code_exec import build_code_exec_scorer
        code_spec = getattr(spec, "code_exec", None) if spec is not None else None
        return build_code_exec_scorer(code_spec)
    if spec_name in BUILTIN_SCORERS:
        return BUILTIN_SCORERS[spec_name]
    if ":" in spec_name:
        mod_name, _, func = spec_name.partition(":")
        if mod_name.endswith(".py"):
            spec = importlib.util.spec_from_file_location("_aqa_scorer", mod_name)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, func)
        module = importlib.import_module(mod_name)
        return getattr(module, func)
    raise AqaError(f"unknown scorer {spec_name!r}")


class NativeHarness:
    name = "native"

    def __init__(self, *, limit: int | None = None):
        self.limit = limit

    def available(self) -> bool:
        return True

    def evaluate_samples(
        self,
        recipe: BenchmarkRecipe,
        samples: list[dict[str, Any]],
        generate_fn: GenerateFn,
    ) -> BenchmarkResult:
        """Pure, testable core: score model outputs over ``samples``."""
        scorer_name = recipe.scoring.scorer or "exact_match"
        scorer = resolve_scorer(scorer_name, recipe.scoring)
        if not recipe.prompt or not recipe.target_field:
            raise AqaError(f"native recipe {recipe.name} needs `prompt` and `target_field`")
        total, correct = 0, 0.0
        per_sample: list[float] = []
        for sample in samples:
            prompt = render_prompt(recipe.prompt, sample)
            prediction = generate_fn(prompt)
            target = str(sample.get(recipe.target_field, ""))
            choices = sample.get("choices")
            result = scorer(ScorerInput(
                prediction=prediction, target=target, choices=choices,
                extra={"question": prompt, "prompt": prompt, "sample": sample},
            ))
            s = float(result.get("score", 0.0))
            per_sample.append(s)
            correct += s
            total += 1
        score = correct / total if total else 0.0
        return BenchmarkResult(
            name=recipe.name, score=score, metric=scorer_name, num_samples=total,
            per_sample=per_sample,
        )

    def run(self, recipe: BenchmarkRecipe, model: str, backend, out_dir: Path) -> BenchmarkResult:
        if not recipe.dataset:
            raise AqaError(f"native recipe {recipe.name} needs a `dataset` URI")
        samples = list(load_samples(recipe.dataset, limit=self.limit))
        generate_fn = _build_hf_generate_fn(model, recipe)
        result = self.evaluate_samples(recipe, samples, generate_fn)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{recipe.name}.native.json").write_text(
            __import__("json").dumps(result.__dict__, indent=2, default=str)
        )
        return result


def _build_hf_generate_fn(model: str, recipe: BenchmarkRecipe) -> GenerateFn:
    """Construct a greedy HF-transformers generation function (needs torch)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise AqaError(
            "native run needs transformers+torch: pip install autoquant-agent[hf]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    max_new = int(recipe.gen_kwargs.get("max_gen_toks", 256))

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
        with torch.no_grad():
            out = lm.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text

    return generate

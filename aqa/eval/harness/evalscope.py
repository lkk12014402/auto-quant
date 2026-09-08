"""EvalScope harness: reasoning benchmarks (aime24/25/26, gpqa_diamond, ...).

EvalScope never launches vLLM itself, so this harness owns the serving
lifecycle: spin up ``vllm serve`` for the target model, wait for health, run
``evalscope eval`` against the OpenAI-compatible endpoint, then tear the
server down (even on failure — an 8-GPU server must never leak).

Thinking modes ride on the request, not the server: for the "off" run
``apply_mode_to_cfg`` drops ``chat_template_kwargs`` into the evalscope
options, and the harness folds them into
``generation_config.extra_body.chat_template_kwargs`` (vLLM honors per-request
template kwargs, so one server can serve both modes).

Dataset names, shot counts and generation defaults follow EvalScope's own
docs (0-shot AIME/GPQA, Qwen3 reasoning generation config).  Reports are
parsed from ``reports/<model>/<dataset>.json`` (schema v2: metrics[0].score).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ...logging_util import AqaError, get_logger
from ..registry import BenchmarkRecipe, BenchmarkResult

_log = get_logger("harness.evalscope")

_FAIL_TAIL_LINES = 120
_HEALTH_POLL_INTERVAL = 10


def _log_tail(path: Path, lines: int = _FAIL_TAIL_LINES) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    tail = "\n".join(text.splitlines()[-lines:]).strip()
    return f"\n--- {path.name} (last {lines} lines) ---\n{tail}" if tail else ""


class EvalScopeHarness:
    name = "evalscope"

    def __init__(
        self,
        *,
        num_gpus: int = 1,
        limit: int | None = None,
        port: int = 8801,
        repeats: int = 1,
        eval_batch_size: int = 32,
        generation_config: dict | None = None,
        dataset_args: dict | None = None,
        server_args: dict | None = None,
        chat_template_kwargs: dict | None = None,
        startup_timeout: int = 3600,
        api_key: str = "EMPTY",
    ):
        self.num_gpus = num_gpus
        self.limit = limit
        self.port = port
        self.repeats = repeats
        self.eval_batch_size = eval_batch_size
        # Reasoning-friendly defaults (Qwen3 best-practice values).
        self.generation_config = generation_config or {
            "max_tokens": 30000, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "stream": True, "timeout": 60000,
        }
        self.dataset_args = dataset_args or {}
        self.server_args = server_args or {}
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.startup_timeout = startup_timeout
        self.api_key = api_key

    def available(self) -> bool:
        return shutil.which("evalscope") is not None and shutil.which("vllm") is not None

    def group_key(self, recipe: BenchmarkRecipe) -> tuple:
        """All evalscope datasets share one server + one eval call by default."""
        return ("evalscope",)

    # ── command builders (pure; unit-tested without a server) ────────────────

    def build_serve_command(self, model: str, served_name: str) -> list[str]:
        cmd = [
            "vllm", "serve", model,
            "--served-model-name", served_name,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.num_gpus),
        ]
        for k, v in self.server_args.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        return cmd

    def build_eval_command(self, recipes: list[BenchmarkRecipe], served_name: str,
                           out_dir: Path) -> list[str]:
        gen = dict(self.generation_config)
        if self.chat_template_kwargs:
            extra = dict(gen.get("extra_body", {}))
            extra["chat_template_kwargs"] = {
                **extra.get("chat_template_kwargs", {}), **self.chat_template_kwargs}
            gen["extra_body"] = extra
        ds_args = dict(self.dataset_args)
        for r in recipes:
            ds_args.setdefault(r.task_name(), {"few_shot_num": 0})
        cmd = [
            "evalscope", "eval",
            "--model", served_name,
            "--api-url", f"http://127.0.0.1:{self.port}/v1",
            "--api-key", self.api_key,
            "--eval-type", "openai_api",
            "--datasets", *[r.task_name() for r in recipes],
            "--dataset-args", json.dumps(ds_args),
            "--repeats", str(self.repeats),
            "--generation-config", json.dumps(gen),
            "--eval-batch-size", str(self.eval_batch_size),
            "--work-dir", str(out_dir),
            "--no-timestamp",
        ]
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        return cmd

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _wait_ready(self, proc: subprocess.Popen, log_path: Path) -> None:
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AqaError(
                    f"vllm serve exited early (rc={proc.returncode}); see {log_path}"
                    + _log_tail(log_path))
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)
        raise AqaError(
            f"vllm serve did not become ready within {self.startup_timeout}s"
            + _log_tail(log_path))

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass

    def run(self, recipe: BenchmarkRecipe, model: str, backend, out_dir: Path) -> BenchmarkResult:
        return self.run_group([recipe], model, backend, out_dir)[0]

    def run_group(self, recipes: list[BenchmarkRecipe], model: str, backend,
                  out_dir: Path) -> list[BenchmarkResult]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.available():
            raise AqaError(
                "evalscope harness needs both `evalscope` and `vllm` on PATH; "
                "install with `pip install evalscope vllm`")
        served_name = self.server_args.get("served_model_name") or "aqa-evalscope"
        serve_cmd = self.build_serve_command(model, served_name)
        eval_cmd = self.build_eval_command(recipes, served_name, out_dir)
        serve_log = out_dir / "vllm_serve.log"
        eval_log = out_dir / "evalscope.log"
        _log.info("[CMD] %s", shlex.join(serve_cmd))
        with serve_log.open("w") as lf:
            proc = subprocess.Popen(
                serve_cmd, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group: clean teardown
            )
        try:
            self._wait_ready(proc, serve_log)
            _log.info("[CMD] %s", shlex.join(eval_cmd))
            from ...proc import stream_subprocess

            rc = stream_subprocess(eval_cmd, eval_log)
            if rc != 0:
                raise AqaError(
                    f"evalscope failed (rc={rc}); see {eval_log}" + _log_tail(eval_log))
        finally:
            self._stop(proc)
        results = [self._result_for(r, served_name, out_dir) for r in recipes]
        for r in results:
            r.command = shlex.join(eval_cmd)
        return results

    def _result_for(self, recipe: BenchmarkRecipe, served_name: str,
                    out_dir: Path) -> BenchmarkResult:
        report = out_dir / "reports" / served_name / f"{recipe.task_name()}.json"
        if not report.is_file():
            candidates = sorted(out_dir.glob(f"reports/*/{recipe.task_name()}.json"))
            if not candidates:
                raise AqaError(f"no evalscope report for {recipe.task_name()} under {out_dir}")
            report = candidates[-1]
        data = json.loads(report.read_text())
        metrics = data.get("metrics") or []
        if not metrics:
            raise AqaError(f"evalscope report {report} carries no metrics")
        score = float(metrics[0]["score"])
        n = int(metrics[0].get("num", 0) or 0)
        return BenchmarkResult(
            name=recipe.name, score=score,
            metric=metrics[0].get("identity", {}).get("name", "accuracy"),
            num_samples=n, raw=data)

"""Pipeline orchestrator: quantize -> evaluate (+ baseline) -> gate -> sinks.

Replaces the historic string-built ``CONTAINER_BOOTSTRAP`` with an explicit,
testable sequence.  Agent fix-loop / patch capture hook points are marked; they
wrap the quantize and evaluate steps when ``agent.fix_loop`` is enabled.
"""

from __future__ import annotations

from pathlib import Path

from .agent.fix_loop import (
    FixLoopConfig,
    PhaseResult,
    default_agent_runner,
    run_fix_loop,
)
from .config import RunConfig
from .eval.registry import EvalResult
from .eval.runner import run_evaluation
from .executors import ProcessState
from .logging_util import AqaError, get_logger
from .quantize import run_quant_command, run_quantization
from .registry import SINKS
from .report import evaluate_gate, write_accuracy_json
from .report.sinks import RunMeta

_log = get_logger("pipeline")


def _tail(path: Path, lines: int = 100) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _fix_config(run_cfg: RunConfig) -> FixLoopConfig:
    a = run_cfg.agent
    return FixLoopConfig(
        max_attempts=a.max_fix_attempts,
        tier_attempt_budget=a.tier_attempt_budget,
        drift_threshold=a.drift_threshold,
        agent_timeout=a.agent_timeout,
        tiers=list(a.tiers),
        user_context=a.context or "",
    )


def _num_gpus(run_cfg: RunConfig) -> int:
    ssh = run_cfg.executor.ssh
    if ssh and ssh.gpus:
        return len([g for g in ssh.gpus.split(",") if g.strip()])
    return 1


def _artifact_name(run_cfg: RunConfig, suffix: str) -> str:
    short = run_cfg.model.split("/")[-1]
    if run_cfg.quant and suffix == "quant":
        return f"{short}-AutoRound-{run_cfg.quant.scheme}-{run_cfg.quant.method}"
    return f"{short}-BF16-Baseline"


def _dispatch_sinks(run_cfg: RunConfig, run_dir: Path, meta: RunMeta) -> None:
    for sink_cfg in run_cfg.output.sinks:
        kwargs = sink_cfg.model_dump(exclude={"type"})
        if sink_cfg.type == "local_fs" and "path" not in kwargs:
            kwargs["path"] = run_cfg.output.dir
        sink = SINKS.create(sink_cfg.type, **kwargs)
        try:
            sink.upload(run_dir, meta)
        except Exception as exc:  # noqa: BLE001 - sinks must not abort the run
            _log.warning("sink %s failed: %s", sink_cfg.type, exc)


def _optimized_command(run_cfg: RunConfig, quant_dir: Path, out_dir: Path) -> str | None:
    """Try agent-optimize; return a validated auto-round command or None (fall back)."""
    from .agent.optimize import run_agent_optimize
    from .agent.tiers import resolve_tiers

    tiers = resolve_tiers(list(run_cfg.agent.tiers))
    runner = default_agent_runner(run_cfg.agent.agent_timeout)
    try:
        return run_agent_optimize(
            run_cfg.model,
            scheme=run_cfg.quant.scheme,
            method=run_cfg.quant.method,
            export_format=run_cfg.quant.export_format,
            output_dir=str(quant_dir),
            tier=tiers[-1],  # optimize with the strongest available tier
            run_agent=runner,
            user_context=run_cfg.agent.context or "",
            log_dir=out_dir / "logs" / "agent_optimize",
        )
    except Exception as exc:  # noqa: BLE001 - optimize can only help, never block
        _log.warning("agent-optimize failed (%s); using deterministic recipe", exc)
        return None


def _make_patch_capture(run_cfg: RunConfig, out_dir: Path):
    """Build a PatchCapture when agent patch capture is enabled, else None."""
    if not (run_cfg.agent.fix_loop and run_cfg.agent.patch.capture):
        return None
    from .agent.patch import PatchCapture
    return PatchCapture(out_dir / "patch_snapshot",
                        track_vllm=run_cfg.agent.patch.track_vllm)


def _run_quantize(run_cfg: RunConfig, quant_dir: Path, out_dir: Path) -> None:
    """Quantize, wrapping the phase in the agent fix-loop when enabled."""
    quant_cfg = run_cfg.quant
    log_path = quant_dir / "quantize.log"

    optimized: str | None = None
    if run_cfg.agent.optimize:
        _log.info("== agent-optimize (analyse model -> quant recipe) ==")
        optimized = _optimized_command(run_cfg, quant_dir, out_dir)
        if optimized:
            _log.info("using agent-optimized quant command")

    def _phase() -> PhaseResult:
        try:
            if optimized:
                run_quant_command(optimized, quant_dir)
            else:
                run_quantization(run_cfg.model, quant_cfg, quant_dir)
            return PhaseResult(ok=True)
        except AqaError as exc:
            return PhaseResult(ok=False, log_text=_tail(log_path) or str(exc))

    if not run_cfg.agent.fix_loop:
        if optimized:
            run_quant_command(optimized, quant_dir)
        else:
            run_quantization(run_cfg.model, quant_cfg, quant_dir)
        return

    fr = run_fix_loop(
        "quantize", _phase, config=_fix_config(run_cfg), out_dir=out_dir,
        lesson_store=None, model_id=run_cfg.model,
        scheme=quant_cfg.scheme, method=quant_cfg.method,
        patch_capture=_make_patch_capture(run_cfg, out_dir),
    )
    if not fr.ok:
        raise AqaError(f"quantize failed after fix-loop: {fr.reason}")


def _run_evaluate(run_cfg: RunConfig, model: str, eval_dir: Path,
                  out_dir: Path, num_gpus: int) -> EvalResult:
    """Evaluate, wrapping the phase in the agent fix-loop when enabled.

    A phase is considered failed when the harness raises OR every benchmark scores
    exactly 0 (acc=0 => fail, matching the leaderboard's failure heuristic).
    """
    holder: dict[str, EvalResult] = {}

    def _phase() -> PhaseResult:
        try:
            res = run_evaluation(model, run_cfg.eval, eval_dir, num_gpus=num_gpus)
        except Exception as exc:  # noqa: BLE001 - surface as a fixable phase failure
            return PhaseResult(ok=False, log_text=str(exc))
        holder["result"] = res
        if not res.benchmarks:
            return PhaseResult(ok=False, log_text="evaluation produced no benchmark results")
        if all(b.score == 0 for b in res.benchmarks.values()):
            return PhaseResult(ok=False, log_text="all benchmark scores are 0 (acc=0 => fail)")
        return PhaseResult(ok=True)

    if not run_cfg.agent.fix_loop:
        return run_evaluation(model, run_cfg.eval, eval_dir, num_gpus=num_gpus)

    fr = run_fix_loop(
        "evaluate", _phase, config=_fix_config(run_cfg), out_dir=out_dir,
        model_id=run_cfg.model,
        scheme=run_cfg.quant.scheme if run_cfg.quant else "",
        method=run_cfg.quant.method if run_cfg.quant else "",
        patch_capture=_make_patch_capture(run_cfg, out_dir),
    )
    if "result" not in holder:
        raise AqaError(f"evaluate failed after fix-loop: {fr.reason}")
    if not fr.ok:
        _log.warning("evaluate fix-loop unresolved (%s); using last result", fr.reason)
    return holder["result"]


def _resolve_model(run_cfg: RunConfig, out_dir: Path, num_gpus: int):
    """Resolve the evaluation target via the configured model provider.

    ``pretrained`` returns the model as-is; ``quantize`` runs the (fix-loop-
    wrapped) quantization and returns the checkpoint dir; ``endpoint`` returns a
    served endpoint.  Quantization is thus one pluggable provider among others.
    """
    from .providers.base import ProviderContext
    from .registry import PROVIDERS

    provider = PROVIDERS.create(run_cfg.provider.type)
    ctx = ProviderContext(
        out_dir=out_dir,
        num_gpus=num_gpus,
        quantize_fn=lambda quant_dir: _run_quantize(run_cfg, quant_dir, out_dir),
    )
    return provider.provide(run_cfg, ctx)


def run_pipeline(run_cfg: RunConfig, out_dir: Path) -> ProcessState:
    """Execute the full pipeline; returns terminal ProcessState."""
    num_gpus = _num_gpus(run_cfg)
    baseline: EvalResult | None = None

    # ── optional bf16 baseline evaluation ────────────────────────────────────
    if run_cfg.eval.baseline:
        _log.info("== baseline eval (bf16) ==")
        base_dir = out_dir / "baseline"
        baseline = run_evaluation(run_cfg.model, run_cfg.eval, base_dir, num_gpus=num_gpus)
        write_accuracy_json(baseline, base_dir)
        _dispatch_sinks(run_cfg, base_dir, RunMeta(
            model=run_cfg.model, artifact_name=_artifact_name(run_cfg, "baseline")))

    # ── resolve the eval target (provider: pretrained | quantize | endpoint) ──
    resolved = _resolve_model(run_cfg, out_dir, num_gpus)
    eval_model = resolved.model

    # ── evaluation of the (provided) model ───────────────────────────────────
    _log.info("== evaluate ==")
    eval_dir = out_dir / "eval"
    result = _run_evaluate(run_cfg, eval_model, eval_dir, out_dir, num_gpus)
    write_accuracy_json(result, eval_dir)

    # ── gate ─────────────────────────────────────────────────────────────────
    gate_verdict: str | None = None
    if run_cfg.gate is not None:
        decision = evaluate_gate(result, baseline, run_cfg.gate)
        gate_verdict = decision.verdict
        (out_dir / "gate.json").write_text(
            __import__("json").dumps({"verdict": decision.verdict, "reasons": decision.reasons},
                                     indent=2))
        _log.info("gate: %s (%s)", decision.verdict, "; ".join(decision.reasons))

    results = {n: b.score for n, b in result.benchmarks.items()}
    _dispatch_sinks(run_cfg, eval_dir, RunMeta(
        model=run_cfg.model,
        scheme=run_cfg.quant.scheme if run_cfg.quant else None,
        method=run_cfg.quant.method if run_cfg.quant else None,
        artifact_name=_artifact_name(run_cfg, "quant"),
        run_root=out_dir,
        status=_derive_status(results, gate_verdict),
        extra={"results": results},
    ))

    # ── optional: upload the quantized checkpoint to a HF model repo ──────────
    _maybe_upload_model(run_cfg, resolved, result, out_dir)
    return ProcessState.DONE


def _derive_status(results: dict[str, float], gate_verdict: str | None) -> str:
    """Status string for the results write-back (mirrors local_dispatch).

    An empty result set or any benchmark scoring exactly 0 is treated as an eval
    failure, so a forgotten status write-back can't masquerade as success.
    """
    if not results:
        return "Eval Failed"
    if any(float(v) == 0.0 for v in results.values()):
        return "Eval Failed"
    if gate_verdict and gate_verdict.lower() in {"fail", "failed", "reject", "rejected"}:
        return "Gate Failed"
    return "Finished"


def _maybe_upload_model(run_cfg: RunConfig, resolved, result: EvalResult, out_dir: Path) -> None:
    """Push the quantized checkpoint to a HF *model* repo on success.

    Only runs for the ``quantize`` provider with an ``upload:`` block.  A failed
    upload is warned about but never aborts the (already successful) run.
    """
    if run_cfg.provider.type != "quantize" or run_cfg.provider.upload is None:
        return
    if resolved.kind != "path":
        return
    model_dir = Path(resolved.model)
    if not model_dir.is_dir():
        _log.warning("model upload skipped: quantized dir missing (%s)", model_dir)
        return

    from .report.sinks import HfModelSink

    cfg = run_cfg.provider.upload
    sink = HfModelSink(
        repo=cfg.repo,
        repo_name=cfg.repo_name,
        token_env=cfg.token_env,
        private=cfg.private,
        endpoint=cfg.endpoint,
        disable_xet=cfg.disable_xet,
        model_card=cfg.model_card,
    )
    meta = RunMeta(
        model=run_cfg.model,
        scheme=run_cfg.quant.scheme if run_cfg.quant else None,
        method=run_cfg.quant.method if run_cfg.quant else None,
        artifact_name=cfg.repo_name or _artifact_name(run_cfg, "quant"),
        extra={"results": {n: b.score for n, b in result.benchmarks.items()}},
    )
    try:
        url = sink.upload(model_dir, meta)
        _log.info("== model uploaded == %s", url)
    except Exception as exc:  # noqa: BLE001 - upload must not abort a successful run
        _log.warning("model upload failed: %s", exc)

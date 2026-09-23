"""Pipeline orchestrator: quantize -> evaluate (+ baseline) -> gate -> sinks.

Replaces the historic string-built ``CONTAINER_BOOTSTRAP`` with an explicit,
testable sequence.  Agent fix-loop / patch capture hook points are marked; they
wrap the quantize and evaluate steps when ``agent.fix_loop`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent.fix_loop import (
    FixLoopConfig,
    PhaseResult,
    default_agent_runner,
    run_fix_loop,
)
from .config import RunConfig
from .eval.registry import BenchmarkRegistry, EvalResult
from .eval.runner import run_evaluation
from .executors import ProcessState
from .hooks import resolve_hooks, run_hooks
from .logging_util import AqaError, get_logger
from .quantize import (
    build_autoround_command,
    ensure_device_map,
    run_quant_command,
    run_quantization,
)
from .registry import SINKS
from .report import evaluate_gate, write_accuracy_json
from .report.manifest import build_manifest, record_quant_command, write_manifest
from .report.sinks import RunMeta

if TYPE_CHECKING:  # pragma: no cover
    from .agent.fix_loop import FixResult

_log = get_logger("pipeline")


@dataclass
class HealTally:
    """Accumulates the agent self-heal effort spent to *produce* a green run.

    The escalating agent fix-loop contributes here, per phase, so the reproducibility
    manifest can report how much effort (turns + best-effort tokens) a score cost.
    ``record_deterministic`` is retained for backward compat with the manifest schema
    (agent-written eval overrides are surfaced via ``EvalResult.healed`` instead).
    """

    _phases: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _slot(self, phase: str) -> dict[str, Any]:
        from .agent_eval.stats import TokenStats
        return self._phases.setdefault(
            phase, {"agent_attempts": 0, "deterministic_heals": 0,
                    "tiers": 0, "tokens": TokenStats()})

    def record_fix(self, phase: str, fr: FixResult) -> None:
        slot = self._slot(phase)
        slot["agent_attempts"] += fr.attempts
        if fr.attempts:
            slot["tiers"] = max(slot["tiers"], fr.tier_index + 1)
        slot["tokens"].merge(fr.tokens)

    def record_deterministic(self, phase: str, applied: list[str]) -> None:
        self._slot(phase)["deterministic_heals"] += len(applied)

    def block(self) -> dict[str, Any] | None:
        """Aggregate a manifest ``self_heal`` block (None when no heal happened)."""
        from .agent_eval.stats import TokenStats
        total_attempts = sum(p["agent_attempts"] for p in self._phases.values())
        total_det = sum(p["deterministic_heals"] for p in self._phases.values())
        tokens = TokenStats()
        for p in self._phases.values():
            tokens.merge(p["tokens"])
        if total_attempts == 0 and total_det == 0 and tokens.is_empty():
            return None
        per_phase: dict[str, Any] = {}
        for name, p in self._phases.items():
            if not (p["agent_attempts"] or p["deterministic_heals"]):
                continue
            entry: dict[str, Any] = {
                "agent_attempts": p["agent_attempts"],
                "deterministic_heals": p["deterministic_heals"],
                "tiers": p["tiers"],
            }
            if not p["tokens"].is_empty():
                entry["tokens"] = p["tokens"].to_dict()
            per_phase[name] = entry
        block: dict[str, Any] = {
            "agent_attempts": total_attempts,
            "deterministic_heals": total_det,
            "per_phase": per_phase,
        }
        if not tokens.is_empty():
            block["tokens"] = tokens.to_dict()
        return block


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
    ex = run_cfg.executor
    if ex.num_gpus:  # explicit wins (e.g. carried into the in-container local run)
        return max(1, ex.num_gpus)
    ssh = ex.ssh
    if ssh and ssh.gpus:
        return max(1, len([g for g in ssh.gpus.split(",") if g.strip()]))
    return 1


def _make_run_id(run_cfg: RunConfig) -> str:
    """Stable per-run id shared by the manifest and the sink upload folder."""
    from .executors._remote import make_run_id

    return make_run_id(run_cfg)


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
    from .agent.optimize import build_quant_constraints, run_agent_optimize
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
            constraints=build_quant_constraints(run_cfg.quant),
            log_dir=out_dir / "logs" / "agent_optimize",
        )
    except Exception as exc:  # noqa: BLE001 - optimize can only help, never block
        # An exception here is always unexpected (a legitimate "no usable
        # command" returns None, not raises), so surface the type + traceback
        # loudly -- a silent one-line degrade previously masked a real bug that
        # discarded a correct recipe and fell back to the default.
        _log.warning(
            "agent-optimize crashed (%s: %s); using deterministic recipe",
            type(exc).__name__, exc, exc_info=True,
        )
        return None


def _make_patch_capture(run_cfg: RunConfig, out_dir: Path):
    """Build a PatchCapture when agent patch capture is enabled, else None."""
    if not (run_cfg.agent.fix_loop and run_cfg.agent.patch.capture):
        return None
    from .agent.patch import PatchCapture
    return PatchCapture(out_dir / "patch_snapshot",
                        track_vllm=run_cfg.agent.patch.track_vllm)


def _make_knowledge(run_cfg: RunConfig):
    """Build the shared knowledge store (synced down once), or None if disabled."""
    if not run_cfg.agent.fix_loop:
        return None
    from .agent.knowledge import KnowledgeStore
    kn = KnowledgeStore.from_config(run_cfg.agent.knowledge)
    if kn is not None:
        try:
            kn.sync_down()
        except Exception as exc:  # noqa: BLE001 - knowledge sync is best-effort
            _log.warning("knowledge: sync_down failed: %s", exc)
    return kn


def _run_quantize(run_cfg: RunConfig, quant_dir: Path, out_dir: Path,
                  num_gpus: int = 1, knowledge=None, run_id: str = "",
                  heal: HealTally | None = None) -> None:
    """Quantize, wrapping the phase in the agent fix-loop when enabled."""
    quant_cfg = run_cfg.quant
    log_path = quant_dir / "quantize.log"

    optimized: str | None = None
    if run_cfg.agent.optimize:
        _log.info("== agent-optimize (analyse model -> quant recipe) ==")
        optimized = _optimized_command(run_cfg, quant_dir, out_dir)
        if optimized:
            optimized = ensure_device_map(optimized, quant_cfg, num_gpus)
            _log.info("using agent-optimized quant command")

    # Freeze the *actual* command that will run so the manifest can reproduce it.
    if optimized:
        record_quant_command(out_dir, optimized, source="agent_optimized")
    else:
        import shlex
        built = build_autoround_command(run_cfg.model, quant_cfg, quant_dir,
                                        num_gpus=num_gpus)
        record_quant_command(out_dir, shlex.join(built), source="builder")

    hook_chain = resolve_hooks(run_cfg)

    def _phase() -> PhaseResult:
        try:
            if optimized:
                run_quant_command(optimized, quant_dir, hooks=hook_chain)
            else:
                run_quantization(run_cfg.model, quant_cfg, quant_dir, num_gpus=num_gpus,
                                 hooks=hook_chain)
            return PhaseResult(ok=True)
        except AqaError as exc:
            return PhaseResult(ok=False, log_text=_tail(log_path) or str(exc))

    if not run_cfg.agent.fix_loop:
        if optimized:
            run_quant_command(optimized, quant_dir, hooks=hook_chain)
        else:
            run_quantization(run_cfg.model, quant_cfg, quant_dir, num_gpus=num_gpus,
                             hooks=hook_chain)
        return

    fr = run_fix_loop(
        "quantize", _phase, config=_fix_config(run_cfg), out_dir=out_dir,
        lesson_store=None, model_id=run_cfg.model,
        scheme=quant_cfg.scheme, method=quant_cfg.method,
        patch_capture=_make_patch_capture(run_cfg, out_dir),
        knowledge=knowledge, replay=run_cfg.agent.knowledge.replay, run_id=run_id,
    )
    if heal is not None:
        heal.record_fix("quantize", fr)
    if not fr.ok:
        raise AqaError(f"quantize failed after fix-loop: {fr.reason}")


def _run_evaluate(run_cfg: RunConfig, model: str, eval_dir: Path,
                  out_dir: Path, num_gpus: int, knowledge=None,
                  run_id: str = "", phase_name: str = "evaluate",
                  heal: HealTally | None = None) -> EvalResult:
    """Evaluate, wrapping the phase in the agent fix-loop when enabled.

    A phase is considered failed when the harness raises OR every benchmark scores
    exactly 0 (acc=0 => fail, matching the leaderboard's failure heuristic). On
    failure the fix-loop hands the agent the full traceback and lets it decide --
    there is no deterministic pre-pass.
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
        # after_benchmark validator hooks (e.g. eval_sanity): an AqaError verdict
        # fails the phase and feeds the fix-loop's error signature.
        registry = BenchmarkRegistry()
        try:
            for bench in res.benchmarks.values():
                run_hooks("after_benchmark", run_cfg, recipe=registry.get(bench.name),
                          result_or_exception=bench, out_dir=eval_dir)
        except AqaError as exc:
            return PhaseResult(ok=False, log_text=str(exc))
        if all(b.score == 0 for b in res.benchmarks.values()):
            return PhaseResult(ok=False, log_text="all benchmark scores are 0 (acc=0 => fail)")
        return PhaseResult(ok=True)

    if not run_cfg.agent.fix_loop:
        return run_evaluation(model, run_cfg.eval, eval_dir, num_gpus=num_gpus)

    fr = run_fix_loop(
        phase_name, _phase, config=_fix_config(run_cfg), out_dir=out_dir,
        model_id=run_cfg.model,
        scheme=run_cfg.quant.scheme if run_cfg.quant else "",
        method=run_cfg.quant.method if run_cfg.quant else "",
        patch_capture=_make_patch_capture(run_cfg, out_dir),
        knowledge=knowledge, replay=run_cfg.agent.knowledge.replay, run_id=run_id,
    )
    if heal is not None:
        heal.record_fix(phase_name, fr)
    if "result" not in holder:
        raise AqaError(f"evaluate failed after fix-loop: {fr.reason}")
    if not fr.ok:
        _log.warning("evaluate fix-loop unresolved (%s); using last result", fr.reason)
    return holder["result"]


def _resolve_model(run_cfg: RunConfig, out_dir: Path, num_gpus: int, knowledge=None,
                   run_id: str = "", heal: HealTally | None = None):
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
        quantize_fn=lambda quant_dir: _run_quantize(
            run_cfg, quant_dir, out_dir, num_gpus, knowledge, run_id, heal=heal),
    )
    return provider.provide(run_cfg, ctx)


def run_pipeline(run_cfg: RunConfig, out_dir: Path) -> ProcessState:
    """Execute the full pipeline; returns terminal ProcessState.

    On a phase failure (quantize/evaluate raises after the fix-loop) the run's
    artifacts are still written back: eval logs, agent fix records and captured
    patches are uploaded via the configured sinks with a failure ``status`` (so a
    crashed run is observable on the leaderboard instead of silently vanishing),
    and if a valid quantized checkpoint exists it is still pushed to its model
    repo. The original error is then re-raised so the caller still sees rc!=0.
    """
    num_gpus = _num_gpus(run_cfg)
    run_id = _make_run_id(run_cfg)
    baseline: EvalResult | None = None
    resolved = None
    eval_dir = out_dir / "eval"
    heal = HealTally()
    # Shared fix-knowledge (lessons + replayable patches); synced down once here.
    knowledge = _make_knowledge(run_cfg)

    try:
        # ── optional bf16 baseline evaluation ────────────────────────────────
        if run_cfg.eval.baseline:
            _log.info("== baseline eval (bf16) ==")
            base_dir = out_dir / "baseline"
            baseline = _run_evaluate(run_cfg, run_cfg.model, base_dir, out_dir,
                                     num_gpus, knowledge, run_id,
                                     phase_name="baseline", heal=heal)
            write_accuracy_json(baseline, base_dir)
            _dispatch_sinks(run_cfg, base_dir, RunMeta(
                model=run_cfg.model, artifact_name=_artifact_name(run_cfg, "baseline")))

        # ── resolve the eval target (provider: pretrained|quantize|endpoint) ──
        resolved = _resolve_model(run_cfg, out_dir, num_gpus, knowledge, run_id,
                                  heal=heal)
        eval_model = resolved.model

        # ── evaluation of the (provided) model ───────────────────────────────
        _log.info("== evaluate ==")
        result = _run_evaluate(run_cfg, eval_model, eval_dir, out_dir, num_gpus,
                               knowledge, run_id, heal=heal)
        write_accuracy_json(result, eval_dir)

        # ── gate ─────────────────────────────────────────────────────────────
        gate_verdict: str | None = None
        if run_cfg.gate is not None:
            decision = evaluate_gate(result, baseline, run_cfg.gate)
            gate_verdict = decision.verdict
            (out_dir / "gate.json").write_text(
                __import__("json").dumps({"verdict": decision.verdict, "reasons": decision.reasons},
                                         indent=2))
            _log.info("gate: %s (%s)", decision.verdict, "; ".join(decision.reasons))

        results = {n: b.score for n, b in result.benchmarks.items()}

        # ── reproducibility manifest (model + config + env + results) ────────
        manifest = build_manifest(
            run_cfg, result, run_id=run_id, baseline=baseline,
            gate_verdict=gate_verdict, out_dir=out_dir, self_heal=heal.block())
        write_manifest(manifest, out_dir)

        _write_terminal_state(out_dir, ProcessState.DONE, run_cfg.model)
        _dispatch_sinks(run_cfg, eval_dir, RunMeta(
            model=run_cfg.model,
            scheme=run_cfg.quant.scheme if run_cfg.quant else None,
            method=run_cfg.quant.method if run_cfg.quant else None,
            artifact_name=_artifact_name(run_cfg, "quant"),
            run_root=out_dir,
            run_id=run_id,
            status=_derive_status(results, gate_verdict),
            extra={"results": results},
            knowledge_dir=knowledge.local_dir if knowledge else None,
        ))

        # ── optional: upload the quantized checkpoint to a HF model repo ──────
        _maybe_upload_model(run_cfg, resolved, result, out_dir)
        _finalize_knowledge(knowledge)
        return ProcessState.DONE
    except AqaError as exc:
        _writeback_on_failure(run_cfg, out_dir, eval_dir, resolved, exc, knowledge,
                              run_id=run_id, heal=heal)
        _finalize_knowledge(knowledge)
        raise


def _finalize_knowledge(knowledge) -> None:
    """Log the replay hit-rate and push new lessons/patches/stats to the repo."""
    if knowledge is None:
        return
    try:
        s = knowledge.stats()
        _log.info("knowledge: replay stats -> replay_hit=%d reference_used=%d agent=%d "
                  "unresolved=%d hit_rate=%.2f",
                  s["replay_hit"], s["reference_used"], s["agent_fixed"],
                  s["unresolved"], s["hit_rate"])
        knowledge.push_up()
    except Exception as exc:  # noqa: BLE001 - finalization is best-effort
        _log.warning("knowledge: finalize failed: %s", exc)


def _writeback_on_failure(run_cfg: RunConfig, out_dir: Path, eval_dir: Path,
                          resolved, exc: AqaError, knowledge=None,
                          *, run_id: str = "", heal: HealTally | None = None) -> None:
    """Best-effort write-back of a *failed* run (results/logs/patches + status).

    Mirrors the success-path sink dispatch so a crashed run is still uploaded and
    marked failed on the leaderboard.  If quantization succeeded but evaluation
    failed, a valid checkpoint may still exist -- it is uploaded too.  All errors
    here are swallowed (we are already unwinding a failure and must re-raise).
    """
    reason = str(exc)
    status = "Quant Failed" if reason.startswith("quantize failed") else "Eval Failed"
    run_dir = eval_dir if eval_dir.exists() else out_dir
    _log.warning("run failed (%s); writing back partial results + status=%s", reason, status)
    _write_terminal_state(out_dir, ProcessState.FAILED, run_cfg.model)
    # A failed run still gets a manifest (empty results) so the attempt is auditable.
    try:
        empty = EvalResult(model=run_cfg.model, backend=run_cfg.eval.backend)
        manifest = build_manifest(
            run_cfg, empty, run_id=run_id or _make_run_id(run_cfg),
            extra_env={"failure": {"status": status, "reason": reason[:500]}},
            out_dir=out_dir, self_heal=heal.block() if heal is not None else None)
        write_manifest(manifest, out_dir)
    except Exception as man_exc:  # noqa: BLE001 - manifest is best-effort
        _log.warning("failure-path manifest error: %s", man_exc)
    try:
        _dispatch_sinks(run_cfg, run_dir, RunMeta(
            model=run_cfg.model,
            scheme=run_cfg.quant.scheme if run_cfg.quant else None,
            method=run_cfg.quant.method if run_cfg.quant else None,
            artifact_name=_artifact_name(run_cfg, "quant"),
            run_root=out_dir,
            run_id=run_id,
            status=status,
            extra={"results": {}, "error": reason},
            knowledge_dir=knowledge.local_dir if knowledge else None,
        ))
    except Exception as sink_exc:  # noqa: BLE001 - failure write-back is best-effort
        _log.warning("failure write-back sink error: %s", sink_exc)

    # A successfully quantized model can still be published even if eval failed.
    if resolved is not None and status != "Quant Failed":
        try:
            empty = EvalResult(model=run_cfg.model, backend=run_cfg.eval.backend)
            _maybe_upload_model(run_cfg, resolved, empty, out_dir)
        except Exception as up_exc:  # noqa: BLE001 - best-effort
            _log.warning("failure-path model upload error: %s", up_exc)


def _write_terminal_state(out_dir: Path, state: "ProcessState", model: str) -> None:
    """Refresh ``aqa_run.json`` with the terminal state *before* sinks upload.

    The executor writes ``{"state": "running"}`` at start and only rewrites the
    final state after ``run_pipeline`` returns — but sink dispatch (the HF
    dataset write-back) happens inside the pipeline, so without this the
    uploaded artifact would permanently claim "running".
    """
    meta = out_dir / "aqa_run.json"
    if not meta.exists():
        return
    try:
        meta.write_text(__import__("json").dumps({"model": model, "state": state.value}))
    except OSError:  # pragma: no cover - cosmetic write, never block the run
        pass


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

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
from .eval.registry import EvalResult
from .eval.runner import run_evaluation
from .executors import ProcessState
from .console_tee import start_console_tee, stop_console_tee
from .logging_util import AqaError, attach_run_log, detach_run_log, get_logger
from .quantize import (
    build_autoround_command,
    ensure_device_map,
    run_quant_command,
    run_quant_script,
    run_quantization,
)
from .registry import SINKS
from .report import evaluate_gate, read_eval_result, write_accuracy_json
from .report.manifest import build_manifest, record_quant_command, write_manifest
from .report.phase_artifacts import (
    write_analysis,
    write_env_lock,
    write_eval_report,
    write_quant_plan,
)
from .report.sinks import RunMeta
from .session import Phase, Session, open_session

if TYPE_CHECKING:  # pragma: no cover
    from .agent.fix_loop import FixResult
    from .agent.recipe_kb import RecipeHit

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


def _fix_hooks(run_cfg: RunConfig, session: "Session | None") -> dict:
    """Build the P4 propose/decide hooks for the fix-loop (RFC §3.5).

    All are advisory-by-default: the Critic only records claim-vs-reality, the
    PolicyGate records named violations (and reverts only when ``enforce``), and
    the event sink / state seed are additive.  Returns a kwargs dict to splat into
    :func:`run_fix_loop` so absent features simply aren't passed.
    """
    a = run_cfg.agent
    hooks: dict = {}
    if a.critic:
        from .agent.critic import Critic
        hooks["critic"] = Critic()
    if a.policy_gate.enabled:
        from .agent.policy_gate import PolicyGate
        hooks["policy_gate"] = PolicyGate(
            enforce=a.policy_gate.enforce,
            max_net_deletions=a.policy_gate.max_net_deletions)
    if session is not None:
        hooks["on_event"] = lambda t, f: session.event(t, **f)
        try:
            hooks["state_seed"] = session.state.prompt_seed()
        except Exception as exc:  # noqa: BLE001 - re-seed is best-effort
            _log.debug("state_seed render failed: %s", exc)
    return hooks


_FIXLOOP_PHASE = {"quantize": Phase.QUANT, "evaluate": Phase.EVAL,
                  "baseline": Phase.BASELINE}


def _allowed_actions_block(phase_name: str) -> str:
    """Render the current phase's allowed-action whitelist for the fix prompt (RFC §S3)."""
    from .session.actions import allowed_actions_for

    ph = _FIXLOOP_PHASE.get(phase_name)
    if ph is None:
        return ""
    acts = allowed_actions_for(ph)
    if not acts:
        return ""
    return f"current phase: {ph.value}\nallowed actions: {', '.join(acts)}"


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
    from .report.names import canonical_artifact_name

    if suffix == "sweep":
        short = run_cfg.model.split("/")[-1]
        return f"{short}-AutoRound-sweep"
    if suffix == "baseline":
        short = run_cfg.model.split("/")[-1]
        return f"{short}-BF16-Baseline"
    # `quant` artifacts use the canonical `<Model>-<Scheme>[-Mixed][-FP8KV][-FP8Attn]
    # [-CT][-RTN]-AutoRound` form (aqa/report/names.py). The previous inline form only
    # carried scheme+method, so mxfp4 v2 (FP8 KV) and v3 (FP8 KV + attention) collapsed
    # onto one name and their HF model repos would overwrite each other.
    return canonical_artifact_name(run_cfg)


def _report_org(run_cfg: RunConfig) -> str | None:
    """Org for dataset/model upload paths, from ``report_model`` when set.

    Local-path runs (``model=/models/...``) cannot derive an org from the model
    id, so ``report_model`` (e.g. ``Tongyi-MAI/Z-Image-Turbo``) supplies it.
    """
    rm = getattr(run_cfg.output, "report_model", None)
    if rm and "/" in rm:
        return rm.split("/", 1)[0]
    return None


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


def _optimized_command(run_cfg: RunConfig, quant_dir: Path, out_dir: Path,
                       hot_start: str = "") -> str | None:
    """Try agent-optimize; return a validated auto-round command or None (fall back).

    ``hot_start`` (a Recipe-KB match rendered as prompt text) is prepended to the
    operator context so the agent starts from a validated recipe rather than
    guessing — advisory only, the agent still re-validates against this box.
    """
    from .agent.knowledge import default_local_dir
    from .agent.optimize import build_quant_constraints, run_agent_optimize
    from .agent.tiers import resolve_tiers

    knowledge_dir = run_cfg.agent.knowledge.local_dir or default_local_dir()
    tiers = resolve_tiers(list(run_cfg.agent.tiers))
    runner = default_agent_runner(run_cfg.agent.agent_timeout)
    user_context = "\n\n".join(
        part for part in (hot_start, run_cfg.agent.context or "") if part.strip())
    try:
        return run_agent_optimize(
            run_cfg.model,
            scheme=run_cfg.quant.scheme,
            method=run_cfg.quant.method,
            export_format=run_cfg.quant.export_format,
            output_dir=str(quant_dir),
            tier=tiers[-1],  # optimize with the strongest available tier
            run_agent=runner,
            user_context=user_context,
            constraints=build_quant_constraints(run_cfg.quant),
            log_dir=out_dir / "logs" / "agent_optimize",
            knowledge_dir=knowledge_dir,
            use_synced=run_cfg.agent.knowledge.use_references,
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
                  heal: HealTally | None = None, hot_start: str = "",
                  session: "Session | None" = None) -> None:
    """Quantize, wrapping the phase in the agent fix-loop when enabled."""
    quant_cfg = run_cfg.quant
    log_path = quant_dir / "quantize.log"
    override = quant_cfg.command_override

    optimized: str | None = None
    if override:
        # RFC 0002 §5 reproduce: replay the reviewed manifest script verbatim.
        _log.info("== reproduce: running the manifest's frozen quant command ==")
        record_quant_command(out_dir, override, source="reproduce")
    else:
        if run_cfg.agent.optimize:
            _log.info("== agent-optimize (analyse model -> quant recipe) ==")
            optimized = _optimized_command(run_cfg, quant_dir, out_dir,
                                           hot_start=hot_start)
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

    def _run() -> None:
        if override:
            run_quant_script(override, quant_dir)
        elif optimized:
            run_quant_command(optimized, quant_dir)
        else:
            run_quantization(run_cfg.model, quant_cfg, quant_dir,
                             num_gpus=num_gpus)

    def _post_export() -> None:
        kind = getattr(quant_cfg, "export_repair", None)
        if not kind:
            return
        from .quantize import repair_export, resolve_export_dir
        from .report.manifest import record_post_export

        info = repair_export(resolve_export_dir(quant_dir), kind)
        record_post_export(out_dir, info)

    def _hooks() -> None:
        from .hooks import resolve_hooks, run_hook_chain

        run_hook_chain("after_quantize", resolve_hooks(run_cfg), run_cfg,
                       quant_cfg=quant_cfg, output_dir=quant_dir)

    def _phase() -> PhaseResult:
        try:
            _run()
            _post_export()
            _hooks()
            return PhaseResult(ok=True)
        except AqaError as exc:
            return PhaseResult(ok=False, log_text=_tail(log_path) or str(exc))

    if not run_cfg.agent.fix_loop:
        _run()
        _post_export()
        _hooks()
        return

    fr = run_fix_loop(
        "quantize", _phase, config=_fix_config(run_cfg), out_dir=out_dir,
        lesson_store=None, model_id=run_cfg.model,
        scheme=quant_cfg.scheme, method=quant_cfg.method,
        patch_capture=_make_patch_capture(run_cfg, out_dir),
        knowledge=knowledge, replay=run_cfg.agent.knowledge.replay, run_id=run_id,
        allowed_actions=_allowed_actions_block("quantize"),
        **_fix_hooks(run_cfg, session),
    )
    if heal is not None:
        heal.record_fix("quantize", fr)
    if not fr.ok:
        raise AqaError(f"quantize failed after fix-loop: {fr.reason}")


def _run_evaluate(run_cfg: RunConfig, model: str, eval_dir: Path,
                  out_dir: Path, num_gpus: int, knowledge=None,
                  run_id: str = "", phase_name: str = "evaluate",
                  heal: HealTally | None = None,
                  session: "Session | None" = None) -> EvalResult:
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
        if all(b.score == 0 for b in res.benchmarks.values()):
            return PhaseResult(ok=False, log_text="all benchmark scores are 0 (acc=0 => fail)")
        from .hooks import resolve_hooks, run_hook_chain
        hooks = resolve_hooks(run_cfg)
        try:
            for benchmark in res.benchmarks.values():
                run_hook_chain("after_benchmark", hooks, run_cfg, recipe=None,
                               result_or_exception=benchmark, out_dir=eval_dir)
        except AqaError as exc:
            return PhaseResult(ok=False, log_text=str(exc))
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
        allowed_actions=_allowed_actions_block(phase_name),
        **_fix_hooks(run_cfg, session),
    )
    if heal is not None:
        heal.record_fix(phase_name, fr)
    if "result" not in holder:
        raise AqaError(f"evaluate failed after fix-loop: {fr.reason}")
    if not fr.ok:
        _log.warning("evaluate fix-loop unresolved (%s); using last result", fr.reason)
    return holder["result"]


def _current_stack_fingerprint(run_cfg: RunConfig) -> dict[str, str]:
    """Compact quant+eval stack fingerprint for recipe staleness detection.

    Reuses the same package/accelerator probes the env_lock artifact records, so a
    recipe's pinned ``stack_fingerprint`` is comparable byte-for-byte against the
    live stack.  Best-effort: any probe failure yields ``{}`` (no false staleness).
    """
    try:
        from .agent.recipe_kb import _stack_fingerprint
        from .report.manifest import _accelerator, _package_versions
        return _stack_fingerprint({"packages": _package_versions(),
                                   "accelerator": _accelerator(run_cfg)})
    except Exception:  # noqa: BLE001 - fingerprint is advisory, never block a run
        return {}


def _recipe_kb_lookup(run_cfg: RunConfig, out_dir: Path, session: Session,
                      run_id: str) -> "RecipeHit | None":
    """ANALYZE read: find the closest validated recipe for the model-under-quant.

    Independent of ``agent.optimize`` (Hyperloom reads the KB before *every* run,
    not only agent-driven ones): always writes the ``recipe_kb.json`` audit
    (query + hit, or a null hit on a miss) and records the decision, then returns
    the hit so the eval phase can replay its proven eval parity and — when
    agent-optimize is on — a hot-start prompt can be rendered from it.  Any
    failure degrades to ``None`` so the run proceeds exactly as before.
    """
    kb_cfg = run_cfg.agent.recipe_kb
    if not kb_cfg.enabled:
        return None
    try:
        from .agent.recipe_kb import RecipeKB
        from .report.phase_artifacts import _read_model_config
        from .session._io import atomic_write_json

        cfg = _read_model_config(run_cfg.model)
        model_type = cfg.get("model_type")
        scheme = run_cfg.quant.scheme if run_cfg.quant else None
        current_stack = _current_stack_fingerprint(run_cfg)
        hit = RecipeKB.from_config(kb_cfg).lookup(
            run_cfg.model, scheme=scheme, model_type=model_type,
            current_stack=current_stack)
        atomic_write_json(out_dir / "recipe_kb.json", {
            "run_id": run_id,
            "query": {"model": run_cfg.model, "model_type": model_type, "scheme": scheme,
                      "current_stack": current_stack},
            "hit": hit.to_dict() if hit else None,
            "used_as_hot_start": bool(hit and run_cfg.agent.optimize)})
        if hit is None:
            _log.info("recipe_kb: no match for %s (scheme=%s)", run_cfg.model, scheme)
            return None
        _log.info("recipe_kb: match %s (%s)", hit.key, hit.match_reason)
        if hit.stale:
            _log.warning("recipe_kb: ⚠ matched recipe is STALE (stack drifted: %s) — "
                         "treating numbers as approximate (confidence=%.2f)",
                         ", ".join(hit.stale_tools), hit.confidence or 0.0)
        session.event("recipe_kb_hit", key=hit.key, level=hit.match_level,
                      reason=hit.match_reason, source=hit.source, scheme=hit.scheme,
                      stale=hit.stale, confidence=hit.confidence)
        session.state.record_decision(
            Phase.ANALYZE, what=f"recipe KB match: {hit.key}",
            why=hit.match_reason, ref=hit.source_path)
        session.commit()
        return hit
    except Exception as exc:  # noqa: BLE001 - KB is advisory, never block a run
        _log.warning("recipe_kb lookup failed: %s", exc)
        return None


def _recipe_kb_hot_start(hit: "RecipeHit | None", run_cfg: RunConfig) -> str:
    """Render the agent-optimize hot-start prompt from a KB hit (optimize only)."""
    if hit is None or not run_cfg.agent.optimize:
        return ""
    from .agent.recipe_kb import RecipeKB
    return RecipeKB.as_hot_start_context(hit)


def _apply_recipe_eval_parity(run_cfg: RunConfig, hit: "RecipeHit | None",
                              session: Session) -> RunConfig:
    """Bridge KB -> eval: seed the eval口径 from the matched recipe's structured
    ``eval_parity``, filling only fields the user left at their defaults (the run
    config always wins).  This is what makes a validated thinking /
    reasoning-parser口径 auto-apply to the next same-family run instead of being
    rediscovered by hitting the generative-task distortion.
    """
    if hit is None or not isinstance(hit.eval_parity, dict):
        return run_cfg
    ep = hit.eval_parity
    eval_set = run_cfg.eval.model_fields_set
    updates: dict[str, Any] = {}
    rec_thinking = ep.get("thinking")
    if isinstance(rec_thinking, str) and "thinking" not in eval_set:
        updates["thinking"] = rec_thinking
    rec_args = ep.get("thinking_off_model_args")
    if isinstance(rec_args, dict) and rec_args:
        merged = {**rec_args, **run_cfg.eval.thinking_off_model_args}
        if merged != run_cfg.eval.thinking_off_model_args:
            updates["thinking_off_model_args"] = merged
    if not updates:
        return run_cfg
    new_cfg = run_cfg.model_copy(
        update={"eval": run_cfg.eval.model_copy(update=updates)})
    _log.info("recipe_kb: applied eval parity from %s -> %s", hit.key, updates)
    try:
        session.state.record_decision(
            Phase.ANALYZE, what="applied recipe eval parity",
            why=f"from KB {hit.key} ({hit.match_reason})", ref=hit.source_path)
        session.commit()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return new_cfg


def _write_recipe_manifest(run_cfg: RunConfig, result: EvalResult, run_id: str,
                           out_dir: Path, baseline: EvalResult | None,
                           gate_verdict: str | None,
                           gate_reasons: list[str]) -> None:
    """CLOSE: emit the INC-Recipes-compatible ``recipe.yaml`` (RFC 0002, best-effort).

    Projects the quantize run onto the INC model-recipe format plus aqa's
    ``quantization_details`` / ``provenance`` extensions.  Only quantize runs
    have a recipe to export; any failure is logged and never aborts the run.
    """
    try:
        from .quantize import resolve_export_dir
        from .report.recipe_manifest import build_recipe_manifest, write_recipe_manifest

        quant_dir: Path | None = None
        try:
            cand = resolve_export_dir(out_dir / "quantized")
            if (cand / "config.json").is_file():
                quant_dir = cand
        except Exception:  # noqa: BLE001 - export layout probe is best-effort
            quant_dir = None
        manifest = build_recipe_manifest(
            run_cfg=run_cfg, result=result, run_id=run_id, out_dir=out_dir,
            quant_dir=quant_dir, baseline=baseline, gate_verdict=gate_verdict,
            gate_reasons=gate_reasons)
        if manifest is not None:
            write_recipe_manifest(out_dir, manifest)
    except Exception as exc:  # noqa: BLE001 - recipe export must never abort a run
        _log.warning("recipe.yaml emit skipped: %s", exc)


def _eval_parity_from_cfg(eval_cfg) -> dict[str, Any]:
    """Structured, replayable eval口径 pinned alongside a validated recipe so the
    next same-family run can auto-apply it (see :func:`_apply_recipe_eval_parity`)."""
    toma = dict(eval_cfg.thinking_off_model_args or {})
    parity: dict[str, Any] = {
        "thinking": eval_cfg.thinking,
        "thinking_off_model_args": toma,
        "generative_tasks": list(eval_cfg.generative_tasks),
    }
    bits = ", ".join(f"{k}={v}" for k, v in toma.items())
    note = f"thinking={eval_cfg.thinking}"
    if bits:
        note += f"; thinking-off model_args: {bits}"
    if eval_cfg.generative_tasks:
        note += f"; generative_tasks={eval_cfg.generative_tasks}"
    parity["note"] = note
    return parity


def _recipe_kb_pin(run_cfg: RunConfig, out_dir: Path, result: EvalResult,
                   gate_verdict: str | None, run_id: str,
                   baseline: "EvalResult | None" = None,
                   gate_reasons: list[str] | None = None) -> None:
    """CLOSE record-back: pin the validated recipe on GO, else record negative
    knowledge (what failed) so the next run is warned.

    A CLOSE *always* records (Hyperloom records both outcomes): GO flows through
    the strict :meth:`RecipeKB.pin` (validated schemes + trust + session row);
    anything else flows through :meth:`RecipeKB.record_outcome`, which appends a
    structured failure + session row WITHOUT ever writing validated schemes.

    The KB root defaults to the code checkout (``recipes/quant/auto``).  When
    that checkout is read-only (common on shared NFS), fall back to
    ``<run>/recipe_kb/auto`` and record the outcome path in ``recipe_kb.json``
    instead of silently dropping knowledge.
    """
    kb_cfg = run_cfg.agent.recipe_kb
    if not (run_cfg.provider.type == "quantize" and kb_cfg.pin and run_cfg.quant):
        return
    import json as _json

    from .session._io import atomic_write_json, now_iso

    def _record(**payload) -> None:
        path = out_dir / "recipe_kb.json"
        try:
            data = _json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if not isinstance(data, dict):
                data = {}
            data["record"] = {"at": now_iso(), **payload}
            atomic_write_json(path, data)
        except Exception as exc:  # noqa: BLE001 - provenance must not abort CLOSE
            _log.warning("recipe_kb: could not record outcome state: %s", exc)

    if gate_verdict is None:
        # No gate was configured: the run has measurements but no GO/NO-GO
        # verdict. Recording it as FAILED poisons the KB with a false negative.
        _log.info("recipe_kb: no gate configured; skipping outcome record-back")
        _record(status="skipped", reason="no gate configured")
        return
    try:
        from .agent.recipe_kb import (
            OUTCOME_FAILED, OUTCOME_INCONCLUSIVE, OUTCOME_NO_GO, RecipeKB,
        )
        from .report.phase_artifacts import _read_model_config

        cfg = _read_model_config(run_cfg.model)
        archs = cfg.get("architectures")
        architecture = archs[0] if isinstance(archs, list) and archs else None
        model_type = cfg.get("model_type")
        scheme = run_cfg.quant.scheme
        mean_after = result.mean_score()
        mean_before = baseline.mean_score() if baseline is not None else None
        fp = _current_stack_fingerprint(run_cfg)
        kb = RecipeKB.from_config(kb_cfg)
        if (gate_verdict or "").upper() == "GO":
            cmd_path = out_dir / ".quant_command.json"
            if not cmd_path.is_file():
                _record(status="skipped", reason="quant command not recorded")
                return
            command = _json.loads(cmd_path.read_text()).get("command")
            if not command:
                _record(status="skipped", reason="quant command empty")
                return
            eval_parity = _eval_parity_from_cfg(run_cfg.eval)
            kwargs = dict(
                model=run_cfg.model, scheme=scheme, command=command,
                model_type=model_type, architecture=architecture,
                ignore_layers=run_cfg.quant.ignore_layers,
                result={"mean_score": mean_after,
                        "results": result.to_dict().get("results", {})},
                eval_parity=eval_parity, lessons=[f"eval parity: {eval_parity['note']}"],
                gate_verdict=gate_verdict, source_run=run_id,
                stack_fingerprint=fp, mean_before=mean_before, mean_after=mean_after)

            def _invoke(store: RecipeKB) -> str | None:
                return store.pin(**kwargs)
        else:
            verdict = gate_verdict.upper()
            outcome = (OUTCOME_NO_GO if verdict == "NO-GO"
                       else OUTCOME_INCONCLUSIVE if verdict == "INCONCLUSIVE"
                       else OUTCOME_FAILED)
            kwargs = dict(
                model=run_cfg.model, scheme=scheme, outcome=outcome,
                model_type=model_type, architecture=architecture,
                result={"mean_score": mean_after},
                mean_before=mean_before, mean_after=mean_after,
                reasons=gate_reasons, source_run=run_id, stack_fingerprint=fp)

            def _invoke(store: RecipeKB) -> str | None:
                return store.record_outcome(**kwargs)

        try:
            path = _invoke(kb)
        except OSError as exc:
            fallback = RecipeKB(
                curated_root=out_dir / "recipe_kb",
                auto_root=out_dir / "recipe_kb" / "auto")
            try:
                path = _invoke(fallback)
            except Exception as fallback_exc:  # noqa: BLE001
                _record(status="failed", error=str(fallback_exc),
                        primary_error=str(exc))
                _log.warning("recipe_kb record failed (primary=%s; fallback=%s)",
                             exc, fallback_exc)
                return
            _record(status="fallback", path=path, primary_error=str(exc))
            _log.warning("recipe_kb: code checkout not writable (%s); wrote "
                         "fallback KB -> %s", exc, path)
            return
        if path:
            _record(status="ok", path=path)
            if (gate_verdict or "").upper() == "GO":
                _log.info("recipe_kb: pinned validated recipe -> %s", path)
            else:
                _log.info("recipe_kb: recorded %s negative knowledge -> %s",
                          gate_verdict, path)
    except Exception as exc:  # noqa: BLE001 - record-back is best-effort provenance
        _record(status="failed", error=str(exc))
        _log.warning("recipe_kb record failed: %s", exc)


def _resolve_model(run_cfg: RunConfig, out_dir: Path, num_gpus: int, knowledge=None,
                   run_id: str = "", heal: HealTally | None = None,
                   hot_start: str = "", session: "Session | None" = None):
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
            run_cfg, quant_dir, out_dir, num_gpus, knowledge, run_id, heal=heal,
            hot_start=hot_start, session=session),
    )
    return provider.provide(run_cfg, ctx)


@dataclass
class _RequantController:
    """Pure control logic for the bounded re-QUANT loop-back (RFC §S3).

    Mirrors Hyperloom's macro-cycle reloop guardrails (``MAX_MACRO_CYCLES`` +
    ``GLOBAL_CONVERGENCE_NO_GAIN_CYCLES``): it owns the candidate cursor, the
    best-score-so-far, and the stop rule (GO reached | ``max_cycles`` |
    candidates exhausted | no-gain convergence). Kept IO-free so the loop *policy*
    is unit-testable without a GPU; the orchestrator feeds it one cycle outcome at
    a time via :meth:`observe`.
    """

    n_candidates: int
    max_cycles: int
    min_gain: float = 0.0
    no_gain_patience: int = 1
    best_score: float = float("-inf")  # score to beat (seeded with the base run's)
    cursor: int = 0
    cycles: int = 0
    no_gain: int = 0
    stopped: bool = False
    stop_reason: str = ""

    def next_index(self) -> int | None:
        """Index of the next candidate to try, or ``None`` when the loop must stop."""
        if self.stopped:
            return None
        if self.cycles >= self.max_cycles:
            self.stopped, self.stop_reason = True, "max_cycles"
            return None
        if self.cursor >= self.n_candidates:
            self.stopped, self.stop_reason = True, "candidates_exhausted"
            return None
        return self.cursor

    def observe(self, score: float, verdict: str) -> bool:
        """Record a completed cycle; return ``True`` iff it is the new best.

        Updates the cursor/best/no-gain state and arms the stop condition (GO, or
        ``no_gain_patience`` consecutive cycles that fail to improve the best score
        by at least ``min_gain``).
        """
        self.cycles += 1
        self.cursor += 1
        improved = score >= self.best_score + self.min_gain
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
        if verdict == "GO":
            self.stopped, self.stop_reason = True, "go"
        elif improved:
            self.no_gain = 0
        else:
            self.no_gain += 1
            if self.no_gain >= self.no_gain_patience:
                self.stopped, self.stop_reason = True, "no_gain_convergence"
        return is_best


def _promote_cycle(out_dir: Path, cycle_dir: Path) -> None:
    """Promote a winning re-QUANT cycle's artifacts to the canonical run paths.

    Downstream (sinks, manifest, pin-back, ``resolve_export_dir``) all read the
    canonical ``out_dir/quantized`` + ``out_dir/eval``; the loop-back keeps each
    cycle in ``out_dir/requant/cycle_<n>/`` and only the winner is moved into
    place, so a losing cycle never becomes the baseline.
    """
    import shutil

    for name in ("quantized", "eval"):
        src = cycle_dir / name
        if not src.exists():
            continue
        dst = out_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    for fname in (".quant_command.json", "quant_plan.json"):
        src = cycle_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)


def _run_requant_loop(run_cfg: RunConfig, out_dir: Path, num_gpus: int, *,
                      baseline: "EvalResult | None", base_result: EvalResult,
                      base_resolved, base_verdict: str | None,
                      base_reasons: list[str], knowledge, run_id: str,
                      heal: "HealTally | None", session: Session):
    """Bounded ``GATE -> QUANT`` re-QUANT loop-back (RFC §S3).

    Runs only when ``gate.requant.enabled`` and the first-pass verdict is not GO.
    Each cycle re-quantizes with the next recipe-sweep candidate into an isolated
    ``out_dir/requant/cycle_<n>/``, re-EVALs and re-GATEs, keeping the best
    ``mean_score``; the winner (if it beats the base run) is promoted to the
    canonical paths. Returns the winning
    ``(result, resolved, verdict, reasons)`` — the base tuple unchanged when no
    cycle improves on it. Opt-in + default-off, so single-pass behaviour is intact.
    """
    from .agent.optimize import analyze_model
    from .gate import evaluate_gate
    from .quantize import resolve_export_dir
    from .quantize.candidates import apply_candidate, resolve_candidates
    from .report.phase_artifacts import write_accuracy_json

    rq = run_cfg.gate.requant
    base_score = base_result.mean_score()
    try:
        analysis = analyze_model(run_cfg.model)
        candidates = resolve_candidates(analysis, run_cfg.quant)
    except Exception as exc:  # noqa: BLE001 - no candidates => keep base, no loop
        _log.warning("requant: candidate resolution failed (%s); keeping base run", exc)
        return base_result, base_resolved, base_verdict, base_reasons

    base_dump = run_cfg.quant.model_dump()
    fresh = [c for c in candidates if apply_candidate(run_cfg.quant, c).model_dump() != base_dump]
    if not fresh:
        _log.info("requant: no recipe candidate differs from the pinned plan; keeping base")
        return base_result, base_resolved, base_verdict, base_reasons

    session.event("requant_start", trigger_verdict=base_verdict, base_score=base_score,
                  max_cycles=rq.max_cycles, candidates=[c.name for c in fresh])
    ctrl = _RequantController(n_candidates=len(fresh), max_cycles=rq.max_cycles,
                             min_gain=rq.min_gain, no_gain_patience=rq.no_gain_patience,
                             best_score=base_score)
    best = {"result": base_result, "resolved": base_resolved,
            "verdict": base_verdict, "reasons": base_reasons, "cycle_dir": None}

    while True:
        idx = ctrl.next_index()
        if idx is None:
            break
        cand = fresh[idx]
        n = ctrl.cycles + 1
        cycle_dir = out_dir / "requant" / f"cycle_{n}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        cycle_cfg = _apply_candidate_cfg(run_cfg, cand)

        # GATE -> QUANT: the one sanctioned loop-back edge (guard records it).
        session.guard_action("requantize", Phase.QUANT, candidate=cand.name, cycle=n)
        session.enter(Phase.QUANT)
        _log.info("== requant cycle %d: recipe '%s' ==", n, cand.name)
        cycle_quant_dir = cycle_dir / "quantized"
        try:
            _run_quantize(cycle_cfg, cycle_quant_dir, cycle_dir, num_gpus, knowledge,
                          run_id, heal=heal, session=session)
        except AqaError as exc:
            session.fail(Phase.QUANT, error=str(exc))
            session.event("requant_cycle", cycle=n, candidate=cand.name,
                          status="quant_failed", error=str(exc)[:300])
            ctrl.observe(float("-inf"), "ERROR")
            continue
        session.done(Phase.QUANT)

        cycle_model = str(resolve_export_dir(cycle_quant_dir))
        session.enter(Phase.EVAL)
        cycle_eval_dir = cycle_dir / "eval"
        cycle_result = _run_evaluate(cycle_cfg, cycle_model, cycle_eval_dir, cycle_dir,
                                     num_gpus, knowledge, run_id, heal=heal, session=session)
        write_accuracy_json(cycle_result, cycle_eval_dir)
        session.done(Phase.EVAL)

        session.enter(Phase.GATE)
        decision = evaluate_gate(cycle_result, baseline, run_cfg.gate)
        (cycle_dir / "gate.json").write_text(
            __import__("json").dumps(
                {"verdict": decision.verdict, "reasons": decision.reasons}, indent=2))
        session.done(Phase.GATE)

        score = cycle_result.mean_score()
        is_best = ctrl.observe(score, decision.verdict)
        session.event("requant_cycle", cycle=n, candidate=cand.name, status="done",
                      mean_score=score, verdict=decision.verdict, is_best=is_best)
        session.state.record_decision(
            Phase.QUANT, what=f"requant cycle {n}: recipe '{cand.name}'",
            why=f"mean_score={score:.4f} verdict={decision.verdict} "
                f"(base={base_score:.4f}); {'NEW BEST' if is_best else 'kept prior best'}",
            ref=str(cycle_dir))
        if is_best:
            best = {"result": cycle_result, "resolved": _resolved_path(cycle_model, run_cfg),
                    "verdict": decision.verdict, "reasons": list(decision.reasons),
                    "cycle_dir": cycle_dir}
        session.commit()

    promoted = best["cycle_dir"] is not None
    if promoted:
        _promote_cycle(out_dir, best["cycle_dir"])
        best["resolved"] = _resolved_path(str(resolve_export_dir(out_dir / "quantized")), run_cfg)
    (out_dir / "requant" / ".requant_done").write_text(ctrl.stop_reason or "done")
    session.event("requant_stop", reason=ctrl.stop_reason, cycles=ctrl.cycles,
                  promoted=promoted, best_score=ctrl.best_score,
                  winning_verdict=best["verdict"])
    _log.info("== requant done: %d cycle(s), %s, best_score=%.4f, verdict=%s ==",
              ctrl.cycles, "promoted a cycle" if promoted else "kept base run",
              ctrl.best_score, best["verdict"])
    return best["result"], best["resolved"], best["verdict"], best["reasons"]


def _apply_candidate_cfg(run_cfg: RunConfig, cand) -> RunConfig:
    """``run_cfg`` with the candidate recipe applied (+ any per-candidate eval args)."""
    from .quantize.candidates import apply_candidate

    cfg = run_cfg.model_copy(update={"quant": apply_candidate(run_cfg.quant, cand)})
    if cand.eval_model_args:
        merged = {**run_cfg.eval.lm_eval.model_args, **cand.eval_model_args}
        new_lm = run_cfg.eval.lm_eval.model_copy(update={"model_args": merged})
        cfg = cfg.model_copy(update={"eval": run_cfg.eval.model_copy(update={"lm_eval": new_lm})})
    return cfg


def _resolved_path(model_dir: str, run_cfg: RunConfig):
    from .providers.base import ResolvedModel

    return ResolvedModel(
        model=model_dir, kind="path",
        metadata={"scheme": run_cfg.quant.scheme if run_cfg.quant else None,
                  "method": run_cfg.quant.method if run_cfg.quant else None})


def build_phase_plan(run_cfg: RunConfig) -> list[Phase]:
    """The ordered, run-specific phase sequence ``run_pipeline`` will execute.

    Trimmed from the full :class:`Phase` set by config, exactly mirroring the
    branches the linear pipeline already takes (baseline on/off, quantize vs
    pretrained/endpoint provider, gate configured, upload configured).
    """
    plan: list[Phase] = []
    if run_cfg.eval.baseline:
        plan.append(Phase.BASELINE)
    if run_cfg.provider.type == "quantize":
        plan.append(Phase.QUANT)
    plan.append(Phase.EVAL)
    if run_cfg.gate is not None:
        plan.append(Phase.GATE)
    plan.append(Phase.CLOSE)
    if run_cfg.reproduce is not None:
        plan.append(Phase.VERIFY)
    if run_cfg.provider.type == "quantize" and run_cfg.provider.upload is not None:
        plan.append(Phase.PUBLISH)
    return plan


def _quant_checkpoint_present(out_dir: Path) -> bool:
    """True when ``quantized/`` already holds a resolvable exported checkpoint."""
    from .quantize import resolve_export_dir

    quant_dir = out_dir / "quantized"
    if not quant_dir.exists():
        return False
    return (resolve_export_dir(quant_dir) / "config.json").exists()


def _accuracy_present(eval_dir: Path) -> bool:
    return (eval_dir / "accuracy.json").exists()


def _score_summary(result: EvalResult) -> dict[str, Any]:
    return {"mean_score": result.mean_score(),
            "per_task": {n: b.score for n, b in result.benchmarks.items()}}


def run_pipeline(run_cfg: RunConfig, out_dir: Path, *, resume: bool = False) -> ProcessState:
    """Execute the full pipeline as a resumable phase sequence; returns terminal
    ProcessState.

    Each phase is wrapped as ``enter -> run -> done`` against a durable
    ``session/`` (``manifest.json`` + ``state.json`` + ``events.jsonl``). When
    ``resume=True`` a phase whose product is already on disk is skipped instead of
    recomputed (re-quantizing after an eval crash was the headline cost); a
    half-baked product re-runs the phase.

    On a phase failure (quantize/evaluate raises after the fix-loop) the run's
    artifacts are still written back: eval logs, agent fix records and captured
    patches are uploaded via the configured sinks with a failure ``status`` (so a
    crashed run is observable on the leaderboard instead of silently vanishing),
    and if a valid quantized checkpoint exists it is still pushed to its model
    repo. The original error is then re-raised so the caller still sees rc!=0.
    """
    from .providers.base import ResolvedModel
    from .quantize import resolve_export_dir

    num_gpus = _num_gpus(run_cfg)
    run_id = _make_run_id(run_cfg)
    baseline: EvalResult | None = None
    resolved = None
    eval_dir = out_dir / "eval"
    heal = HealTally()
    # Shared fix-knowledge (lessons + replayable patches); synced down once here.
    knowledge = _make_knowledge(run_cfg)

    plan = build_phase_plan(run_cfg)
    session = open_session(out_dir, run_cfg, run_id, plan, resume=resume)
    if resume:
        # Resume must be byte-stable: state/manifest/events keep the original
        # run_id, so every artifact written after this point must reuse it too.
        # Generating a fresh id here produced three ids in one run directory
        # (state vs env_lock vs report) and broke provenance checks.
        run_id = session.state.run_id
    is_quantize = run_cfg.provider.type == "quantize"

    # Per-phase execution留痕: tee the run logger to logs/pipeline.log (always on,
    # regardless of agent/executor), so every run has a readable phase trace.
    run_log = attach_run_log(out_dir)
    # Full-console留痕: tee fds 1/2 (logger stderr + subprocess stdout — lm_eval
    # tables, auto-round progress) to logs/run.console.log, the single readable
    # transcript of the whole run.  Best-effort (None under captured stdio).
    console_tee = start_console_tee(out_dir)

    # P2 ENV/ANALYZE artifacts: deterministic, no GPU, written once at run start
    # so inspect can validate them regardless of where a later phase crashes.
    write_env_lock(out_dir, run_cfg, run_id)
    # ANALYZE read: consult the Recipe KB (independent of agent.optimize) and
    # bridge its validated eval parity into the eval config BEFORE baseline, so a
    # proven thinking/reasoning-parser口径 auto-applies instead of being
    # rediscovered. The hit is reused later for the agent-optimize hot-start.
    recipe_hit = _recipe_kb_lookup(run_cfg, out_dir, session, run_id) if is_quantize else None
    run_cfg = _apply_recipe_eval_parity(run_cfg, recipe_hit, session)
    write_analysis(out_dir, run_cfg, run_id)

    try:
        # ── optional bf16 baseline evaluation ────────────────────────────────
        if run_cfg.eval.baseline:
            base_dir = out_dir / "baseline"
            if session.should_skip(Phase.BASELINE, _accuracy_present(base_dir)):
                _log.info("== resume: baseline present, skipping ==")
                baseline = read_eval_result(base_dir)
                session.skip(Phase.BASELINE, reason="baseline/accuracy.json present")
            else:
                session.enter(Phase.BASELINE)
                _log.info("== baseline eval (bf16) ==")
                baseline = _run_evaluate(run_cfg, run_cfg.model, base_dir, out_dir,
                                         num_gpus, knowledge, run_id,
                                         phase_name="baseline", heal=heal,
                                         session=session)
                write_accuracy_json(baseline, base_dir)
                _dispatch_sinks(run_cfg, base_dir, RunMeta(
                    model=run_cfg.model, artifact_name=_artifact_name(run_cfg, "baseline"),
                    org_override=_report_org(run_cfg)))
                session.state.baseline = _score_summary(baseline)
                session.done(Phase.BASELINE)

        # ── resolve the eval target (provider: pretrained|quantize|endpoint) ──
        if is_quantize and session.should_skip(Phase.QUANT, _quant_checkpoint_present(out_dir)):
            # Reuse the existing checkpoint without re-running quantization — this
            # is the expensive phase resume exists to save.
            _log.info("== resume: quantized checkpoint present, skipping quantize ==")
            model_dir = resolve_export_dir(out_dir / "quantized")
            resolved = ResolvedModel(
                model=str(model_dir), kind="path",
                metadata={"scheme": run_cfg.quant.scheme if run_cfg.quant else None,
                          "method": run_cfg.quant.method if run_cfg.quant else None})
            session.skip(Phase.QUANT, reason="quantized checkpoint present")
        else:
            hot_start = ""
            if is_quantize:
                session.enter(Phase.QUANT)
                hot_start = _recipe_kb_hot_start(recipe_hit, run_cfg)
            resolved = _resolve_model(run_cfg, out_dir, num_gpus, knowledge, run_id,
                                      heal=heal, hot_start=hot_start, session=session)
            if is_quantize:
                session.state.current_best = {
                    "scheme": run_cfg.quant.scheme if run_cfg.quant else None,
                    "model_dir": resolved.model}
                session.done(Phase.QUANT)
        eval_model = resolved.model

        # P2 QUANT artifact: freeze the recipe + the actual command (both the
        # fresh-run and resume-skip branches have .quant_command.json by now).
        if is_quantize:
            write_quant_plan(out_dir, run_cfg, run_id)

        # ── evaluation of the (provided) model ───────────────────────────────
        if session.should_skip(Phase.EVAL, _accuracy_present(eval_dir)):
            _log.info("== resume: eval present, skipping ==")
            result = read_eval_result(eval_dir)
            session.skip(Phase.EVAL, reason="eval/accuracy.json present")
        else:
            session.enter(Phase.EVAL)
            _log.info("== evaluate ==")
            result = _run_evaluate(run_cfg, eval_model, eval_dir, out_dir, num_gpus,
                                   knowledge, run_id, heal=heal, session=session)
            write_accuracy_json(result, eval_dir)
            session.state.current_best = {
                **(session.state.current_best or {}),
                "mean_score": result.mean_score()}
            session.done(Phase.EVAL)

        # ── gate ─────────────────────────────────────────────────────────────
        gate_verdict: str | None = None
        gate_reasons: list[str] = []
        if run_cfg.gate is not None:
            session.enter(Phase.GATE)
            decision = evaluate_gate(result, baseline, run_cfg.gate)
            gate_verdict = decision.verdict
            gate_reasons = list(decision.reasons)
            (out_dir / "gate.json").write_text(
                __import__("json").dumps({"verdict": decision.verdict, "reasons": decision.reasons},
                                         indent=2))
            _log.info("gate: %s (%s)", decision.verdict, "; ".join(decision.reasons))
            session.state.gate = {"verdict": decision.verdict, "reasons": decision.reasons}
            session.done(Phase.GATE)

            # ── bounded GATE->QUANT re-QUANT loop-back (RFC §S3; opt-in) ──────
            rq = run_cfg.gate.requant
            requant_done = (out_dir / "requant" / ".requant_done").exists()
            if rq.enabled and is_quantize and gate_verdict != "GO" and not requant_done:
                result, resolved, gate_verdict, gate_reasons = _run_requant_loop(
                    run_cfg, out_dir, num_gpus, baseline=baseline, base_result=result,
                    base_resolved=resolved, base_verdict=gate_verdict,
                    base_reasons=gate_reasons, knowledge=knowledge, run_id=run_id,
                    heal=heal, session=session)
                eval_model = resolved.model
                write_accuracy_json(result, eval_dir)
                session.state.gate = {"verdict": gate_verdict, "reasons": gate_reasons}
                session.state.current_best = {
                    **(session.state.current_best or {}),
                    "mean_score": result.mean_score(), "model_dir": eval_model}
                session.commit()

        results = {n: b.score for n, b in result.benchmarks.items()}

        # ── close: reproducibility manifest (model + config + env + results) ──
        session.enter(Phase.CLOSE)
        # P2 EVAL artifact: source(bf16) vs quant scores + per-benchmark gap.
        write_eval_report(out_dir, run_cfg, run_id, result, baseline,
                          gate_verdict, gate_reasons)
        # P3 record-back: pin on GO, else record negative knowledge (always records).
        _recipe_kb_pin(run_cfg, out_dir, result, gate_verdict, run_id,
                       baseline=baseline, gate_reasons=gate_reasons)
        manifest = build_manifest(
            run_cfg, result, run_id=run_id, baseline=baseline,
            gate_verdict=gate_verdict, out_dir=out_dir, self_heal=heal.block())
        write_manifest(manifest, out_dir)
        # RFC 0002: INC-Recipes-compatible recipe.yaml (+ aqa extensions).
        if is_quantize:
            _write_recipe_manifest(run_cfg, result, run_id, out_dir, baseline,
                                   gate_verdict, gate_reasons)

        # Hard-flow step ④: the publishable model card is a run artifact, generated
        # BEFORE the dataset sink so it rides the same upload as analysis/report.
        _ensure_model_card(run_cfg, result, baseline, out_dir, resolved)
        _dispatch_sinks(run_cfg, eval_dir, RunMeta(
            model=run_cfg.model,
            scheme=run_cfg.quant.scheme if run_cfg.quant else None,
            method=run_cfg.quant.method if run_cfg.quant else None,
            artifact_name=_artifact_name(run_cfg, "quant"),
            org_override=_report_org(run_cfg),
            run_root=out_dir,
            run_id=run_id,
            status=_derive_status(results, gate_verdict),
            extra={"results": results},
            knowledge_dir=knowledge.local_dir if knowledge else None,
        ))
        session.done(Phase.CLOSE)

        # ── RFC 0002 §5: read-only reproduction verification (opt-in) ─────────
        if Phase.VERIFY in plan:
            session.enter(Phase.VERIFY)
            try:
                from .reproduce import verify_and_write

                report = verify_and_write(run_cfg, out_dir, result, baseline)
                verdict = report.get("verdict", "inconclusive")
                _log.info("verify: %s (%s)", verdict, out_dir / "reproduction_report.json")
                session.event("reproduction_report", verdict=verdict,
                              path=str(out_dir / "reproduction_report.json"))
                session.done(Phase.VERIFY, verdict=verdict)
            except Exception as exc:  # noqa: BLE001 - verify must not hide a run
                _log.warning("VERIFY failed: %s", exc)
                session.fail(Phase.VERIFY, error=str(exc))

        # ── optional: upload the quantized checkpoint to a HF model repo ──────
        if Phase.PUBLISH in plan:
            session.enter(Phase.PUBLISH)
        _maybe_upload_model(run_cfg, resolved, result, out_dir, baseline=baseline)
        if Phase.PUBLISH in plan:
            session.done(Phase.PUBLISH)
        _finalize_knowledge(knowledge)
        return ProcessState.DONE
    except AqaError as exc:
        _fail_current_phase(session, exc)
        _writeback_on_failure(run_cfg, out_dir, eval_dir, resolved, exc, knowledge,
                              run_id=run_id, heal=heal, failed_phase=session.state.phase)
        _finalize_knowledge(knowledge)
        raise
    finally:
        detach_run_log(run_log)
        stop_console_tee(console_tee)


def _fail_current_phase(session: Session, exc: Exception) -> None:
    """Mark whatever phase was RUNNING as FAILED before unwinding (best-effort)."""
    try:
        session.fail(Phase(session.state.phase), error=str(exc))
    except Exception:  # noqa: BLE001 - never mask the real failure
        pass


# Write-back status label per failed phase.  Driven by the session state machine
# (which phase was RUNNING when we crashed) rather than by string-matching the
# exception message -- so e.g. a QUANT crash is never mislabelled "Eval Failed".
_PHASE_FAIL_STATUS = {
    "BASELINE": "Baseline Failed",
    "QUANT": "Quant Failed",
    "EVAL": "Eval Failed",
    "GATE": "Gate Failed",
    "PUBLISH": "Publish Failed",
    "CLOSE": "Eval Failed",
}


def _failure_status(failed_phase: str | None, reason: str) -> str:
    """Derive the leaderboard status label for a failed run.

    Prefer the session's failed phase; fall back to the legacy message heuristic
    only when the phase is unknown (defensive -- the session should always set it).
    """
    if failed_phase:
        return _PHASE_FAIL_STATUS.get(failed_phase.upper(), "Run Failed")
    return "Quant Failed" if reason.startswith("quantize failed") else "Eval Failed"


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
                          *, run_id: str = "", heal: HealTally | None = None,
                          failed_phase: str | None = None) -> None:
    """Best-effort write-back of a *failed* run (results/logs/patches + status).

    Mirrors the success-path sink dispatch so a crashed run is still uploaded and
    marked failed on the leaderboard.  If quantization succeeded but evaluation
    failed, a valid checkpoint may still exist -- it is uploaded too.  All errors
    here are swallowed (we are already unwinding a failure and must re-raise).
    """
    reason = str(exc)
    status = _failure_status(failed_phase, reason)
    run_dir = eval_dir if eval_dir.exists() else out_dir
    _log.warning("run failed (%s); writing back partial results + status=%s", reason, status)
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
            org_override=_report_org(run_cfg),
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
            _maybe_upload_model(run_cfg, resolved, empty, out_dir,
                                allow_generate=False)
        except Exception as up_exc:  # noqa: BLE001 - best-effort
            _log.warning("failure-path model upload error: %s", up_exc)


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


def _ensure_model_card(run_cfg: RunConfig, result: EvalResult,
                       baseline: EvalResult | None, out_dir: Path, resolved) -> None:
    """Generate ``model_card.md`` before sinks for every quantize run with eval.

    A publishing run uses ``upload.signoff`` for a real card; any other quantize
    run gets a **draft** with the template's placeholders (``{member name}`` /
    ``{base_repo_id}``), so every run with evaluation accuracies carries a card.
    ``upload.model_card=false`` is the legacy opt-out.  An existing card is kept
    unless a newly supplied publish sign-off makes it invalid.
    """
    cfg = run_cfg.provider.upload
    if run_cfg.provider.type != "quantize" or getattr(resolved, "kind", None) != "path":
        return
    if cfg is not None and not cfg.model_card:
        return
    if not result.benchmarks:
        return
    card_path = out_dir / "model_card.md"
    signoff = cfg.signoff if cfg is not None else None
    if card_path.is_file():
        from .report.model_card import signoff_problem

        if not (signoff and signoff_problem(card_path.read_text(errors="replace"))):
            return
        # Existing draft + a now-confirmed sign-off -> regenerate the real card.
    from .report.model_card import write_model_card

    try:
        write_model_card(
            run_cfg, out_dir=out_dir,
            repo_name=(cfg.repo_name if cfg is not None and cfg.repo_name
                       else _artifact_name(run_cfg, "quant")),
            result=result, baseline=baseline, quant_dir=Path(resolved.model),
            signoff=signoff)
        _log.info("model card: wrote %s", card_path)
    except Exception as exc:  # noqa: BLE001 - the upload step will skip and warn
        _log.warning("model card not generated: %s", exc)


def _maybe_upload_model(run_cfg: RunConfig, resolved, result: EvalResult, out_dir: Path,
                        *, baseline: EvalResult | None = None,
                        allow_generate: bool = True) -> None:
    """Push the quantized checkpoint to a HF *model* repo on success.

    Only runs for the ``quantize`` provider with an ``upload:`` block.  The
    publishable ``model_card.md`` is a hard-flow artifact: it is generated from
    this run's artifacts (never a template example) and must carry the
    operator's explicitly supplied sign-off.  Without a valid card the upload is
    skipped rather than publishing an unsigned README.  A failed upload is
    warned about but never aborts the (already successful) run.
    """
    if run_cfg.provider.type != "quantize" or run_cfg.provider.upload is None:
        return
    if resolved.kind != "path":
        return
    model_dir = Path(resolved.model)
    if not model_dir.is_dir():
        _log.warning("model upload skipped: quantized dir missing (%s)", model_dir)
        return

    from .report.model_card import CARD_FILE, signoff_problem, write_model_card

    cfg = run_cfg.provider.upload
    card_path = out_dir / CARD_FILE
    if not cfg.model_card:
        # Explicit opt-out keeps the historic auto-generated README.  The hard-flow
        # artifact check still flags a publishing run without a signed card.
        _log.warning(
            "model upload: provider.upload.model_card=false; using the legacy "
            "auto-generated README (no signed model_card.md)")
    elif card_path.is_file():
        problem = signoff_problem(card_path.read_text(errors="replace"))
        if problem:
            _log.warning("model upload skipped: %s", problem)
            return
    elif allow_generate:
        try:
            write_model_card(
                run_cfg, out_dir=out_dir, repo_name=cfg.repo_name or
                _artifact_name(run_cfg, "quant"), result=result,
                baseline=baseline, quant_dir=model_dir, signoff=cfg.signoff)
            _log.info("model card: wrote %s", card_path)
        except Exception as exc:  # noqa: BLE001 - never abort the run
            _log.warning("model upload skipped: model card generation failed: %s", exc)
            return
    else:
        _log.warning(
            "model upload skipped: no signed model_card.md exists for this "
            "failed run (generate one with provider.upload.signoff and re-run)")
        return

    from .report.sinks import HfModelSink

    sink = HfModelSink(
        repo=cfg.repo,
        repo_name=cfg.repo_name,
        token_env=cfg.token_env,
        private=cfg.private,
        endpoint=cfg.endpoint,
        disable_xet=cfg.disable_xet,
        model_card=cfg.model_card,
    )
    from .report.names import scheme_display

    meta = RunMeta(
        # The generated card puts this into `base_model:`, which HF validates as an
        # `org/name` repo id. `model:` is usually a local checkpoint path here, and that
        # makes the whole upload fail ("Invalid metadata in README.md").
        # `output.report_model` is the canonical `Org/Model` id for exactly this case.
        model=run_cfg.output.report_model or run_cfg.model,
        # The card's Scheme line is human-facing prose, so show the *effective* label
        # (MXFP4-Mixed for a MXFP8 base whose experts drop to 4 bit), not the base
        # recipe knob — same value the artifact/repo name leads with.
        scheme=scheme_display(run_cfg.quant) if run_cfg.quant else None,
        method=run_cfg.quant.method if run_cfg.quant else None,
        artifact_name=cfg.repo_name or _artifact_name(run_cfg, "quant"),
        org_override=_report_org(run_cfg),
        # HfModelSink._card_commands_versions() reads run_manifest.json from
        # meta.run_root; without it the card loses its mandatory
        # "Reproduce & Environment" section (the actual auto-round + lm_eval
        # commands and the package pins — e.g. which vLLM build a CT artifact needs).
        run_root=out_dir,
        extra={"results": {n: b.score for n, b in result.benchmarks.items()}},
    )
    try:
        url = sink.upload(model_dir, meta)
        _log.info("== model uploaded == %s", url)
    except Exception as exc:  # noqa: BLE001 - upload must not abort a successful run
        _log.warning("model upload failed: %s", exc)

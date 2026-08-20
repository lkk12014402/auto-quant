"""Agent fix-loop state machine.

Ports ``agent_fix_loop`` from ``local_dispatch/agent/agent_fix_loop.sh`` into
Python.  The *orchestration* (tier escalation, drift detection, verdict handling,
progress override, lessons) lives here; the *execution* of a phase and the *agent
CLI call* are injected callables so the loop is fully unit-testable and reusable
by any phase (quantize / evaluate).

Escalation triggers, faithful to the original:
  * ``VERDICT: UNFIXABLE`` -> escalate if a stronger tier exists, else give up.
  * ``VERDICT: ESCALATE``  -> escalate if a stronger tier exists.
  * drift (same error_class ``drift_threshold`` times in a row) -> escalate/abort.
  * per-tier ``tier_attempt_budget`` exhausted -> fail-safe escalate.
  * global ``max_attempts`` (auto-raised to tiers x budget) -> abort.
Deeper quant progress (a later ``layers.N``) resets drift -- real progress is never drift.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging_util import get_logger
from .backends import DEFAULT_TIMEOUT, AgentUnavailable, run_agent_fix
from .drift import classify_error, extract_progress, logs_similarity
from .lessons import LessonStore
from .prompt import build_fix_prompt
from .tiers import Tier, resolve_tiers
from .verdict import AgentAnalysis, Verdict

if TYPE_CHECKING:
    from .patch import PatchCapture

_log = get_logger("agent.fixloop")


@dataclass
class PhaseResult:
    """Outcome of one phase execution."""

    ok: bool
    log_text: str = ""          # captured output / error tail
    progress: int | None = None  # deepest quant layer; auto-derived if None

    def depth(self) -> int:
        return self.progress if self.progress is not None else extract_progress(self.log_text)


# Runs the phase once and returns its result.
PhaseRunner = Callable[[], PhaseResult]
# Runs the agent for one tier: (tier, prompt, attempt_log_path, session_id) -> output text.
AgentRunner = Callable[[Tier, str, Path, str], str]


@dataclass
class FixLoopConfig:
    max_attempts: int = 5
    tier_attempt_budget: int = 3
    drift_threshold: int = 2
    agent_timeout: int = DEFAULT_TIMEOUT
    tiers: list[str] = field(default_factory=lambda: ["openclaw", "minimax"])
    user_context: str = ""


@dataclass
class FixResult:
    ok: bool
    attempts: int
    tier_index: int
    reason: str = ""
    final_analysis: AgentAnalysis | None = None


def default_agent_runner(timeout: int) -> AgentRunner:
    def _runner(tier: Tier, prompt: str, log_path: Path, session_id: str) -> str:
        try:
            run_agent_fix(tier, prompt, log_path, session_id=session_id, timeout=timeout)
        except AgentUnavailable as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"[agent-unavailable] {exc}\n")
            _log.warning("tier %s unavailable: %s", tier.name, exc)
        return log_path.read_text() if log_path.exists() else ""
    return _runner


class FixLoop:
    """Drive a phase to success via escalating agent fix attempts."""

    def __init__(
        self,
        phase_name: str,
        run_phase: PhaseRunner,
        *,
        config: FixLoopConfig | None = None,
        out_dir: Path | None = None,
        run_agent: AgentRunner | None = None,
        lesson_store: LessonStore | None = None,
        model_id: str = "unknown",
        scheme: str = "",
        method: str = "",
        cuda_guard: Callable[[], bool] | None = None,
        patch_capture: PatchCapture | None = None,
    ):
        self.phase_name = phase_name
        self.run_phase = run_phase
        self.cfg = config or FixLoopConfig()
        self.out_dir = Path(out_dir or Path.cwd())
        self.fix_log_dir = self.out_dir / "logs" / "agent_fixes" / phase_name
        self.run_agent = run_agent or default_agent_runner(self.cfg.agent_timeout)
        self.lessons = lesson_store or LessonStore(self.out_dir / "lessons")
        self.model_id = model_id
        self.scheme = scheme
        self.method = method
        self.cuda_guard = cuda_guard
        self.patch_capture = patch_capture

        self.tiers: list[Tier] = resolve_tiers(self.cfg.tiers)
        self.session_id = f"fix_{phase_name}_{uuid.uuid4().hex[:8]}"

    # -- helpers ---------------------------------------------------------
    def _effective_class(self, analysis: AgentAnalysis, err_text: str) -> str:
        if analysis.error_class and analysis.error_class != "unknown":
            return analysis.error_class
        return classify_error(err_text)

    def _same_error(self, cur_class: str, prev_class: str,
                    cur_text: str, prev_text: str) -> bool | None:
        """True/False if we can tell attempts share the same error, else None."""
        if (cur_class and cur_class != "unknown"
                and prev_class and prev_class != "unknown"):
            return cur_class == prev_class
        if prev_text:
            sim = logs_similarity(cur_text, prev_text)
            if sim is None:
                return None
            return sim >= 0.90
        return None

    def _save(self, err: str, status: str, note: str, analysis: AgentAnalysis | None):
        self.lessons.save(self.phase_name, err, status, note,
                          model_id=self.model_id, scheme=self.scheme,
                          method=self.method, analysis=analysis)

    def _write_patches(self) -> None:
        if self.patch_capture is None:
            return
        try:
            written = self.patch_capture.write_patches(
                self.out_dir / "patches" / self.phase_name)
            if written:
                _log.info("patch: captured %d source patch(es)", len(written))
        except Exception as exc:  # noqa: BLE001 - patch capture is best-effort
            _log.warning("patch write failed: %s", exc)

    # -- main loop -------------------------------------------------------
    def run(self) -> FixResult:
        cfg = self.cfg
        self.fix_log_dir.mkdir(parents=True, exist_ok=True)

        tier_count = len(self.tiers)
        tier_idx = 0
        attempts_on_tier = 0
        drift_count = 0
        max_progress = -1
        prev_class = ""
        prev_text = ""

        # Ensure the global cap lets every tier reach its budget.
        max_attempts = max(cfg.max_attempts, tier_count * cfg.tier_attempt_budget)

        cuda_was_available = bool(self.cuda_guard()) if self.cuda_guard else False

        # Pristine snapshot BEFORE any run/edit so agent source changes become patches.
        if self.patch_capture is not None:
            try:
                self.patch_capture.snapshot()
            except Exception as exc:  # noqa: BLE001 - patch capture is best-effort
                _log.warning("patch snapshot failed: %s", exc)

        _log.info("phase %s: initial run", self.phase_name)
        result = self.run_phase()
        if result.ok:
            _log.info("phase %s succeeded on first run", self.phase_name)
            return FixResult(ok=True, attempts=0, tier_index=0)

        _log.warning("phase %s failed; entering fix loop (tiers=%s)",
                     self.phase_name, [t.name for t in self.tiers])

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            attempts_on_tier += 1

            # Fail-safe budget escalation.
            if attempts_on_tier > cfg.tier_attempt_budget and tier_idx + 1 < tier_count:
                tier_idx += 1
                _log.warning("tier budget exhausted -> escalate to %s (fail-safe)",
                             self.tiers[tier_idx].name)
                drift_count, prev_class, attempts_on_tier = 0, "", 1

            tier = self.tiers[tier_idx]
            err_text = result.log_text
            cur_progress = result.depth()
            prior_class = classify_error(err_text)
            _log.info("attempt %d/%d %s tier=%s (%d/%d) prior=%s progress=%d",
                      attempt, max_attempts, self.phase_name, tier.name,
                      attempts_on_tier, cfg.tier_attempt_budget, prior_class, cur_progress)

            lessons = self.lessons.format_for_prompt(self.phase_name)
            prompt = build_fix_prompt(
                self.phase_name, err_text, lessons=lessons, attempt=attempt,
                prior_class=prior_class, user_context=cfg.user_context,
            )
            (self.fix_log_dir / f"prompt_{attempt}.txt").write_text(prompt)

            agent_log = self.fix_log_dir / f"attempt_{attempt}.log"
            output = self.run_agent(tier, prompt, agent_log, self.session_id)
            analysis = AgentAnalysis.parse(output)

            # UNFIXABLE: escalate if possible, else abort.
            if analysis.verdict is Verdict.UNFIXABLE:
                reason = analysis.unfixable_reason or "declared UNFIXABLE"
                if tier_idx + 1 < tier_count:
                    tier_idx += 1
                    _log.warning("UNFIXABLE (%s) -> escalate to %s", reason,
                                 self.tiers[tier_idx].name)
                    drift_count, prev_class, attempts_on_tier = 0, "", 0
                    continue
                _log.warning("UNFIXABLE (%s), no stronger tier; aborting", reason)
                self._save(err_text, "unfixable", f"UNFIXABLE: {reason}", analysis)
                return FixResult(ok=False, attempts=attempt, tier_index=tier_idx,
                                 reason=f"unfixable: {reason}", final_analysis=analysis)

            # ESCALATE request.
            if analysis.verdict is Verdict.ESCALATE and tier_idx + 1 < tier_count:
                tier_idx += 1
                _log.warning("agent requested ESCALATE -> %s", self.tiers[tier_idx].name)
                drift_count, prev_class, attempts_on_tier = 0, "", 0
                continue

            # Drift / progress detection.
            eff_class = self._effective_class(analysis, err_text)
            if cur_progress > max_progress:
                if drift_count > 0:
                    _log.info("progress to layer %d -> reset drift", cur_progress)
                drift_count = 0
            elif attempt > 1:
                same = self._same_error(eff_class, prev_class, err_text, prev_text)
                if same is True:
                    drift_count += 1
                    _log.warning("same error '%s' streak=%d/%d",
                                 eff_class, drift_count, cfg.drift_threshold)
                    if drift_count >= cfg.drift_threshold:
                        if tier_idx + 1 < tier_count:
                            tier_idx += 1
                            _log.warning("drift on '%s' -> escalate to %s",
                                         eff_class, self.tiers[tier_idx].name)
                            drift_count, prev_class, attempts_on_tier = 0, "", 0
                            prev_text = err_text
                            continue
                        _log.warning("drift on '%s', no stronger tier; aborting", eff_class)
                        self._save(err_text, "drift",
                                   f"Stuck on '{eff_class}' for {drift_count} attempts",
                                   analysis)
                        return FixResult(ok=False, attempts=attempt, tier_index=tier_idx,
                                         reason=f"drift: {eff_class}", final_analysis=analysis)
                elif same is False:
                    if drift_count > 0:
                        _log.info("error changed '%s'->'%s' (progress)", prev_class, eff_class)
                    drift_count = 0
                # same is None -> fail-safe: keep trying, no reset

            if eff_class and eff_class != "unknown":
                prev_class = eff_class
            prev_text = err_text
            max_progress = max(max_progress, cur_progress)

            # CUDA guard: a fix must not silently drop us to CPU.
            if cuda_was_available and self.cuda_guard and not self.cuda_guard():
                _log.error("CUDA became unavailable after fix (attempt %d); refusing CPU re-run",
                           attempt)
                with agent_log.open("a") as fh:
                    fh.write("\n[harness] REGRESSION: torch.cuda.is_available()==False after "
                             "your fix. Restore the matching CUDA torch wheel and verify before "
                             "anything else.\n")
                self._save(err_text, "still_failing",
                           f"Fix broke CUDA (attempt {attempt}); refused CPU re-run", analysis)
                result = PhaseResult(ok=False, log_text=agent_log.read_text())
                continue

            # Re-run the phase to verify the fix.
            _log.info("re-running %s after fix (attempt %d)", self.phase_name, attempt)
            result = self.run_phase()
            if result.ok:
                note = analysis.fix_plan or analysis.root_cause or f"fixed on attempt {attempt}"
                self._save(err_text, "fixed", note[:500], analysis)
                self._write_patches()
                _log.info("phase %s fixed on attempt %d (tier %s)",
                          self.phase_name, attempt, tier.name)
                return FixResult(ok=True, attempts=attempt, tier_index=tier_idx,
                                 final_analysis=analysis)

            self._save(err_text, "still_failing",
                       f"Attempt {attempt} did not resolve", analysis)

        _log.error("%s failed after %d fix attempts", self.phase_name, max_attempts)
        return FixResult(ok=False, attempts=attempt, tier_index=tier_idx,
                         reason="max_attempts_exhausted")


def run_fix_loop(
    phase_name: str,
    run_phase: PhaseRunner,
    *,
    config: FixLoopConfig | None = None,
    out_dir: Path | None = None,
    run_agent: AgentRunner | None = None,
    lesson_store: LessonStore | None = None,
    model_id: str = "unknown",
    scheme: str = "",
    method: str = "",
    cuda_guard: Callable[[], bool] | None = None,
    patch_capture: PatchCapture | None = None,
) -> FixResult:
    """Convenience wrapper constructing and running a :class:`FixLoop`."""
    return FixLoop(
        phase_name, run_phase, config=config, out_dir=out_dir, run_agent=run_agent,
        lesson_store=lesson_store, model_id=model_id, scheme=scheme, method=method,
        cuda_guard=cuda_guard, patch_capture=patch_capture,
    ).run()

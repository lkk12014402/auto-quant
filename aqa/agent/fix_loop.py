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
from ..proc import TIMEOUT_RC
from .backends import DEFAULT_TIMEOUT, AgentUnavailable, run_agent_fix
from .drift import classify_error, classify_error_detail, extract_progress, logs_similarity
from .lessons import LessonStore
from .prompt import build_fix_prompt, load_fix_playbook, playbook_section_for
from .tiers import Tier, resolve_tiers
from .trajectories import build_summary
from .verdict import AgentAnalysis, Verdict

if TYPE_CHECKING:
    from ..agent_eval.stats import TokenStats
    from .knowledge import KnowledgeStore
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


@dataclass
class AgentOutcome:
    """Structured result of one agent CLI invocation.

    Carries the process ``rc`` (and the derived ``timed_out``) alongside the
    captured ``text`` so the fix-loop can tell a wall-clock-killed attempt (whose
    streamed log is usually truncated mid-thought) apart from a clean finish -- a
    distinction the old ``str``-only contract silently dropped.
    """

    text: str
    rc: int = 0
    timed_out: bool = False


# Runs the agent for one tier: (tier, prompt, attempt_log_path, session_id) -> outcome.
# A plain ``str`` return is still accepted (coerced) for backward-compatible test doubles.
AgentRunner = Callable[[Tier, str, Path, str], "AgentOutcome | str"]


def _coerce_outcome(value: AgentOutcome | str) -> AgentOutcome:
    """Accept either an :class:`AgentOutcome` or a bare ``str`` (legacy runners)."""
    return value if isinstance(value, AgentOutcome) else AgentOutcome(text=str(value))


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
    tokens: TokenStats | None = None  # best-effort self-heal token usage (if reported)


@dataclass
class PriorAttempt:
    """Distilled state of the immediately preceding fix attempt.

    Injected into the next attempt's prompt so the loop survives what the agent's
    own session memory cannot carry across a tier escalation (a different backend
    shares no session state) or a SIGTERM-truncated transcript. The one field no
    agent-side artifact contains -- the fix-loop's OWN authoritative ``run_phase``
    re-verification -- lives here (``actual_ok`` / ``actual_class``).
    """

    attempt: int
    agent_verdict: str = ""      # what the agent CLAIMED (may be empty if truncated)
    agent_summary: str = ""      # its one-line root-cause / fix plan
    actual_ok: bool = False      # authoritative: did the fix-loop's re-run pass?
    actual_class: str = ""       # error class of that re-run (if still failing)
    truncated: bool = False      # attempt was wall-clock-killed (edits may be half-done)
    diff: str = ""               # cumulative on-disk edits so far
    evidence: str = ""           # its probe: EVIDENCE_CMD + EVIDENCE_RESULT (~500 chars)


def default_agent_runner(timeout: int) -> AgentRunner:
    def _runner(tier: Tier, prompt: str, log_path: Path, session_id: str) -> AgentOutcome:
        rc = -1
        try:
            rc = run_agent_fix(tier, prompt, log_path, session_id=session_id, timeout=timeout)
        except AgentUnavailable as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"[agent-unavailable] {exc}\n")
            _log.warning("tier %s unavailable: %s", tier.name, exc)
        text = log_path.read_text() if log_path.exists() else ""
        return AgentOutcome(text=text, rc=rc, timed_out=(rc == TIMEOUT_RC))
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
        knowledge: KnowledgeStore | None = None,
        replay: bool = True,
        run_id: str = "",
    ):
        self.phase_name = phase_name
        self.run_phase = run_phase
        self.cfg = config or FixLoopConfig()
        self.out_dir = Path(out_dir or Path.cwd())
        self.fix_log_dir = self.out_dir / "logs" / "agent_fixes" / phase_name
        self.run_agent = run_agent or default_agent_runner(self.cfg.agent_timeout)
        self.knowledge = knowledge
        self.replay = replay
        self.run_id = run_id
        # Lessons accumulate in the shared knowledge store when available, so they
        # outlive the run (and are pushed cross-machine); else fall back to run-local.
        if lesson_store is not None:
            self.lessons = lesson_store
        elif knowledge is not None:
            self.lessons = knowledge.lessons
        else:
            self.lessons = LessonStore(self.out_dir / "lessons")
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

    def _write_patches(self) -> dict[str, Path]:
        if self.patch_capture is None:
            return {}
        try:
            written = self.patch_capture.write_patches(
                self.out_dir / "patches" / self.phase_name)
            if written:
                _log.info("patch: captured %d source patch(es)", len(written))
            return written
        except Exception as exc:  # noqa: BLE001 - patch capture is best-effort
            _log.warning("patch write failed: %s", exc)
            return {}

    def _record_patch(self, err_text: str) -> None:
        """Archive the just-written patches into the shared knowledge store."""
        if self.knowledge is None:
            return
        written = self._write_patches()
        if not written:
            return
        try:
            self.knowledge.record(
                self.phase_name, err_text, written,
                model_id=self.model_id, scheme=self.scheme, method=self.method)
            self.knowledge.push_up()
        except Exception as exc:  # noqa: BLE001 - knowledge is best-effort
            _log.warning("knowledge: record failed: %s", exc)

    @staticmethod
    def _trail_append(trail: list[str], analysis: AgentAnalysis) -> None:
        """Record one attempt's hypothesis as a compact 'class: root_cause' line."""
        cls = (analysis.error_class or "").strip()
        cause = (analysis.root_cause or analysis.fix_plan or "").strip()
        if not cause:
            return
        entry = f"{cls}: {cause}" if cls else cause
        entry = entry[:160]
        if entry not in trail:
            trail.append(entry)

    def _make_prior(self, attempt: int, outcome: AgentOutcome,
                    analysis: AgentAnalysis, result: PhaseResult) -> PriorAttempt:
        """Snapshot the attempt that just ran, for injection into the next one.

        ``result`` is the authoritative state AFTER this attempt (the fix-loop's own
        ``run_phase`` re-run, or the pre-attempt failure on a punt/escalate path) --
        the single signal the agent's own transcript never contains.
        """
        diff = ""
        if self.patch_capture is not None:
            try:
                diff = self.patch_capture.combined_diff()
            except Exception as exc:  # noqa: BLE001 - diff is best-effort
                _log.debug("prior-state diff failed: %s", exc)
        evidence = ""
        if analysis is not None:
            probe = []
            if analysis.evidence_cmd:
                probe.append(f"$ {analysis.evidence_cmd}")
            if analysis.evidence_result:
                probe.append(analysis.evidence_result)
            evidence = "\n".join(probe)[:500]
        return PriorAttempt(
            attempt=attempt,
            agent_verdict=(analysis.verdict.name if analysis.verdict else ""),
            agent_summary=(analysis.root_cause or analysis.fix_plan or "").strip()[:200],
            actual_ok=result.ok,
            actual_class=classify_error(result.log_text),
            truncated=outcome.timed_out,
            diff=diff,
            evidence=evidence,
        )

    def _prior_attempt_files(self, attempt: int) -> list[str]:
        """Existing artifact files for ``attempt`` the next agent may read for depth."""
        stem = f"attempt_{attempt}"
        candidates = [
            f"{stem}.share.md",         # copilot human-readable transcript
            f"{stem}.events.jsonl",     # copilot structured session (survives truncation)
            f"{stem}.session.jsonl",    # openclaw structured session
            f"{stem}.log",              # streamed stdout (may be truncated)
            f"prompt_{attempt}.txt",    # the exact prompt that attempt received
        ]
        return [str(self.fix_log_dir / name)
                for name in candidates if (self.fix_log_dir / name).exists()]

    def _format_prior_state(self, prior: PriorAttempt | None) -> str:
        """Render the prior attempt into the prompt's high-priority state block."""
        if prior is None:
            return ""
        verdict = prior.agent_verdict or "<none -- output truncated>"
        lines = [
            f"- agent self-report: {verdict}"
            + (f" / plan: {prior.agent_summary}" if prior.agent_summary else ""),
        ]
        if prior.actual_ok:
            lines.append("- ACTUAL re-run (authoritative, the fix-loop verified this): PASSED")
        else:
            cls = f", error_class={prior.actual_class}" if prior.actual_class else ""
            lines.append(
                "- ACTUAL re-run (authoritative, the fix-loop verified this): STILL FAILING"
                f"{cls} -- so the previous attempt did NOT resolve it; do not just repeat it")
        if prior.truncated:
            lines.append("- timed out / truncated: YES -- edits may be HALF-APPLIED; reconcile "
                         "against the diff below before continuing (do not blindly redo them)")
        if prior.evidence:
            lines.append("- what the probe showed (its EVIDENCE_CMD + EVIDENCE_RESULT):\n"
                         + prior.evidence)
        files = self._prior_attempt_files(prior.attempt)
        if files:
            lines.append("- full record of that attempt (READ these if you need its reasoning / "
                         "tool calls / evidence): " + ", ".join(files))
        return "\n".join(lines)

    def _format_history(self, history: list[dict], rejected_trail: list[str],
                        cur_class: str) -> str:
        """Render the in-run attempt history: table + error evolution + dead ends.

        Unlike ``_format_prior_state`` (only the immediately preceding attempt),
        this covers EVERY attempt of the run, so a later tier can see the whole
        path -- including which hypotheses were already ruled out (the rejected
        trail is otherwise only persisted to the trajectory store on success).
        """
        if not history:
            return ""
        lines = [
            "| # | tier | error before | progress | agent verdict | summary |",
            "|---|------|--------------|----------|---------------|---------|",
        ]
        for h in history[-6:]:  # keep the block bounded
            lines.append(
                f"| {h['attempt']} | {h['tier']} | {h['error_class'] or '?'} "
                f"| {h['progress']} | {h['verdict'] or '-'} | {h['summary'] or '-'} |"
            )
        seq = " → ".join([h["error_class"] or "?" for h in history] + [cur_class or "?"])
        evolution = f"error evolution: {seq}"
        depths = [h["progress"] for h in history]
        if any(d > 0 for d in depths):
            advanced = depths[-1] > depths[0]
            evolution += (f" -- quantize progress depth {depths[0]} → {depths[-1]} "
                          f"({'advanced' if advanced else 'NOT advanced'})")
        lines.append(evolution)
        if rejected_trail:
            lines.append("hypotheses already ruled out this run (do NOT retry these):")
            lines.extend(f"- {t}" for t in rejected_trail[-6:])
        return "\n".join(lines)

    def _accrue_tokens(self, agent_output: str) -> None:
        """Fold an agent attempt's self-reported token usage into the running total.

        CLI fix agents (codex/openclaw/...) emit their own token accounting as JSON
        events in the attempt log; parsing is best-effort so a silent agent simply
        contributes turns (attempts) without tokens.
        """
        from ..agent_eval.stats import parse_usage_from_text
        try:
            self._heal_tokens.merge(parse_usage_from_text(agent_output))
        except Exception as exc:  # noqa: BLE001 - token accounting must never break a fix
            _log.debug("heal token parse failed: %s", exc)

    def _trajectory_playbook(self, prior_class: str) -> str:
        """Distilled reasoning playbook from past successful fixes (best-effort)."""
        if self.knowledge is None:
            return ""
        try:
            return self.knowledge.trajectories.format_for_prompt(
                self.phase_name, error_class=prior_class)
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            _log.warning("knowledge: trajectory lookup failed: %s", exc)
            return ""

    def _record_trajectory(self, err_text: str, analysis: AgentAnalysis,
                           trail: list[str], *, tier: str, steps: int) -> None:
        """Distil and persist the winning fix trajectory (best-effort).

        The winning hypothesis is the last trail entry; everything before it is
        the dead-end path future runs should skip.
        """
        if self.knowledge is None:
            return
        rejected = trail[:-1] if trail else []
        try:
            summary = build_summary(
                self.phase_name, err_text, analysis,
                tried_and_rejected=rejected, tier=tier, steps=steps,
                model_id=self.model_id, scheme=self.scheme, method=self.method,
                run_id=self.run_id,
            )
            self.knowledge.record_trajectory(summary)
        except Exception as exc:  # noqa: BLE001 - knowledge is best-effort
            _log.warning("knowledge: trajectory record failed: %s", exc)

    def _replay_reference(self, error_text: str, *, max_candidates: int = 2,
                          inline_threshold: int = 1500, total_budget: int = 3000) -> str:
        """Format recorded fixes for THIS failure as agent *reference* (Option A).

        We deliberately do NOT auto-apply recorded patches: a diff captured on a
        different model/checkpoint can silently break the environment, and the
        revert path is itself risky. Instead the best-matching recorded diffs +
        their provenance are handed to the agent, which decides whether the
        mechanism applies to the current model/checkpoint and adapts or ignores it.

        To keep the prompt lean, only SMALL diffs are inlined; a large (e.g.
        multi-file source) patch is summarised (files touched, +/- line counts)
        with a pointer to its on-disk path for the agent to read on demand -- never
        hard-truncated mid-hunk. A running ``total_budget`` caps inlined bytes so
        two large candidates cannot both bloat the prompt. Returns "" when replay
        is disabled, there is no knowledge store, or nothing matches.
        """
        if not (self.replay and self.knowledge):
            return ""
        try:
            records = self.knowledge.find_patches(self.phase_name, error_text)
        except Exception as exc:  # noqa: BLE001 - lookup is best-effort
            _log.warning("replay: lookup failed: %s", exc)
            return ""
        blocks: list[str] = []
        remaining = total_budget
        for rec in records[:max_candidates]:
            prov = (f"model={rec.model_id or '?'} scheme={rec.scheme or '?'} "
                    f"class={rec.error_class or '?'} sig={rec.sig_hash}")
            diff = rec.diff_text(max_chars=inline_threshold + 1)
            if diff and len(diff) <= inline_threshold and len(diff) <= remaining:
                remaining -= len(diff)
                blocks.append(f"### candidate ({prov})\n```diff\n{diff}\n```")
            else:
                # Too big to inline: summarise + point the agent at the patch on disk.
                paths = ", ".join(rec.patch_paths())
                blocks.append(
                    f"### candidate ({prov})\n{rec.stat_summary()}\n"
                    f"full patch on disk (READ it if the mechanism looks relevant): {paths}")
        if not blocks:
            return ""
        _log.info("replay: injecting %d recorded fix(es) as reference for %s "
                  "(no auto-apply)", len(blocks), self.phase_name)
        return "\n\n".join(blocks)

    def _log_stat(self, outcome: str, *, sig_hash: str = "", error_class: str = "",
                  tier: str = "") -> None:
        if self.knowledge is None:
            return
        try:
            self.knowledge.log_event(self.phase_name, outcome, sig_hash=sig_hash,
                                     error_class=error_class, model_id=self.model_id,
                                     tier=tier)
        except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
            _log.warning("knowledge: stat log failed: %s", exc)

    # -- main loop -------------------------------------------------------
    def run(self) -> FixResult:
        """Drive the phase to success, stamping best-effort self-heal token usage."""
        from ..agent_eval.stats import TokenStats
        self._heal_tokens = TokenStats()
        fr = self._run_loop()
        if not self._heal_tokens.is_empty():
            fr.tokens = self._heal_tokens
        return fr

    def _run_loop(self) -> FixResult:
        cfg = self.cfg
        self.fix_log_dir.mkdir(parents=True, exist_ok=True)

        tier_count = len(self.tiers)
        tier_idx = 0
        attempts_on_tier = 0
        drift_count = 0
        max_progress = -1
        prev_class = ""
        # Distilled record of hypotheses tried on the way to a fix; on success the
        # entries before the winner become the trajectory's "dead ends".
        rejected_trail: list[str] = []
        # In-run attempt history (attempt/tier/class-before/progress/verdict/
        # summary), rendered into each prompt so later attempts see the whole path.
        history: list[dict] = []
        prev_text = ""
        prior: PriorAttempt | None = None  # distilled state of the previous attempt

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
            detail = classify_error_detail(err_text)
            prior_class = detail.class_name
            _log.info("attempt %d/%d %s tier=%s (%d/%d) prior=%s progress=%d",
                      attempt, max_attempts, self.phase_name, tier.name,
                      attempts_on_tier, cfg.tier_attempt_budget, prior_class, cur_progress)

            lessons = self.lessons.format_for_prompt(self.phase_name)
            playbook = self._trajectory_playbook(prior_class)
            known_fixes = self._replay_reference(err_text)
            prompt = build_fix_prompt(
                self.phase_name, err_text, lessons=lessons, attempt=attempt,
                prior_class=prior_class, prior_evidence=detail.evidence,
                user_context=cfg.user_context,
                skill_playbook=load_fix_playbook(),
                playbook_pointer=playbook_section_for(prior_class),
                trajectories=playbook, known_fixes=known_fixes,
                history=self._format_history(history, rejected_trail, prior_class),
                prior_diff=prior.diff if prior else "",
                prior_state=self._format_prior_state(prior),
            )
            (self.fix_log_dir / f"prompt_{attempt}.txt").write_text(prompt)

            agent_log = self.fix_log_dir / f"attempt_{attempt}.log"
            outcome = _coerce_outcome(
                self.run_agent(tier, prompt, agent_log, self.session_id))
            output = outcome.text
            if outcome.timed_out:
                _log.warning("attempt %d timed out (rc=%d); edits may be partial -- "
                             "surfacing to next attempt as prior-state", attempt, outcome.rc)
            self._accrue_tokens(output)
            analysis = AgentAnalysis.parse(output)
            self._trail_append(rejected_trail, analysis)
            history.append({
                "attempt": attempt,
                "tier": tier.name,
                "error_class": prior_class,
                "progress": cur_progress,
                "verdict": analysis.verdict.name if analysis.verdict else "",
                "summary": (analysis.root_cause or analysis.fix_plan or "").strip()[:120],
            })
            # Snapshot this attempt for the NEXT one. On the punt/escalate paths below
            # `result` is still the pre-attempt failure (correct: nothing was fixed);
            # the re-run paths refresh it and re-snapshot before looping.
            prior = self._make_prior(attempt, outcome, analysis, result)

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
                self._log_stat("unresolved", error_class=self._effective_class(analysis, err_text),
                               tier=tier.name)
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
                        self._log_stat("unresolved", error_class=eff_class, tier=tier.name)
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
                prior = self._make_prior(attempt, outcome, analysis, result)
                continue

            # Re-run the phase to verify the fix.
            _log.info("re-running %s after fix (attempt %d)", self.phase_name, attempt)
            result = self.run_phase()
            if result.ok:
                note = analysis.fix_plan or analysis.root_cause or f"fixed on attempt {attempt}"
                self._save(err_text, "fixed", note[:500], analysis)
                self._record_patch(err_text)
                self._record_trajectory(err_text, analysis, rejected_trail,
                                        tier=tier.name, steps=attempt)
                # Under reference-only replay a matching recorded fix is never
                # auto-applied, so a success while `known_fixes` was injected means
                # the agent consulted+adopted recorded knowledge -> credit it.
                fix_outcome = "reference_used" if known_fixes else "agent_fixed"
                self._log_stat(fix_outcome, error_class=self._effective_class(analysis, err_text),
                               tier=tier.name)
                _log.info("phase %s fixed on attempt %d (tier %s)",
                          self.phase_name, attempt, tier.name)
                return FixResult(ok=True, attempts=attempt, tier_index=tier_idx,
                                 final_analysis=analysis)

            self._save(err_text, "still_failing",
                       f"Attempt {attempt} did not resolve", analysis)
            # Refresh prior-state with the authoritative re-run outcome for the next attempt.
            prior = self._make_prior(attempt, outcome, analysis, result)

        _log.error("%s failed after %d fix attempts", self.phase_name, max_attempts)
        self._log_stat("unresolved", error_class=prev_class)
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
    knowledge: KnowledgeStore | None = None,
    replay: bool = True,
    run_id: str = "",
) -> FixResult:
    """Convenience wrapper constructing and running a :class:`FixLoop`."""
    return FixLoop(
        phase_name, run_phase, config=config, out_dir=out_dir, run_agent=run_agent,
        lesson_store=lesson_store, model_id=model_id, scheme=scheme, method=method,
        cuda_guard=cuda_guard, patch_capture=patch_capture,
        knowledge=knowledge, replay=replay, run_id=run_id,
    ).run()


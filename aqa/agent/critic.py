"""Critic: calibrate the agent's self-report against the harness's own re-run.

RFC §3.5 separates **propose** (the fix agent) from **decide** (the deterministic
coordinator).  The fix-loop already *decides* deterministically — it keeps a fix
only when its own :func:`run_phase` re-run passes, never because the agent said
``VERDICT: FIXABLE``.  What was missing is the **learning record**: each attempt,
compare what the agent *claimed* against what the re-run *actually* found, and
emit that delta as calibration.

An agent that claims a fix the re-run rejects (``overclaimed`` / ``silent_fail``)
is the anti-drift signal worth surfacing: it feeds future prompts/lessons and
tells an operator which tier over-reports.  The Critic **observes and records**;
it never overrides the deterministic keep/revert.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .verdict import AgentAnalysis, Verdict


class Calibration(str, Enum):
    """How the agent's self-report lined up with the authoritative re-run."""

    CONFIRMED = "confirmed"        # claimed/attempted a fix AND the re-run passed
    OVERCLAIMED = "overclaimed"    # claimed FIXABLE but the re-run still fails
    SILENT_FAIL = "silent_fail"    # no verdict emitted and the re-run still fails
    CONCEDED = "conceded"          # declared UNFIXABLE and the re-run still fails
    UNDERSTATED = "understated"    # declared UNFIXABLE yet the re-run actually passed
    ESCALATED = "escalated"        # asked to escalate (a punt, not a decision)


# The calibrations that mean the agent's self-report disagreed with reality — the
# signals an operator (and future prompts) should weight.
_MISCALIBRATED = {Calibration.OVERCLAIMED, Calibration.SILENT_FAIL,
                  Calibration.UNDERSTATED}


@dataclass
class CriticRecord:
    """One attempt's claim-vs-reality calibration (a learning record)."""

    phase: str
    attempt: int
    tier: str
    calibration: Calibration
    agent_verdict: str            # what the agent CLAIMED (may be empty)
    actual_ok: bool               # authoritative: did the re-run pass?
    actual_class: str             # error class of the re-run (if still failing)
    note: str

    @property
    def honest(self) -> bool:
        """True when the self-report matched the authoritative outcome."""
        return self.calibration not in _MISCALIBRATED

    def to_event(self) -> dict:
        return {
            "phase": self.phase,
            "attempt": self.attempt,
            "tier": self.tier,
            "calibration": self.calibration.value,
            "agent_verdict": self.agent_verdict or None,
            "actual_ok": self.actual_ok,
            "actual_class": self.actual_class or None,
            "honest": self.honest,
            "note": self.note,
        }


class Critic:
    """Deterministic calibrator of agent self-report vs the authoritative re-run."""

    def calibrate(self, analysis: AgentAnalysis, *, actual_ok: bool,
                  actual_class: str = "", phase: str = "", attempt: int = 0,
                  tier: str = "") -> CriticRecord:
        verdict = analysis.verdict
        cal = self._classify(verdict, actual_ok)
        note = self._note(cal, verdict, actual_class)
        return CriticRecord(
            phase=phase, attempt=attempt, tier=tier, calibration=cal,
            agent_verdict=(verdict.name if verdict else ""),
            actual_ok=actual_ok, actual_class=(actual_class if not actual_ok else ""),
            note=note)

    @staticmethod
    def _classify(verdict: Verdict | None, actual_ok: bool) -> Calibration:
        if verdict is Verdict.ESCALATE:
            return Calibration.ESCALATED
        if verdict is Verdict.UNFIXABLE:
            return Calibration.UNDERSTATED if actual_ok else Calibration.CONCEDED
        # FIXABLE or no explicit verdict: the agent (implicitly) attempted a fix.
        if actual_ok:
            return Calibration.CONFIRMED
        return Calibration.OVERCLAIMED if verdict is Verdict.FIXABLE else Calibration.SILENT_FAIL

    @staticmethod
    def _note(cal: Calibration, verdict: Verdict | None, actual_class: str) -> str:
        claim = verdict.name if verdict else "no verdict"
        cls = f" (re-run error_class={actual_class})" if actual_class else ""
        return {
            Calibration.CONFIRMED:
                f"agent claim '{claim}' confirmed by authoritative re-run",
            Calibration.OVERCLAIMED:
                f"agent claimed FIXABLE but re-run still fails{cls} — self-report over-reported",
            Calibration.SILENT_FAIL:
                f"agent emitted no verdict and re-run still fails{cls}",
            Calibration.CONCEDED:
                f"agent conceded UNFIXABLE and re-run agrees{cls}",
            Calibration.UNDERSTATED:
                "agent declared UNFIXABLE yet the re-run passed — under-reported",
            Calibration.ESCALATED:
                "agent requested ESCALATE — deferred to a stronger tier",
        }[cal]

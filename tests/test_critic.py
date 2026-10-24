"""Critic: deterministic calibration of agent self-report vs authoritative re-run."""

from __future__ import annotations

from aqa.agent.critic import Calibration, Critic, CriticRecord
from aqa.agent.verdict import AgentAnalysis, Verdict


def _analysis(verdict: Verdict | None) -> AgentAnalysis:
    return AgentAnalysis(verdict=verdict)


def test_fixable_and_passes_is_confirmed():
    rec = Critic().calibrate(_analysis(Verdict.FIXABLE), actual_ok=True)
    assert rec.calibration is Calibration.CONFIRMED
    assert rec.honest is True


def test_fixable_but_still_failing_is_overclaimed():
    rec = Critic().calibrate(_analysis(Verdict.FIXABLE), actual_ok=False,
                             actual_class="import_error")
    assert rec.calibration is Calibration.OVERCLAIMED
    assert rec.honest is False
    assert rec.actual_class == "import_error"


def test_no_verdict_and_failing_is_silent_fail():
    rec = Critic().calibrate(_analysis(None), actual_ok=False)
    assert rec.calibration is Calibration.SILENT_FAIL
    assert rec.honest is False


def test_no_verdict_but_passes_is_confirmed():
    rec = Critic().calibrate(_analysis(None), actual_ok=True)
    assert rec.calibration is Calibration.CONFIRMED


def test_unfixable_and_failing_is_conceded():
    rec = Critic().calibrate(_analysis(Verdict.UNFIXABLE), actual_ok=False)
    assert rec.calibration is Calibration.CONCEDED
    assert rec.honest is True  # an honest concession is not miscalibration


def test_unfixable_but_passes_is_understated():
    rec = Critic().calibrate(_analysis(Verdict.UNFIXABLE), actual_ok=True)
    assert rec.calibration is Calibration.UNDERSTATED
    assert rec.honest is False


def test_escalate_is_escalated():
    rec = Critic().calibrate(_analysis(Verdict.ESCALATE), actual_ok=False)
    assert rec.calibration is Calibration.ESCALATED


def test_to_event_shape():
    rec = Critic().calibrate(_analysis(Verdict.FIXABLE), actual_ok=False,
                             actual_class="oom", phase="quantize", attempt=2,
                             tier="opus")
    ev = rec.to_event()
    assert ev["phase"] == "quantize" and ev["attempt"] == 2 and ev["tier"] == "opus"
    assert ev["calibration"] == "overclaimed"
    assert ev["agent_verdict"] == "FIXABLE"
    assert ev["actual_ok"] is False and ev["honest"] is False
    assert ev["actual_class"] == "oom"


def test_actual_class_cleared_when_passing():
    rec = Critic().calibrate(_analysis(Verdict.FIXABLE), actual_ok=True,
                             actual_class="ignored")
    assert rec.actual_class == ""
    assert isinstance(rec, CriticRecord)

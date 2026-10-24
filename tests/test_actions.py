"""Tests for the S3 phase-action guard + bounded re-QUANT loop controller.

Covers docs/rfc/0001-reproducible-session-state-machine.md §S3
(Hyperloom PHASE_ALLOWED_ACTIONS / allowed_actions_for + macro-cycle reloop).
"""

from __future__ import annotations

import json

import pytest

from aqa.session.actions import (
    PHASE_ALLOWED_ACTIONS,
    PhaseActionDenied,
    action_contract,
    allowed_actions_for,
    is_action_allowed,
    is_loopback,
    is_transition_allowed,
)
from aqa.session.state import Phase


# ── allowed-actions table ─────────────────────────────────────────────────────
def test_every_phase_has_an_action_entry():
    for p in Phase:
        assert p in PHASE_ALLOWED_ACTIONS


def test_allowed_actions_for_is_sorted_and_deterministic():
    assert allowed_actions_for(Phase.QUANT) == ("integrate_recipe", "quantize", "requantize")
    assert allowed_actions_for("QUANT") == allowed_actions_for(Phase.QUANT)
    assert allowed_actions_for(Phase.ANALYZE) == ("recipe_kb_lookup", "target_analysis")


def test_gate_is_coordinator_reserved_no_proposable_action():
    assert allowed_actions_for(Phase.GATE) == ()


def test_is_action_allowed_scopes_requantize_to_quant():
    assert is_action_allowed("requantize", Phase.QUANT)
    assert not is_action_allowed("requantize", Phase.EVAL)
    assert is_action_allowed("evaluate", Phase.EVAL)
    assert not is_action_allowed("quantize", Phase.EVAL)


def test_bad_phase_or_action_is_rejected_not_raised():
    assert allowed_actions_for("NOPE") == ()
    assert not is_action_allowed("evaluate", "NOPE")
    assert not is_action_allowed("", Phase.EVAL)


# ── transition machine ────────────────────────────────────────────────────────
_PLAN = [Phase.BASELINE, Phase.QUANT, Phase.EVAL, Phase.GATE, Phase.CLOSE]


def test_forward_transitions_follow_the_plan():
    assert is_transition_allowed(Phase.QUANT, Phase.EVAL, _PLAN)
    assert is_transition_allowed(Phase.GATE, Phase.CLOSE, _PLAN)
    # skipping a node that is not next is not a forward step
    assert not is_transition_allowed(Phase.BASELINE, Phase.EVAL, _PLAN)


def test_forward_skips_trimmed_out_nodes():
    plan = [Phase.QUANT, Phase.EVAL, Phase.CLOSE]  # no GATE in this run
    assert is_transition_allowed(Phase.EVAL, Phase.CLOSE, plan)


def test_gate_to_quant_is_the_only_loopback():
    assert is_loopback(Phase.GATE, Phase.QUANT)
    assert is_transition_allowed(Phase.GATE, Phase.QUANT, _PLAN)
    # the dangerous silent re-quant inside EVAL is illegal
    assert not is_transition_allowed(Phase.EVAL, Phase.QUANT, _PLAN)
    assert not is_loopback(Phase.EVAL, Phase.QUANT)


def test_arbitrary_backward_jump_denied():
    assert not is_transition_allowed(Phase.CLOSE, Phase.BASELINE, _PLAN)
    assert not is_transition_allowed(Phase.EVAL, Phase.BASELINE, _PLAN)


def test_action_contract_is_serialisable_and_lists_loopback():
    c = action_contract()
    json.dumps(c)  # must round-trip
    assert c["phases"]["QUANT"] == ["integrate_recipe", "quantize", "requantize"]
    assert c["phases"]["GATE"] == []
    assert ["GATE", "QUANT"] in c["loopback_edges"]


# ── session guard (records + enforces) ────────────────────────────────────────
def _cfg(tmp_path, **agent):
    from aqa.config import RunConfig

    base = {
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": False},
        "output": {"dir": str(tmp_path), "sinks": []},
    }
    if agent:
        base["agent"] = agent
    return RunConfig.model_validate(base)


def _read_events(session_dir):
    return [json.loads(x) for x in (session_dir / "events.jsonl").read_text().splitlines()]


def test_open_session_writes_allowed_actions_contract(tmp_path):
    from aqa.session import open_session

    sess = open_session(tmp_path, _cfg(tmp_path), "run-1",
                        [Phase.QUANT, Phase.EVAL, Phase.CLOSE])
    contract = json.loads((sess.dir / "allowed_actions.json").read_text())
    assert contract["phases"]["EVAL"] == ["evaluate"]


def test_guard_action_advisory_records_but_does_not_raise(tmp_path):
    from aqa.session import open_session

    sess = open_session(tmp_path, _cfg(tmp_path), "run-1",
                        [Phase.QUANT, Phase.EVAL, Phase.CLOSE])
    sess.enter(Phase.EVAL)
    assert sess.guard_action("requantize") is False  # advisory default -> no raise
    denied = [e for e in _read_events(sess.dir) if e["type"] == "phase_action_denied"]
    assert denied and denied[-1]["action"] == "requantize" and denied[-1]["phase"] == "EVAL"
    assert denied[-1]["enforced"] is False


def test_guard_action_enforce_raises_on_out_of_phase_action(tmp_path):
    from aqa.session import open_session

    sess = open_session(tmp_path, _cfg(tmp_path, action_guard={"enforce": True}), "run-1",
                        [Phase.QUANT, Phase.EVAL, Phase.CLOSE])
    sess.enter(Phase.EVAL)
    with pytest.raises(PhaseActionDenied, match="requantize"):
        sess.guard_action("requantize")
    # a legal action still passes
    sess.enter(Phase.QUANT)
    assert sess.guard_action("quantize") is True


def test_guard_disabled_is_noop(tmp_path):
    from aqa.session import open_session

    sess = open_session(tmp_path, _cfg(tmp_path, action_guard={"enabled": False}), "run-1",
                        [Phase.QUANT, Phase.EVAL])
    sess.enter(Phase.EVAL)
    assert sess.guard_action("requantize") is False
    assert not [e for e in _read_events(sess.dir) if e["type"] == "phase_action_denied"]


def test_enter_annotates_loopback_transition(tmp_path):
    from aqa.session import open_session

    sess = open_session(tmp_path, _cfg(tmp_path), "run-1",
                        [Phase.QUANT, Phase.EVAL, Phase.GATE, Phase.CLOSE])
    sess.enter(Phase.QUANT); sess.done(Phase.QUANT)
    sess.enter(Phase.EVAL); sess.done(Phase.EVAL)
    sess.enter(Phase.GATE); sess.done(Phase.GATE)
    sess.enter(Phase.QUANT)  # GATE -> QUANT loop-back
    trans = [e for e in _read_events(sess.dir) if e["type"] == "phase_transition"]
    loop = [t for t in trans if t["from"] == "GATE" and t["to"] == "QUANT"]
    assert loop and loop[-1]["loopback"] is True and loop[-1]["allowed"] is True


# ── re-QUANT controller (pure loop policy) ────────────────────────────────────
def test_controller_stops_on_go():
    from aqa.pipeline import _RequantController

    c = _RequantController(n_candidates=3, max_cycles=3, best_score=0.50)
    assert c.next_index() == 0
    assert c.observe(0.55, "GO") is True  # improved + GO
    assert c.next_index() is None and c.stop_reason == "go"


def test_controller_keeps_best_and_bounds_cycles():
    from aqa.pipeline import _RequantController

    c = _RequantController(n_candidates=5, max_cycles=2, min_gain=0.0, no_gain_patience=9,
                          best_score=0.40)
    assert c.observe(0.45, "NO-GO") is True   # new best
    assert c.observe(0.42, "NO-GO") is False  # worse than best, kept
    assert c.next_index() is None and c.stop_reason == "max_cycles"
    assert c.best_score == pytest.approx(0.45)


def test_controller_no_gain_convergence():
    from aqa.pipeline import _RequantController

    c = _RequantController(n_candidates=5, max_cycles=9, min_gain=0.01, no_gain_patience=2,
                          best_score=0.50)
    c.observe(0.505, "NO-GO")  # +0.005 < min_gain 0.01 -> no-gain #1
    assert not c.stopped
    c.observe(0.50, "NO-GO")   # no improvement -> no-gain #2 -> stop
    assert c.stopped and c.stop_reason == "no_gain_convergence"


def test_controller_exhausts_candidates():
    from aqa.pipeline import _RequantController

    c = _RequantController(n_candidates=1, max_cycles=5, no_gain_patience=9, best_score=0.0)
    c.observe(0.10, "NO-GO")
    assert c.next_index() is None and c.stop_reason == "candidates_exhausted"

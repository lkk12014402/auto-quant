"""PolicyGate: deterministic named-rule review of a captured unified diff."""

from __future__ import annotations

from aqa.agent.policy_gate import PolicyGate, parse_diff


def _diff(label_rel: str, *, added=(), removed=()) -> str:
    """Build a minimal one-file unified diff with the given +/- content lines."""
    lines = [f"--- a/{label_rel}", f"+++ b/{label_rel}"]
    lines += [f"-{x}" for x in removed]
    lines += [f"+{x}" for x in added]
    return "\n".join(lines) + "\n"


def test_parse_diff_extracts_paths_and_lines():
    pd = parse_diff(_diff("auto_round/foo.py", added=["a", "b"], removed=["c"]))
    assert pd.paths == ["b/auto_round/foo.py"]
    assert pd.added == ["a", "b"] and pd.removed == ["c"]
    assert pd.net_deletions == -1


def test_clean_source_edit_is_allowed():
    diff = _diff("auto_round/quant.py", added=["    x = fixed()"], removed=["    x = bug()"])
    decision = PolicyGate().review(diff)
    assert decision.allowed is True and decision.violations == []


def test_empty_diff_is_allowed():
    assert PolicyGate().review("").allowed is True
    assert PolicyGate().review("   \n").allowed is True


def test_editing_test_files_is_flagged():
    decision = PolicyGate().review(_diff("auto_round/tests/test_quant.py",
                                         added=["assert True"]))
    assert decision.allowed is False
    assert "edit_tests" in decision.rule_names


def test_skip_marker_is_flagged():
    decision = PolicyGate().review(_diff("auto_round/quant.py",
                                         added=["@pytest.mark.skip(reason='later')"]))
    assert "skip_tests" in decision.rule_names


def test_disable_cuda_is_flagged():
    decision = PolicyGate().review(_diff("auto_round/env.py",
                                         added=["torch.cuda.is_available = lambda: True"]))
    assert "disable_cuda" in decision.rule_names


def test_vcs_ci_edit_is_flagged():
    decision = PolicyGate().review(_diff(".github/workflows/ci.yml", added=["  run: true"]))
    assert "touches_vcs_ci" in decision.rule_names


def test_mass_deletion_threshold():
    big = _diff("auto_round/quant.py", removed=[f"line{i}" for i in range(50)])
    assert PolicyGate(max_net_deletions=200).review(big).allowed is True
    decision = PolicyGate(max_net_deletions=10).review(big)
    assert "mass_deletion" in decision.rule_names


def test_multiple_violations_named():
    diff = _diff("auto_round/tests/test_x.py",
                 added=["@pytest.mark.xfail", "device = 'cpu'"])
    decision = PolicyGate().review(diff)
    assert {"edit_tests", "skip_tests", "disable_cuda"} <= set(decision.rule_names)


def test_to_event_and_enforce_flag():
    gate = PolicyGate(enforce=True)
    assert gate.enforce is True
    decision = gate.review(_diff("auto_round/tests/test_x.py", added=["x"]))
    ev = decision.to_event()
    assert ev["allowed"] is False
    assert ev["violations"][0]["rule"] == "edit_tests"
    assert "detail" in ev["violations"][0]

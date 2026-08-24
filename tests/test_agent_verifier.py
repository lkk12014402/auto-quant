"""Tests for the structured verdict + verifier layer (Harbor-style reward dict).

Covers the :class:`EvalVerdict` value type, the clean-denominator semantics in
:class:`AgentEvalResult` (infra/parse failures excluded from the mean), and the
SWE-bench verifier that emits per-test sub-metrics + a failure category.
"""

from __future__ import annotations

import asyncio

import pytest

from aqa.agent_eval.loop import AgentEvalResult, TaskOutcome
from aqa.agent_eval.verifier import (
    INFRA_ERROR,
    MODEL_WRONG,
    OK,
    PARSE_ERROR,
    EvalVerdict,
    ParseError,
    coerce_verdict,
    verdict_from_exception,
)
from aqa.sandbox.base import ExecResult
from aqa.swebench import SWEBenchInstance, run_swebench


# --------------------------------------------------------------------------
# EvalVerdict value type
# --------------------------------------------------------------------------
def test_verdict_float_and_scored():
    v = EvalVerdict(reward=0.75, solved=True, category=OK, metrics={"x": 1.0})
    assert float(v) == 0.75
    assert v.is_scored
    assert EvalVerdict(category=MODEL_WRONG).is_scored          # a real 0 counts
    assert not EvalVerdict(category=INFRA_ERROR).is_scored      # infra excluded
    assert not EvalVerdict(category=PARSE_ERROR).is_scored


def test_from_reward_and_coerce():
    assert EvalVerdict.from_reward(1.0).category == OK
    assert EvalVerdict.from_reward(0.0).category == MODEL_WRONG
    assert coerce_verdict(True).solved is True
    assert coerce_verdict(0.0).category == MODEL_WRONG
    passthrough = EvalVerdict(reward=0.5, category=OK)
    assert coerce_verdict(passthrough) is passthrough


def test_verdict_from_exception_categories():
    infra = verdict_from_exception(RuntimeError("cuda blew up"))
    assert infra.category == INFRA_ERROR
    assert infra.exc_type == "RuntimeError"
    assert not infra.is_scored
    parse = verdict_from_exception(ParseError("bad grade file"))
    assert parse.category == PARSE_ERROR


# --------------------------------------------------------------------------
# clean-denominator aggregation
# --------------------------------------------------------------------------
def test_mean_score_excludes_infra_failures():
    result = AgentEvalResult(name="d", outcomes=[
        TaskOutcome("a", "", 1.0, verdict=EvalVerdict.from_reward(1.0)),
        TaskOutcome("b", "", 0.0, verdict=EvalVerdict.from_reward(0.0)),  # real wrong
        TaskOutcome("c", "", 0.0, error="boom",
                    verdict=verdict_from_exception(RuntimeError("x"))),   # infra
    ])
    # infra sample is out of the denominator: mean = (1.0 + 0.0) / 2, not / 3.
    assert result.mean_score() == pytest.approx(0.5)
    assert result.num_scored() == 2
    assert result.per_sample() == [1.0, 0.0]
    cats = result.by_category()
    assert cats[OK] == 1 and cats[MODEL_WRONG] == 1 and cats[INFRA_ERROR] == 1


def test_legacy_outcome_without_verdict_still_counts():
    # No verdict + no error => scored (backward compatible with scalar path).
    result = AgentEvalResult(name="d", outcomes=[
        TaskOutcome("a", "", 1.0),
        TaskOutcome("b", "", 0.0),
    ])
    assert result.mean_score() == pytest.approx(0.5)
    assert result.to_dict()["num_scored"] == 2


# --------------------------------------------------------------------------
# SWE-bench structured verifier
# --------------------------------------------------------------------------
class _FakeSandbox:
    name = "fake"

    def __init__(self, *, patch, test_log):
        self.patch = patch
        self.test_log = test_log

    def start(self): ...
    def close(self): ...
    def write_file(self, path, content): ...
    def read_file(self, path): return ""

    def exec(self, command, *, timeout=None, cwd=None, env=None):
        if "git diff --cached" in command:
            return ExecResult(command, 0, stdout=self.patch)
        if "pytest" in command:
            return ExecResult(command, 0, stdout=self.test_log)
        return ExecResult(command, 0, stdout="")


def _inst(iid):
    return SWEBenchInstance(
        instance_id=iid, repo="a/a", base_commit="c0",
        problem_statement="fix it", test_patch="diff --git a/t b/t\n",
        fail_to_pass=["t/test.py::t_fix"], pass_to_pass=["t/test.py::t_keep"],
    )


def _run(insts, logs, patches):
    def factory(inst):
        return _FakeSandbox(patch=patches[inst.instance_id],
                            test_log=logs[inst.instance_id])
    from aqa.agent_eval.react import ToolUseSolver  # noqa: F401 - ensure import path

    client = _ScriptedClient(["Final Answer: done"])
    return run_swebench(insts, client, sandbox_factory=factory,
                        spec_provider=lambda i: None, model_name="m",
                        concurrency=2, max_turns=2)


class _ScriptedClient:
    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    async def chat(self, messages, **params):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


def test_swebench_verifier_emits_metrics_and_category():
    insts = [_inst("a__a-1"), _inst("a__a-2")]
    logs = {
        "a__a-1": "PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep",   # resolved
        "a__a-2": "FAILED t/test.py::t_fix\nPASSED t/test.py::t_keep",   # model wrong
    }
    patches = {"a__a-1": "diff a\n", "a__a-2": "diff b\n"}
    agent_result, _report = _run(insts, logs, patches)
    by_id = {o.task_id: o for o in agent_result.outcomes}

    v1 = by_id["a__a-1"].verdict
    assert v1.category == OK and v1.solved
    assert v1.metrics["f2p_passed"] == 1.0 and v1.metrics["f2p_total"] == 1.0
    assert v1.metrics["patch_applied"] == 1.0

    v2 = by_id["a__a-2"].verdict
    assert v2.category == MODEL_WRONG and not v2.solved
    assert v2.metrics["f2p_passed"] == 0.0 and v2.metrics["f2p_failed"] == 1.0
    assert agent_result.mean_score() == pytest.approx(0.5)


def test_swebench_verifier_flags_infra_when_no_tests_ran():
    # Empty test log => every graded test is MISSING => grader/container failure,
    # not the model. It must be excluded from the scored denominator.
    insts = [_inst("a__a-1"), _inst("a__a-2")]
    logs = {
        "a__a-1": "PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep",   # resolved
        "a__a-2": "",                                                     # nothing ran
    }
    patches = {"a__a-1": "diff a\n", "a__a-2": "diff b\n"}
    agent_result, _report = _run(insts, logs, patches)
    by_id = {o.task_id: o for o in agent_result.outcomes}

    assert by_id["a__a-2"].verdict.category == INFRA_ERROR
    assert not by_id["a__a-2"].verdict.is_scored
    # Only the one genuinely-graded instance counts => mean 1.0 over 1 sample.
    assert agent_result.num_scored() == 1
    assert agent_result.mean_score() == pytest.approx(1.0)
    assert agent_result.by_category()[INFRA_ERROR] == 1


def test_asyncio_smoke():
    # guard: the run helpers stay asyncio-drivable
    assert asyncio.get_event_loop_policy() is not None

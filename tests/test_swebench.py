"""Tests for the SWE-bench Verified adapter (dataset / spec / grading / run).

Everything is exercised without Docker or the upstream ``swebench`` package: a
fake sandbox returns scripted patch + test-log output, and a scripted client
drives the ReAct agent.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from aqa.logging_util import AqaError
from aqa.sandbox.base import ExecResult
from aqa.swebench import (
    SWEBenchInstance,
    build_instruction,
    build_report,
    compute_resolution,
    docker_image_for,
    load_swebench,
    parse_pytest_log,
    resolve_spec,
    run_swebench,
    write_predictions,
    write_report,
)
from aqa.swebench.environment import SWEBenchEnvironment
from aqa.swebench.spec import MISSING, image_key


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class FakeSandbox:
    """Records exec/write calls; scripts patch-capture + test-run output."""

    name = "fake"

    def __init__(self, *, patch="diff --git a b\n", test_log="", test_rc=0):
        self.patch = patch
        self.test_log = test_log
        self.test_rc = test_rc
        self.started = False
        self.closed = False
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def write_file(self, path, content):
        self.files[path] = content

    def read_file(self, path):
        return self.files[path]

    def exec(self, command, *, timeout=None, cwd=None, env=None):
        self.commands.append(command)
        if "git diff --cached" in command:
            return ExecResult(command, 0, stdout=self.patch)
        if "pytest" in command or "test_cmd" in command:
            return ExecResult(command, self.test_rc, stdout=self.test_log)
        return ExecResult(command, 0, stdout="")


class ScriptedClient:
    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    async def chat(self, messages, **params):
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def _inst(iid="astropy__astropy-12907", **kw):
    base = {
        "instance_id": iid,
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "problem_statement": "separability_matrix is wrong",
        "test_patch": "diff --git a/test b/test\n",
        "fail_to_pass": ["t/test.py::t_fix"],
        "pass_to_pass": ["t/test.py::t_keep"],
    }
    base.update(kw)
    return SWEBenchInstance(**base)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def test_from_row_parses_json_test_lists():
    inst = SWEBenchInstance.from_row({
        "instance_id": "x__y-1", "repo": "x/y", "base_commit": "c",
        "problem_statement": "p",
        "FAIL_TO_PASS": '["a::t1", "a::t2"]',
        "PASS_TO_PASS": ["b::t3"],
    })
    assert inst.fail_to_pass == ["a::t1", "a::t2"]
    assert inst.pass_to_pass == ["b::t3"]


def test_from_row_requires_core_fields():
    with pytest.raises(AqaError, match="instance_id"):
        SWEBenchInstance.from_row({"repo": "x/y", "base_commit": "c"})
    with pytest.raises(AqaError, match="repo/base_commit"):
        SWEBenchInstance.from_row({"instance_id": "a__b-1"})


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_load_jsonl_filter_limit(tmp_path):
    rows = [
        {"instance_id": f"r__r-{n}", "repo": "r/r", "base_commit": "c",
         "problem_statement": "p", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]",
         "difficulty": "easy" if n < 2 else "hard"}
        for n in range(4)
    ]
    p = _write_jsonl(tmp_path, rows)
    assert len(load_swebench(p)) == 4
    assert len(load_swebench(p, limit=2)) == 2
    assert {i.instance_id for i in load_swebench(p, instance_ids=["r__r-1", "r__r-3"])} == {
        "r__r-1", "r__r-3"}
    assert len(load_swebench(p, difficulty="easy")) == 2


def test_load_missing_instance_id_errors(tmp_path):
    p = _write_jsonl(tmp_path, [{"instance_id": "a__a-1", "repo": "a/a",
                                 "base_commit": "c", "problem_statement": "p"}])
    with pytest.raises(AqaError, match="not found"):
        load_swebench(p, instance_ids=["nope"])


# --------------------------------------------------------------------------
# spec / scoring
# --------------------------------------------------------------------------
def test_image_key_and_image():
    assert image_key("astropy__astropy-12907") == "astropy_1776_astropy-12907"
    img = docker_image_for(_inst(), namespace="myorg", tag="v1")
    assert img == "myorg/sweb.eval.x86_64.astropy_1776_astropy-12907:v1"


def test_instruction_includes_problem_and_testbed():
    text = build_instruction(_inst())
    assert "separability_matrix is wrong" in text
    assert "/testbed" in text


def test_parse_pytest_log_statuses():
    log = "PASSED a::t1\nFAILED a::t2\nSKIPPED a::t3\nXFAIL a::t4\nERROR a::t5"
    sm = parse_pytest_log(log)
    assert sm == {"a::t1": "PASSED", "a::t2": "FAILED", "a::t3": "SKIPPED",
                  "a::t4": "PASSED", "a::t5": "ERROR"}


def test_resolution_rule():
    sm = {"f": "PASSED", "p": "PASSED"}
    assert compute_resolution(["f"], ["p"], sm).resolved
    # a failing FAIL_TO_PASS => unresolved
    assert not compute_resolution(["f"], ["p"], {"f": "FAILED", "p": "PASSED"}).resolved
    # a regressed PASS_TO_PASS => unresolved
    assert not compute_resolution(["f"], ["p"], {"f": "PASSED", "p": "FAILED"}).resolved
    # missing test never counts as pass
    rep = compute_resolution(["f"], [], {})
    assert not rep.resolved
    assert rep.fail_to_pass["f"] == MISSING
    # no FAIL_TO_PASS tests => cannot be resolved
    assert not compute_resolution([], ["p"], {"p": "PASSED"}).resolved


def test_resolve_spec_generic_includes_node_ids():
    spec = resolve_spec(_inst(), spec_provider=lambda i: None)
    assert "t/test.py::t_fix" in spec.test_cmd
    assert "t/test.py::t_keep" in spec.test_cmd
    assert "pytest" in spec.test_cmd


# --------------------------------------------------------------------------
# environment grading
# --------------------------------------------------------------------------
def test_environment_grades_resolved():
    inst = _inst()
    sb = FakeSandbox(patch="diff --git a/src b/src\n+fix\n",
                     test_log="PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep")
    env = SWEBenchEnvironment(inst, sb, spec_provider=lambda i: None)
    solved = asyncio.run(_reset_and_solve(env))
    assert solved is True
    assert env.report.resolved
    assert "+fix" in env.model_patch
    assert sb.files["/tmp/test.patch"] == inst.test_patch  # gold tests applied
    assert sb.started


def test_environment_grades_unresolved():
    inst = _inst()
    sb = FakeSandbox(test_log="FAILED t/test.py::t_fix\nPASSED t/test.py::t_keep")
    env = SWEBenchEnvironment(inst, sb, spec_provider=lambda i: None)
    assert asyncio.run(_reset_and_solve(env)) is False
    assert not env.report.resolved


async def _reset_and_solve(env):
    await env.reset()
    solved = await env.is_solved()
    await env.close()
    return solved


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def test_predictions_and_report(tmp_path):
    from aqa.swebench.report import InstanceResult

    results = [
        InstanceResult("a__a-1", resolved=True, model_patch="diff1"),
        InstanceResult("a__a-2", resolved=False, model_patch="diff2"),
        InstanceResult("a__a-3", resolved=False, error="boom"),
    ]
    pred = write_predictions(tmp_path / "predictions.jsonl", results, "m1")
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    assert rows[0] == {"instance_id": "a__a-1", "model_name_or_path": "m1",
                       "model_patch": "diff1"}
    rep = build_report(results, "m1")
    assert rep["total_instances"] == 3
    assert rep["resolved_instances"] == 1
    assert rep["error_instances"] == 1
    assert rep["resolved_rate"] == pytest.approx(1 / 3)
    assert rep["resolved_ids"] == ["a__a-1"]
    out = write_report(tmp_path / "report.json", results, "m1")
    assert json.loads(out.read_text())["resolved_instances"] == 1


# --------------------------------------------------------------------------
# end-to-end runner
# --------------------------------------------------------------------------
def test_run_swebench_end_to_end(tmp_path):
    insts = [
        _inst("a__a-1"),
        _inst("a__a-2"),
    ]
    logs = {
        "a__a-1": "PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep",   # resolved
        "a__a-2": "FAILED t/test.py::t_fix\nPASSED t/test.py::t_keep",   # not
    }

    def factory(inst):
        return FakeSandbox(patch=f"diff {inst.instance_id}\n", test_log=logs[inst.instance_id])

    client = ScriptedClient(["Final Answer: done"])
    agent_result, report = run_swebench(
        insts, client, sandbox_factory=factory, spec_provider=lambda i: None,
        model_name="m1", out_dir=tmp_path, concurrency=2, max_turns=3,
    )
    assert report.resolved_rate == pytest.approx(0.5)
    assert agent_result.mean_score() == pytest.approx(0.5)
    by_id = {r.instance_id: r for r in report.results}
    assert by_id["a__a-1"].resolved
    assert not by_id["a__a-2"].resolved
    assert "diff a__a-1" in by_id["a__a-1"].model_patch
    # artifacts written
    preds = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(preds) == 2
    assert json.loads((tmp_path / "report.json").read_text())["resolved_instances"] == 1

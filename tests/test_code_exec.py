"""Tests for the code_exec scorer (extract/build_program/build_code_exec_scorer)
and its native-harness integration. A fake CodeRunner keeps most tests offline; a
couple exercise real python via LocalCodeRunner (host, trivial + safe programs).
"""

from __future__ import annotations

from aqa.eval.code_exec import (
    CodeExecResult,
    LocalCodeRunner,
    build_code_exec_scorer,
    build_program,
    extract_code,
)
from aqa.eval.harness.native import NativeHarness, resolve_scorer
from aqa.eval.registry import BenchmarkRecipe, CodeExecSpec, ScoringSpec
from aqa.eval.scoring import ScorerInput


class FakeRunner:
    def __init__(self, passed: bool):
        self.result = CodeExecResult(passed=passed, returncode=0 if passed else 1)
        self.program: str | None = None
        self.calls = 0

    def run(self, program: str, *, timeout: float) -> CodeExecResult:
        self.program, self.calls = program, self.calls + 1
        return self.result


def test_extract_code_prefers_fenced_block():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1"


def test_extract_code_falls_back_to_raw():
    assert extract_code("def g():\n    return 2") == "def g():\n    return 2"


def test_build_program_assembles_parts():
    sample = {"prompt": "def add(a, b):\n", "test": "def check(f):\n    assert f(1, 2) == 3",
              "entry_point": "add"}
    spec = CodeExecSpec(include_prompt=True)
    program = build_program(sample, "    return a + b", spec)
    assert "def add(a, b):" in program
    assert "return a + b" in program
    assert "check(add)" in program


def test_scorer_passes_and_fails_via_fake_runner():
    spec = CodeExecSpec()
    ok = build_code_exec_scorer(spec, runner=FakeRunner(True))
    out = ok(ScorerInput(prediction="```python\nx=1\n```", target="", extra={"sample": {}}))
    assert out["score"] == 1.0 and out["metric"] == "code_exec"
    bad = build_code_exec_scorer(spec, runner=FakeRunner(False))
    assert bad(ScorerInput(prediction="x=1", target="", extra={"sample": {}}))["score"] == 0.0


def test_scorer_forwards_assembled_program_to_runner():
    runner = FakeRunner(True)
    scorer = build_code_exec_scorer(CodeExecSpec(include_prompt=True), runner=runner)
    sample = {"prompt": "def f():\n", "test": "def check(g):\n    assert g() == 5", "entry_point": "f"}
    scorer(ScorerInput(prediction="    return 5", target="", extra={"sample": sample}))
    assert "check(f)" in runner.program
    assert runner.calls == 1


def test_resolve_scorer_builds_code_exec_from_spec():
    scorer = resolve_scorer("code_exec", ScoringSpec(scorer="code_exec", code_exec=CodeExecSpec()))
    assert scorer.__name__ == "code_exec"


# -- real execution (host, trivial + safe) ------------------------------------
def test_local_runner_passing_program():
    runner = LocalCodeRunner()
    res = runner.run("def add(a, b):\n    return a + b\nassert add(1, 2) == 3\n", timeout=15)
    assert res.passed and res.returncode == 0


def test_local_runner_failing_program():
    res = LocalCodeRunner().run("assert 1 == 2\n", timeout=15)
    assert not res.passed and res.returncode != 0


def test_native_harness_scores_code_by_execution():
    recipe = BenchmarkRecipe(
        name="humaneval", harness="native", type="generative",
        prompt="{prompt}", target_field="entry_point",
        scoring=ScoringSpec(scorer="code_exec", code_exec=CodeExecSpec(include_prompt=True)),
    )
    samples = [{
        "prompt": "def add(a, b):\n",
        "test": "def check(f):\n    assert f(2, 3) == 5\n",
        "entry_point": "add",
    }]
    harness = NativeHarness()
    result = harness.evaluate_samples(recipe, samples, generate_fn=lambda p: "    return a + b")
    assert result.score == 1.0
    assert result.num_samples == 1

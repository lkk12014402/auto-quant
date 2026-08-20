"""Tests for the llm_judge scorer (build_judge_scorer / parse_judge_score) and
its integration with the native harness. A fake JudgeClient keeps these offline.
"""

from __future__ import annotations

from aqa.eval.harness.native import NativeHarness, resolve_scorer
from aqa.eval.registry import BenchmarkRecipe, JudgeSpec, ScoringSpec
from aqa.eval.scoring import ScorerInput, build_judge_scorer, parse_judge_score


class FakeJudge:
    """Records the last prompt and returns a scripted reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.system, self.user, self.calls = system, user, self.calls + 1
        return self.reply


def test_parse_score_line():
    assert parse_judge_score("SCORE: 8", 10) == 8.0
    assert parse_judge_score("reasoning...\nScore = 3.5\n", 10) == 3.5


def test_parse_falls_back_to_last_number_then_verdict():
    assert parse_judge_score("the answer is 7 out of 10", 10) == 10.0  # last number is 10
    assert parse_judge_score("Verdict: YES, correct.", 10) == 10.0
    assert parse_judge_score("No, this is wrong.", 10) == 0.0
    assert parse_judge_score("uncertain", 10) == 0.0


def test_parse_clamps_to_scale():
    assert parse_judge_score("SCORE: 42", 10) == 10.0
    assert parse_judge_score("SCORE: -5", 10) == 0.0


def test_judge_scorer_normalises_to_unit_interval():
    judge = FakeJudge("SCORE: 6")
    scorer = build_judge_scorer(JudgeSpec(scale=10), client=judge)
    out = scorer(ScorerInput(prediction="p", target="t", extra={"question": "q?"}))
    assert out["score"] == 0.6
    assert out["metric"] == "llm_judge"
    assert out["judge_raw"] == 6.0
    assert judge.calls == 1
    assert "q?" in judge.user
    assert "p" in judge.user


def test_judge_threshold_binarises():
    scorer = build_judge_scorer(JudgeSpec(scale=10, threshold=7), client=FakeJudge("SCORE: 6"))
    assert scorer(ScorerInput(prediction="p", target="t"))["score"] == 0.0
    scorer2 = build_judge_scorer(JudgeSpec(scale=10, threshold=7), client=FakeJudge("SCORE: 9"))
    assert scorer2(ScorerInput(prediction="p", target="t"))["score"] == 1.0


def test_resolve_scorer_builds_judge_from_spec():
    spec = ScoringSpec(scorer="llm_judge", judge=JudgeSpec(scale=5))
    # resolve returns the judge callable; feed it a client-less spec path won't be
    # exercised here (no network) — we assert it is the judge, not a builtin.
    scorer = resolve_scorer("llm_judge", spec)
    assert scorer.__name__ == "llm_judge"


def test_native_harness_uses_judge(monkeypatch):
    judge = FakeJudge("SCORE: 10")
    recipe = BenchmarkRecipe(
        name="openqa_judge", harness="native", type="generative",
        prompt="Q: {question}", target_field="answer",
        scoring=ScoringSpec(scorer="llm_judge", judge=JudgeSpec(scale=10)),
    )
    # Patch resolve_scorer's judge builder to inject the fake client.
    import aqa.eval.harness.native as native_mod

    real_build = native_mod.build_judge_scorer
    monkeypatch.setattr(
        native_mod, "build_judge_scorer",
        lambda spec, client=None: real_build(spec, client=judge),
    )
    harness = NativeHarness()
    samples = [{"question": "2+2?", "answer": "4"}, {"question": "3+3?", "answer": "6"}]
    result = harness.evaluate_samples(recipe, samples, generate_fn=lambda p: "an answer")
    assert result.score == 1.0
    assert result.num_samples == 2
    assert judge.calls == 2

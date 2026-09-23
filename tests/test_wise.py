"""WISE_Verified tests -- no GPU/network; the judge is an injected fake.

Covers the reference quirks that matter for score parity: the ``Score: 0|1``
extraction incl. its bare-number fallback, the retry-then-skip behavior on
unparseable output, the official category boundaries, and the 0.40/0.12 weighted
overall.
"""

from __future__ import annotations

import json

import pytest

from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.eval.wise import (
    PROMPT_ID_RANGES,
    WISE_WEIGHTS,
    WiseScorer,
    build_evaluation_messages,
    calculate_wiscore,
    category_for_prompt_id,
    extract_scores,
    load_records,
    summarize,
)
from aqa.logging_util import AqaError


class _FakeJudge:
    """Scripted raw judge replies; optionally raises instead."""

    def __init__(self, replies=None, error=None):
        self.replies = list(replies or ["Score: 1"])
        self.error = error
        self.calls = []

    def evaluate(self, prompt_data, image_base64):
        self.calls.append((prompt_data["Prompt"], image_base64))
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else ""


def _record(idx, prompt_id, prompt="p", explanation="e", category=None):
    return {"idx": idx, "prompt_id": prompt_id, "prompt": prompt,
            "explanation": explanation,
            "category": category or category_for_prompt_id(prompt_id),
            "subcategory": "sub"}


def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx:04d}.png", "seed": "42"}


# ── reference primitives ─────────────────────────────────────────────────────
def test_category_boundaries():
    assert category_for_prompt_id(1) == "CULTURE"
    assert category_for_prompt_id(400) == "CULTURE"
    assert category_for_prompt_id(401) == "TIME"
    assert category_for_prompt_id(520) == "TIME"
    assert category_for_prompt_id(521) == "SPACE"
    assert category_for_prompt_id(640) == "SPACE"
    assert category_for_prompt_id(641) == "BIOLOGY"
    assert category_for_prompt_id(760) == "BIOLOGY"
    assert category_for_prompt_id(761) == "PHYSICS"
    assert category_for_prompt_id(880) == "PHYSICS"
    assert category_for_prompt_id(881) == "CHEMISTRY"
    assert category_for_prompt_id(1000) == "CHEMISTRY"
    assert category_for_prompt_id(0) is None
    assert category_for_prompt_id(1001) is None


def test_weights_sum_to_one():
    assert sum(WISE_WEIGHTS.values()) == pytest.approx(1.0)
    assert PROMPT_ID_RANGES["CULTURE"] == (1, 400)


def test_extract_scores_variants():
    assert extract_scores("Score: 1") == {"score": 1.0}
    assert extract_scores("**Score: 0**") == {"score": 0.0}
    assert extract_scores("score: 0") == {"score": 0.0}
    assert extract_scores("1") == {"score": 1.0}  # bare single-line fallback
    assert extract_scores("nothing to see") == {}
    assert extract_scores("Score: 2") == {}  # only binary accepted


def test_calculate_wiscore_is_identity():
    assert calculate_wiscore(1) == 1.0
    assert calculate_wiscore(0) == 0.0


def test_build_evaluation_messages_multimodal():
    msgs = build_evaluation_messages(
        {"Prompt": "the pastry", "Explanation": "mooncakes"}, "QUJD"
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    text = msgs[1]["content"][0]["text"]
    assert 'PROMPT: "the pastry"' in text
    assert 'EXPLANATION: "mooncakes"' in text
    assert msgs[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_strip_think():
    from aqa.eval.wise import _strip_think

    assert _strip_think(" thinkinghmm</think>Score: 1") == "Score: 1"
    assert _strip_think("Score: 0</think>") == "Score: 0"


# ── aggregation ──────────────────────────────────────────────────────────────
def test_summarize_official_weighted_overall():
    rows = [
        {"prompt_id": 1, "category": "CULTURE", "score": 1.0},
        {"prompt_id": 2, "category": "CULTURE", "score": 0.0},
        {"prompt_id": 401, "category": "TIME", "score": 1.0},
        {"prompt_id": 521, "category": "SPACE", "score": 1.0},
        {"prompt_id": 641, "category": "BIOLOGY", "score": 1.0},
        {"prompt_id": 761, "category": "PHYSICS", "score": 1.0},
        {"prompt_id": 881, "category": "CHEMISTRY", "score": 1.0},
    ]
    summary = summarize(rows)
    assert summary["categories"]["CULTURE"] == pytest.approx(0.5)
    assert summary["overall"] == pytest.approx(0.4 * 0.5 + 0.12 * 5)
    assert summary["num_images"] == 7
    assert summary["complete"] is True
    assert summary["num_missing"] == 0


def test_summarize_counts_missing_and_incomplete():
    rows = [
        {"prompt_id": 1, "category": "CULTURE", "score": 1.0},
        {"prompt_id": 2, "category": "CULTURE", "score": None},
    ]
    summary = summarize(rows)
    assert summary["num_images"] == 1
    assert summary["num_missing"] == 1
    assert summary["missing_prompt_ids"] == [2]
    assert summary["complete"] is False
    assert summary["overall"] == pytest.approx(0.4 * 1.0)


# ── scorer (fake judge) ──────────────────────────────────────────────────────
def test_scorer_scores_row_and_passes_base64():
    judge = _FakeJudge(["Score: 1"])
    scorer = WiseScorer(judge=judge)
    rows = scorer.evaluate_rows(
        [_image_row(0)], [_record(0, 1)], image_encoder=lambda p: "B64"
    )
    assert rows[0]["prompt_id"] == 1
    assert rows[0]["category"] == "CULTURE"
    assert rows[0]["score"] == 1.0
    assert judge.calls == [("p", "B64")]


def test_scorer_uses_official_category_not_dataset_label():
    # vendored records carry the dataset's descriptive Category ("Cultural knowledge");
    # the aggregation category must come from the prompt_id range instead
    record = _record(0, 1, category="Cultural knowledge")
    scorer = WiseScorer(judge=_FakeJudge(["Score: 1"]))
    rows = scorer.evaluate_rows([_image_row(0)], [record], image_encoder=lambda p: "B64")
    assert rows[0]["category"] == "CULTURE"
    assert rows[0]["source_category"] == "Cultural knowledge"
    assert summarize(rows)["categories"]["CULTURE"] == 1.0


def test_scorer_retries_unparseable_then_succeeds():
    judge = _FakeJudge(["no score here", "Score: 0"])
    scorer = WiseScorer(judge=judge, max_retries=3)
    row = scorer.score_image("/x.png", _record(0, 1), "B64")
    assert row == 0.0
    assert len(judge.calls) == 2


def test_scorer_returns_none_after_retries():
    judge = _FakeJudge(["nope", "still nope", "nope"])
    scorer = WiseScorer(judge=judge, max_retries=3)
    row = scorer.score_image("/x.png", _record(0, 1), "B64")
    assert row is None
    assert len(judge.calls) == 3


def test_scorer_retries_transient_judge_exception():
    class _Flaky:
        def __init__(self):
            self.n = 0

        def evaluate(self, prompt_data, image_base64):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("connection reset")
            return "Score: 1"

    scorer = WiseScorer(judge=_Flaky(), max_retries=2)
    assert scorer.score_image("/x.png", _record(0, 1), "B64") == 1.0


def test_scorer_propagates_aqa_error():
    scorer = WiseScorer(judge=_FakeJudge(error=AqaError("judge down")))
    with pytest.raises(AqaError, match="judge down"):
        scorer.score_image("/x.png", _record(0, 1), "B64")


def test_scorer_skips_rows_without_record():
    scorer = WiseScorer(judge=_FakeJudge())
    rows = scorer.evaluate_rows([_image_row(9)], [_record(0, 1)],
                                image_encoder=lambda p: "B64")
    assert rows == []


def test_write_results(tmp_path):
    scorer = WiseScorer(judge=_FakeJudge(["Score: 1"]))
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0, 1)],
                                image_encoder=lambda p: "B64")
    jsonl, txt = scorer.write_results(rows, tmp_path / "wise_results")
    assert next(json.loads(x) for x in jsonl.read_text().splitlines())["prompt_id"] == 1
    assert "overall:" in txt.read_text()


# ── recipe + harness dispatch ────────────────────────────────────────────────
def test_wise_recipe_loads():
    recipe = BenchmarkRegistry().get("wise")
    assert recipe.harness == "diffusion"
    assert recipe.prompt_field == "prompt"
    assert recipe.scoring.scorer == "wise"
    assert recipe.scoring.wise.judge_model == "Qwen/Qwen3.5-35B-A3B"
    assert recipe.scoring.wise.base_url.startswith("http://")


def test_group_key_separates_wise(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    harness = DiffusionHarness()
    clip = BenchmarkRecipe(name="c", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt")
    wise = BenchmarkRecipe(name="w", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt", scoring={"scorer": "wise"})
    assert harness.group_key(wise) != harness.group_key(clip)


def test_harness_dispatch(tmp_path, monkeypatch):
    fake_rows = [
        {"filename": "0.png", "idx": 0, "prompt_id": 1, "category": "CULTURE",
         "subcategory": "Festival", "prompt": "p", "score": 1.0},
        {"filename": "1.png", "idx": 1, "prompt_id": 2, "category": "CULTURE",
         "subcategory": "Festival", "prompt": "p", "score": 0.0},
    ]
    monkeypatch.setattr(WiseScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: fake_rows)
    recipe = BenchmarkRecipe(
        name="wise", harness="diffusion", type="generative",
        dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
        scoring={"scorer": "wise"},
    )
    result = DiffusionHarness()._result_for(
        recipe, "org/m", tmp_path / "out",
        [{"idx": "0", "prompt": "x", "image_path": "0.png", "seed": "42"}],
        [{"prompt": "p", "prompt_id": 1}],
    )
    assert result.metric == "wise"
    assert result.score == pytest.approx(0.2)  # 0.40 * 0.5
    assert result.num_samples == 2
    assert result.per_sample == [1.0, 0.0]
    assert result.raw["categories"]["CULTURE"] == pytest.approx(0.5)
    assert "wise(" in result.command


def test_harness_dispatch_no_scorable_images_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(WiseScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [])
    recipe = BenchmarkRecipe(name="wise", harness="diffusion",
                             dataset=str(tmp_path / "p.jsonl"),
                             scoring={"scorer": "wise"})
    with pytest.raises(AqaError, match="no generated images"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out", [], [])


# ── vendored data ────────────────────────────────────────────────────────────
def test_vendored_prompts():
    from aqa.eval.wise import DATA_DIR

    records = load_records(DATA_DIR / "prompts.jsonl")
    assert len(records) == 1000
    assert [r["prompt_id"] for r in records] == list(range(1, 1001))
    counts: dict[str, int] = {}
    for r in records:
        counts[category_for_prompt_id(r["prompt_id"])] = counts.get(
            category_for_prompt_id(r["prompt_id"]), 0
        ) + 1
    assert counts == {"CULTURE": 400, "TIME": 120, "SPACE": 120,
                      "BIOLOGY": 120, "PHYSICS": 120, "CHEMISTRY": 120}
    assert all(r["explanation"] for r in records)

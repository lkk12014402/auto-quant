"""CVTG-2K tests -- no GPU/network; OCR + CLIP are injected fakes.

Ports the reference algorithm quirks (single-quote word extraction, closest-word
NED, word-weighted accuracy, and the empty-OCR ``['']`` fallback) so a change in
those semantics fails loudly.
"""

from __future__ import annotations

import json

import pytest

from aqa.eval.cvtg2k import (
    Cvtg2kScorer,
    clipscore_from_features,
    extract_words_from_prompt,
    load_records,
    normalized_edit_distance,
    summarize,
    word_scores,
)
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError


class _FakeOCR:
    """Scripted OCR lines (all outputs lowercased+split by the scorer)."""

    def __init__(self, lines=None, fail=False):
        self.lines = lines or []
        self.fail = fail
        self.calls = []

    def recognize(self, image):
        self.calls.append(image)
        if self.fail:
            raise RuntimeError("ocr boom")
        return list(self.lines)


class _FakeClip:
    """Fixed CLIP score per call."""

    def __init__(self, score=0.3, fail=False):
        self.score_value = score
        self.fail = fail
        self.calls = []

    def score(self, image_path, text):
        self.calls.append((image_path, text))
        if self.fail:
            raise RuntimeError("clip boom")
        return self.score_value


def _record(idx, prompt, subset="CVTG", area=2, index=None):
    return {"idx": idx, "subset": subset, "area": area, "index": index if index is not None else idx,
            "prompt": prompt}


def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx:04d}.png", "seed": "42"}


# ── reference primitives ─────────────────────────────────────────────────────
def test_extract_words_from_prompt_reference_regex():
    # only single-quoted spans count; lowercased and split (punctuation kept)
    assert extract_words_from_prompt("a sign that says 'READ'. And 'Morning!'") == [
        "read", "morning!"
    ]
    assert extract_words_from_prompt('no quotes here "READ"') == []


def test_normalized_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == pytest.approx(1.0, abs=1e-4)
    assert normalized_edit_distance("abc", "") == pytest.approx(0.0, abs=1e-4)
    # one substitution over length 3 -> 2/3
    assert normalized_edit_distance("abc", "abd") == pytest.approx(2 / 3, abs=1e-4)


def test_word_scores_exact_membership_and_closest_ned():
    gt = ["read", "morning"]
    pred = ["read", "mornings", "extra"]
    out = word_scores(gt, pred)
    assert out["total_words"] == 2
    assert out["correct_words"] == 1  # "morning" != "mornings"
    # NED compares against the closest predicted word (difflib, cutoff 0)
    assert out["word_neds"][0] == pytest.approx(1.0, abs=1e-4)
    assert out["word_neds"][1] == pytest.approx(1 - 1 / 8, abs=1e-3)


def test_word_scores_empty_ocr_uses_empty_prediction():
    out = word_scores(["read"], [])
    assert out["correct_words"] == 0
    assert out["word_neds"] == [pytest.approx(0.0, abs=1e-4)]  # 1 - 4/4


def test_clipscore_from_features_x2_5_default_and_zero_to_one():
    import numpy as np

    img = np.array([1.0, 0.0])
    txt = np.array([1.0, 0.0])
    # default is the CLIPScore-paper 2.5x scale (what the reference and the
    # published anchors use); x2_5=False gives the bare 0-1 cosine.
    assert clipscore_from_features(img, txt) == pytest.approx(2.5)
    assert clipscore_from_features(img, txt, x2_5=False) == pytest.approx(1.0)
    # negative cosine is clipped to zero, not flipped
    assert clipscore_from_features(img, np.array([-1.0, 0.0])) == 0.0


# ── scorer (fakes) ───────────────────────────────────────────────────────────
def test_scorer_scores_one_row():
    record = _record(0, "a sign that says 'READ'")
    scorer = Cvtg2kScorer(ocr=_FakeOCR(["READ"]), clip=_FakeClip(0.42))
    rows = scorer.evaluate_rows([_image_row(0)], [record], image_loader=lambda p: object())
    assert len(rows) == 1
    row = rows[0]
    assert row["gt_words"] == ["read"]
    assert row["pred_words"] == ["read"]
    assert row["correct_words"] == 1
    assert row["total_words"] == 1
    assert row["clip_score"] == 0.42
    assert row["subset"] == "CVTG" and row["area"] == 2


def test_scorer_skips_rows_without_record():
    scorer = Cvtg2kScorer(ocr=_FakeOCR(["x"]), clip=_FakeClip())
    rows = scorer.evaluate_rows([_image_row(5)], [_record(0, "'a'")],
                                image_loader=lambda p: object())
    assert rows == []


def test_scorer_ocr_failure_counts_as_zero_words():
    record = _record(0, "says 'word'")
    scorer = Cvtg2kScorer(ocr=_FakeOCR(fail=True), clip=_FakeClip(0.5))
    row = scorer.score_image(object(), record, image_path="/x.png")
    assert row["pred_words"] == []
    assert row["correct_words"] == 0
    assert row["word_neds"] == [pytest.approx(0.0, abs=1e-4)]
    assert row["clip_score"] == 0.5


def test_scorer_clip_failure_scores_zero():
    record = _record(0, "says 'word'")
    scorer = Cvtg2kScorer(ocr=_FakeOCR(["word"]), clip=_FakeClip(fail=True))
    row = scorer.score_image(object(), record, image_path="/x.png")
    assert row["clip_score"] == 0.0


# ── aggregation ──────────────────────────────────────────────────────────────
def test_summarize_word_weighted_accuracy_and_mean_ned():
    rows = [
        {"subset": "CVTG", "area": 2, "total_words": 3, "correct_words": 1,
         "word_neds": [1.0, 0.5, 0.0], "clip_score": 0.2},
        {"subset": "CVTG", "area": 3, "total_words": 1, "correct_words": 1,
         "word_neds": [1.0], "clip_score": 0.4},
    ]
    summary = summarize(rows)
    assert summary["overall"]["word_accuracy"] == pytest.approx(2 / 4)
    assert summary["overall"]["ned"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert summary["overall"]["clipscore"] == pytest.approx(0.3)
    assert summary["area_results"]["CVTG/2"]["word_accuracy"] == pytest.approx(1 / 3)
    assert summary["area_results"]["CVTG/3"]["num_images"] == 1
    assert summary["num_images"] == 2


def test_summarize_empty():
    summary = summarize([])
    assert summary["overall"]["word_accuracy"] == 0.0
    assert summary["overall"]["clipscore"] == 0.0
    assert summary["num_images"] == 0


def test_write_results(tmp_path):
    scorer = Cvtg2kScorer(ocr=_FakeOCR(["read"]), clip=_FakeClip(0.9))
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0, "'read'")],
                                image_loader=lambda p: object())
    jsonl, txt = scorer.write_results(rows, tmp_path / "cvtg2k_results")
    back = [json.loads(x) for x in jsonl.read_text().splitlines()]
    assert back[0]["idx"] == 0
    text = txt.read_text()
    assert "overall:" in text and "word_accuracy=" in text


# ── recipe + harness dispatch ────────────────────────────────────────────────
def test_cvtg2k_recipe_loads():
    recipe = BenchmarkRegistry().get("cvtg2k")
    assert recipe.harness == "diffusion"
    assert recipe.prompt_field == "prompt"
    assert recipe.scoring.scorer == "cvtg2k"
    assert recipe.scoring.cvtg2k.ocr_lang == "en"
    assert recipe.scoring.cvtg2k.clip_model == "ViT-L/14"
    assert recipe.scoring.cvtg2k.clipscore_x2_5 is True


def test_group_key_separates_cvtg2k(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    harness = DiffusionHarness()
    clip = BenchmarkRecipe(name="c", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt")
    cvtg = BenchmarkRecipe(name="v", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt", scoring={"scorer": "cvtg2k"})
    assert harness.group_key(cvtg) != harness.group_key(clip)


def test_harness_dispatch_emits_three_metrics(tmp_path, monkeypatch):
    fake_rows = [
        {"filename": "0.png", "idx": 0, "prompt": "'read'", "subset": "CVTG", "area": 2,
         "index": 0, "gt_words": ["read"], "pred_words": ["read"], "total_words": 1,
         "correct_words": 1, "word_neds": [1.0], "clip_score": 0.8},
        {"filename": "1.png", "idx": 1, "prompt": "'cat'", "subset": "CVTG", "area": 2,
         "index": 1, "gt_words": ["cat"], "pred_words": [], "total_words": 1,
         "correct_words": 0, "word_neds": [0.0], "clip_score": 0.2},
    ]
    monkeypatch.setattr(Cvtg2kScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: fake_rows)
    recipe = BenchmarkRecipe(
        name="cvtg2k", harness="diffusion", type="generative",
        dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
        scoring={"scorer": "cvtg2k"},
    )
    results = DiffusionHarness()._result_for(
        recipe, "org/m", tmp_path / "out",
        [{"idx": "0", "prompt": "x", "image_path": "0.png", "seed": "42"}],
        [{"prompt": "'read'"}],
    )
    by_metric = {r.metric: r for r in results}
    assert set(by_metric) == {"cvtg2k_acc", "cvtg2k_ned", "cvtg2k_clipscore"}
    assert by_metric["cvtg2k_acc"].score == pytest.approx(0.5)
    assert by_metric["cvtg2k_ned"].score == pytest.approx(0.5)
    assert by_metric["cvtg2k_clipscore"].score == pytest.approx(0.5)
    assert by_metric["cvtg2k_acc"].per_sample == [1.0, 0.0]
    assert "area_results" in by_metric["cvtg2k_acc"].raw


def test_harness_dispatch_no_scorable_images_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Cvtg2kScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [])
    recipe = BenchmarkRecipe(name="cvtg2k", harness="diffusion",
                             dataset=str(tmp_path / "p.jsonl"),
                             scoring={"scorer": "cvtg2k"})
    with pytest.raises(AqaError, match="no generated images"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out", [], [])


# ── vendored data ────────────────────────────────────────────────────────────
def test_vendored_prompts():
    from aqa.eval.cvtg2k import DATA_DIR

    records = load_records(DATA_DIR / "prompts.jsonl")
    assert len(records) == 2000
    assert [r["idx"] for r in records] == list(range(2000))
    subsets = {r["subset"] for r in records}
    assert subsets == {"CVTG", "CVTG-Style"}
    assert sum(1 for r in records if r["subset"] == "CVTG") == 1002
    assert sum(1 for r in records if r["subset"] == "CVTG-Style") == 998
    assert sum(len(r["gt_words"]) for r in records) == 16206

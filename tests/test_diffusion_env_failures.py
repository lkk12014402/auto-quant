"""Diffusion env-failure policy tests -- no GPU/network/judge.

An unavailable scorer component (OCR / CLIP / VQA judge / VLM) must FAIL the run
rather than report a ``0.0`` that is really an environment artifact.  These tests
cover, per scorer: (a) every image failed for an environmental reason -> the
harness raises; (b) a partial failure -> the result records the missing count and
is marked incomplete; (c) all images scored -> the numeric result is unchanged.

Fakes mirror ``tests/test_dpgbench.py``: components are injected, nothing heavy
is imported.
"""

from __future__ import annotations

import pytest

from aqa.eval.cvtg2k import Cvtg2kScorer
from aqa.eval.cvtg2k import summarize as cvtg_summarize
from aqa.eval.dpgbench import DpgBenchScorer
from aqa.eval.glyphdraw2 import GlyphDraw2Scorer
from aqa.eval.glyphdraw2 import summarize as glyph_summarize
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe
from aqa.eval.wise import WiseScorer
from aqa.logging_util import AqaError


class _EnvFailOCR:
    """OCR whose engine is unavailable: raises AqaError on every call."""

    def recognize(self, image, *args, **kwargs):
        raise AqaError("PaddleOCR is not installed: pip install paddleocr")


class _OkOCR:
    def __init__(self, lines):
        self.lines = list(lines)

    def recognize(self, image, *args, **kwargs):
        return list(self.lines)


class _OkClip:
    def __init__(self, score=0.5):
        self.score_value = score

    def score(self, image_path, text):
        return self.score_value


class _EnvFailClip:
    def score(self, image_path, text):
        raise AqaError("cvtg2k CLIPScore needs the official OpenAI CLIP: pip install openai-clip")


class _EnvFailJudge:
    """VQA judge whose model is unavailable: raises AqaError on every call."""

    def vqa(self, image, question):
        raise AqaError("modelscope mPLUG VQA stack is unavailable")


def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx:04d}.png", "seed": "42"}


# ── CVTG-2K ──────────────────────────────────────────────────────────────────
def _cvtg_record(idx, prompt="a sign that says 'read'"):
    return {"idx": idx, "subset": "CVTG", "area": 2, "index": idx, "prompt": prompt}


def _cvtg_row(idx, *, scored=True, correct=1, total=1):
    return {
        "filename": f"{idx}.png", "idx": idx, "prompt": "'read'", "subset": "CVTG",
        "area": 2, "index": idx, "gt_words": ["read"], "pred_words": ["read"] if scored else [],
        "total_words": total, "correct_words": correct if scored else 0,
        "word_neds": [1.0] if scored else [0.0],
        "clip_score": 0.8 if scored else 0.0, "scored": scored,
    }


def test_cvtg2k_unavailable_ocr_marks_row_unscored():
    scorer = Cvtg2kScorer(ocr=_EnvFailOCR(), clip=_OkClip(0.5))
    rows = scorer.evaluate_rows([_image_row(0)], [_cvtg_record(0)],
                                image_loader=lambda p: object())
    assert rows[0]["ocr_scored"] is False
    assert rows[0]["scored"] is False
    summary = cvtg_summarize(rows)
    assert summary["num_scored"] == 0
    assert summary["num_env_failed"] == 1
    assert summary["complete"] is False


def test_cvtg2k_all_env_failed_harness_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Cvtg2kScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [_cvtg_row(0, scored=False)])
    recipe = BenchmarkRecipe(name="cvtg2k", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "cvtg2k"})
    with pytest.raises(AqaError, match="every image failed to score"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                       [_image_row(0)], [_cvtg_record(0)])


def test_cvtg2k_partial_env_failure_is_recorded_and_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Cvtg2kScorer, "evaluate_rows",
        lambda self, rows, records, **kw: [_cvtg_row(0), _cvtg_row(1, scored=False)],
    )
    recipe = BenchmarkRecipe(name="cvtg2k", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "cvtg2k"})
    results = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                             [_image_row(0)], [_cvtg_record(0)])
    by_metric = {r.metric: r for r in results}
    # the numeric score comes only from the scored image...
    assert by_metric["cvtg2k_acc"].score == 1.0
    assert by_metric["cvtg2k_acc"].per_sample == [1.0]
    assert by_metric["cvtg2k_acc"].num_samples == 1
    # ...and the partial failure is visible in the written result.
    for r in results:
        assert r.raw["complete"] is False
        assert r.raw["num_missing"] == 1
        assert r.raw["num_env_failed"] == 1


def test_cvtg2k_all_scored_result_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Cvtg2kScorer, "evaluate_rows",
        lambda self, rows, records, **kw: [_cvtg_row(0), _cvtg_row(1)],
    )
    recipe = BenchmarkRecipe(name="cvtg2k", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "cvtg2k"})
    results = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                             [_image_row(0)], [_cvtg_record(0)])
    by_metric = {r.metric: r for r in results}
    assert by_metric["cvtg2k_acc"].score == 1.0
    assert by_metric["cvtg2k_ned"].score == 1.0
    assert by_metric["cvtg2k_clipscore"].score == 0.8
    for r in results:
        assert r.raw["complete"] is True
        assert r.raw["num_missing"] == 0


# ── GlyphDraw2 ───────────────────────────────────────────────────────────────
def _glyph_record(idx, prompt="a poster with “READ”"):
    return {"idx": idx, "kind": "complex", "lang": "en", "prompt": prompt,
            "gt_texts": ["READ"]}


def _glyph_row(idx, *, scored=True):
    return {"filename": f"{idx}.png", "idx": idx, "prompt": "“READ”", "kind": "complex",
            "lang": "en", "gt_texts": ["READ"], "pred_texts": ["READ"] if scored else [],
            "matched": ["READ"], "num_lines": 1,
            "line_acc": [1.0] if scored else [0.0],
            "line_neds": [1.0] if scored else [0.0], "scored": scored}


def test_glyphdraw2_unavailable_ocr_marks_row_unscored():
    scorer = GlyphDraw2Scorer(ocr=_EnvFailOCR())
    rows = scorer.evaluate_rows([_image_row(0)], [_glyph_record(0)],
                                image_loader=lambda p: object())
    assert rows[0]["scored"] is False
    summary = glyph_summarize(rows)
    assert summary["num_scored"] == 0 and summary["num_env_failed"] == 1
    assert summary["complete"] is False


def test_glyphdraw2_all_env_failed_harness_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(GlyphDraw2Scorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [_glyph_row(0, scored=False)])
    recipe = BenchmarkRecipe(name="glyphdraw2", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "glyphdraw2"})
    with pytest.raises(AqaError, match="every image failed to score"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                       [_image_row(0)], [_glyph_record(0)])


def test_glyphdraw2_partial_env_failure_is_recorded_and_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GlyphDraw2Scorer, "evaluate_rows",
        lambda self, rows, records, **kw: [_glyph_row(0), _glyph_row(1, scored=False)],
    )
    recipe = BenchmarkRecipe(name="glyphdraw2", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "glyphdraw2"})
    results = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                             [_image_row(0)], [_glyph_record(0)])
    by_metric = {r.metric: r for r in results}
    assert by_metric["glyphdraw2_acc"].score == 1.0
    assert by_metric["glyphdraw2_acc"].num_samples == 1
    for r in results:
        assert r.raw["complete"] is False and r.raw["num_missing"] == 1


# ── DPG-Bench ────────────────────────────────────────────────────────────────
class _PartialJudge:
    """Raises AqaError on the first image's question, then answers yes."""

    def __init__(self):
        self.calls = 0

    def vqa(self, image, question):
        self.calls += 1
        if self.calls == 1:
            raise AqaError("vLLM judge is not reachable")
        return "yes"


class _FakeImage:
    def crop(self, box):
        return self


def _dpg_question(qid=1, dep=(0,)):
    return {"qid": qid, "question": "Is there a cat?", "answer": "entity (cat)",
            "dependency": list(dep), "category": "entity - whole"}


def _dpg_record(idx):
    return {"idx": idx, "item_id": f"item{idx}", "prompt": "p",
            "questions": [_dpg_question()], "category": "entity - whole"}


def _dpg_row(idx):
    return {"filename": f"{idx}.png", "idx": idx, "item_id": f"item{idx}", "prompt": "p",
            "score": 1.0, "crop_scores": [1.0], "answers": {"1": "yes"},
            "raw_scores": {1: 1.0}, "adj_scores": {1: 1.0},
            "categories": {1: "entity - whole"}}


def test_dpgbench_judge_unavailable_all_raises(tmp_path, monkeypatch):
    scorer = DpgBenchScorer(judge=_EnvFailJudge())
    rows = scorer.evaluate_rows([_image_row(0), _image_row(1)],
                                [_dpg_record(0), _dpg_record(1)],
                                image_loader=lambda p: _FakeImage())
    assert rows == []
    assert scorer.num_env_failed == 2

    def _fake_eval(self, rows, records, **kw):
        self.num_env_failed = 2
        return []

    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows", _fake_eval)
    recipe = BenchmarkRecipe(name="dpgbench", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "dpgbench"})
    with pytest.raises(AqaError, match="VQA judge is unavailable"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                       [_image_row(0), _image_row(1)],
                                       [_dpg_record(0), _dpg_record(1)])


def test_dpgbench_partial_env_failure_is_recorded_and_incomplete(tmp_path, monkeypatch):
    scorer = DpgBenchScorer(judge=_PartialJudge())
    rows = scorer.evaluate_rows([_image_row(0), _image_row(1)],
                                [_dpg_record(0), _dpg_record(1)],
                                image_loader=lambda p: _FakeImage())
    assert [r["idx"] for r in rows] == [1]  # image 0's judge call failed
    assert scorer.num_env_failed == 1 and scorer.num_skipped == 0

    def _fake_eval(self, rows, records, **kw):
        self.num_env_failed = 1
        return [_dpg_row(1)]

    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows", _fake_eval)
    recipe = BenchmarkRecipe(name="dpgbench", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "dpgbench"})
    result = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                            [_image_row(0), _image_row(1)],
                                            [_dpg_record(0), _dpg_record(1)])
    assert result.score == 1.0
    assert result.raw["complete"] is False
    assert result.raw["num_missing"] == 1
    assert result.raw["num_env_failed"] == 1


def test_dpgbench_all_scored_result_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [_dpg_row(0), _dpg_row(1)])
    recipe = BenchmarkRecipe(name="dpgbench", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "dpgbench"})
    result = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                            [_image_row(0)], [_dpg_record(0)])
    assert result.score == 1.0
    assert result.raw["complete"] is True
    assert result.raw["num_missing"] == 0


# ── WISE (consistency: partial miss marked incomplete) ───────────────────────
def test_wise_partial_miss_recorded_as_incomplete(tmp_path, monkeypatch):
    fake_rows = [
        {"filename": "0.png", "idx": 0, "prompt_id": 1, "category": "CULTURE",
         "subcategory": "s", "prompt": "p", "score": 1.0},
        {"filename": "1.png", "idx": 1, "prompt_id": 2, "category": "CULTURE",
         "subcategory": "s", "prompt": "p", "score": None},
    ]
    monkeypatch.setattr(WiseScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: fake_rows)
    recipe = BenchmarkRecipe(name="wise", harness="diffusion", type="generative",
                             dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
                             scoring={"scorer": "wise"})
    result = DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out",
                                            [_image_row(0)], [{"prompt": "p", "prompt_id": 1}])
    assert result.raw["complete"] is False
    assert result.raw["num_missing"] == 1

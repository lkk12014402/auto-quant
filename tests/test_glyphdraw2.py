"""GlyphDraw2 tests -- no GPU/network; OCR is an injected fake.

Pins the AnyText-derived semantics (per-line exact match for Acc, normalized
character edit distance for NED, whole-image closest-line matching) and the
optional char-dict OOV collapsing.
"""

from __future__ import annotations

import json

import pytest

from aqa.eval.glyphdraw2 import (
    GlyphDraw2Scorer,
    extract_gt_texts,
    line_scores,
    load_char_ids,
    load_records,
    normalized_edit_distance,
    summarize,
)
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError


class _FakeOCR:
    """Scripted OCR lines, recording the requested language."""

    def __init__(self, lines=None, fail=False):
        self.lines = lines or []
        self.fail = fail
        self.calls = []

    def recognize(self, image, lang="en"):
        self.calls.append(lang)
        if self.fail:
            raise RuntimeError("ocr boom")
        return list(self.lines)


def _record(idx, prompt, kind="complex", lang="en"):
    return {"idx": idx, "kind": kind, "lang": lang, "prompt": prompt,
            "gt_texts": extract_gt_texts(prompt)}


def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx:04d}.png", "seed": "42"}


# ── ground-truth extraction ──────────────────────────────────────────────────
def test_extract_gt_texts_curly_and_straight_quotes():
    assert extract_gt_texts('a sign that says “READ” and "OPEN"') == ["READ", "OPEN"]
    assert extract_gt_texts("no quoted text") == []


# ── AnyText metric primitives ────────────────────────────────────────────────
def test_normalized_edit_distance_bounds():
    assert normalized_edit_distance("READ", "READ") == pytest.approx(1.0, abs=1e-4)
    assert normalized_edit_distance("READ", "REAX") == pytest.approx(0.75, abs=1e-4)
    assert normalized_edit_distance("READ", "") == pytest.approx(0.0, abs=1e-4)


def test_char_id_collapses_out_of_vocab():
    char2id = {"a": 0, "b": 1}
    # 'X' and 'Y' are both OOV -> collapse to the same id -> distance 0
    assert normalized_edit_distance("aX", "aY", char2id=char2id) == pytest.approx(1.0)
    # without the dict the raw strings differ
    assert normalized_edit_distance("aX", "aY") == pytest.approx(0.5, abs=1e-3)


def test_load_char_ids_file(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("a\nb\n\nc\n")
    assert load_char_ids(path) == {"a": 0, "b": 1, "c": 2}  # blank line skipped


def test_line_scores_exact_match_and_closest_line():
    out = line_scores(["READ", "OPEN"], ["READ", "OPEM", "noise"])
    assert out["num_lines"] == 2
    assert out["line_acc"] == [1.0, 0.0]
    # "OPEN" isn't present; closest line is "OPEM" -> 1 - 1/4
    assert out["line_neds"][1] == pytest.approx(0.75, abs=1e-3)
    assert out["matched"] == ["READ", "OPEM"]


def test_line_scores_empty_ocr():
    out = line_scores(["READ"], [])
    assert out["line_acc"] == [0.0]
    assert out["line_neds"] == [pytest.approx(0.0, abs=1e-4)]


# ── scorer (fake OCR) ────────────────────────────────────────────────────────
def test_scorer_uses_record_language():
    prompt = "a poster with “HELLO”"
    scorer = GlyphDraw2Scorer(ocr=_FakeOCR(["HELLO"]))
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0, prompt, lang="zh")],
                                image_loader=lambda p: object())
    assert rows[0]["lang"] == "zh"
    assert scorer._ocr.calls == ["zh"]
    assert rows[0]["line_acc"] == [1.0]


def test_scorer_ocr_failure_scores_zero():
    scorer = GlyphDraw2Scorer(ocr=_FakeOCR(fail=True))
    row = scorer.score_image(object(), _record(0, "says “word”"), image_path="/x.png")
    assert row["pred_texts"] == []
    assert row["line_acc"] == [0.0]


def test_scorer_skips_rows_without_record():
    scorer = GlyphDraw2Scorer(ocr=_FakeOCR(["x"]))
    rows = scorer.evaluate_rows([_image_row(9)], [_record(0, "“a”")],
                                image_loader=lambda p: object())
    assert rows == []


# ── aggregation ──────────────────────────────────────────────────────────────
def test_summarize_flat_line_means_and_breakdown():
    rows = [
        {"kind": "complex", "lang": "en", "line_acc": [1.0, 0.0], "line_neds": [1.0, 0.5]},
        {"kind": "poster", "lang": "en", "line_acc": [1.0], "line_neds": [1.0]},
    ]
    summary = summarize(rows)
    assert summary["overall"]["acc"] == pytest.approx(2 / 3)
    assert summary["overall"]["ned"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert summary["overall"]["num_lines"] == 3
    assert summary["breakdown"]["complex/en"]["acc"] == pytest.approx(0.5)
    assert summary["breakdown"]["poster/en"]["num_images"] == 1


def test_summarize_empty():
    summary = summarize([])
    assert summary["overall"]["acc"] == 0.0
    assert summary["num_images"] == 0


def test_write_results(tmp_path):
    scorer = GlyphDraw2Scorer(ocr=_FakeOCR(["READ"]))
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0, "“READ”")],
                                image_loader=lambda p: object())
    jsonl, txt = scorer.write_results(rows, tmp_path / "glyphdraw2_results")
    assert next(json.loads(x) for x in jsonl.read_text().splitlines())["idx"] == 0
    assert "overall:" in txt.read_text()


# ── recipe + harness dispatch ────────────────────────────────────────────────
def test_glyphdraw2_recipe_loads():
    recipe = BenchmarkRegistry().get("glyphdraw2")
    assert recipe.harness == "diffusion"
    assert recipe.prompt_field == "prompt"
    assert recipe.scoring.scorer == "glyphdraw2"
    assert recipe.scoring.glyphdraw2.device == "auto"


def test_group_key_separates_glyphdraw2(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    harness = DiffusionHarness()
    clip = BenchmarkRecipe(name="c", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt")
    gd = BenchmarkRecipe(name="g", harness="diffusion", dataset=str(prompts),
                         prompt_field="prompt", scoring={"scorer": "glyphdraw2"})
    assert harness.group_key(gd) != harness.group_key(clip)


def test_harness_dispatch_emits_two_metrics(tmp_path, monkeypatch):
    fake_rows = [
        {"filename": "0.png", "idx": 0, "prompt": "“READ”", "kind": "complex", "lang": "en",
         "gt_texts": ["READ"], "pred_texts": ["READ"], "matched": ["READ"],
         "num_lines": 1, "line_acc": [1.0], "line_neds": [1.0]},
        {"filename": "1.png", "idx": 1, "prompt": "“OPEN”", "kind": "complex", "lang": "en",
         "gt_texts": ["OPEN"], "pred_texts": [], "matched": [""],
         "num_lines": 1, "line_acc": [0.0], "line_neds": [0.0]},
    ]
    monkeypatch.setattr(GlyphDraw2Scorer, "evaluate_rows",
                        lambda self, rows, records, **kw: fake_rows)
    recipe = BenchmarkRecipe(
        name="glyphdraw2", harness="diffusion", type="generative",
        dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
        scoring={"scorer": "glyphdraw2"},
    )
    results = DiffusionHarness()._result_for(
        recipe, "org/m", tmp_path / "out",
        [{"idx": "0", "prompt": "x", "image_path": "0.png", "seed": "42"}],
        [{"prompt": "“READ”"}],
    )
    by_metric = {r.metric: r for r in results}
    assert set(by_metric) == {"glyphdraw2_acc", "glyphdraw2_ned"}
    assert by_metric["glyphdraw2_acc"].score == pytest.approx(0.5)
    assert by_metric["glyphdraw2_ned"].score == pytest.approx(0.5)
    assert by_metric["glyphdraw2_acc"].per_sample == [1.0, 0.0]


def test_harness_dispatch_no_scorable_images_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(GlyphDraw2Scorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [])
    recipe = BenchmarkRecipe(name="glyphdraw2", harness="diffusion",
                             dataset=str(tmp_path / "p.jsonl"),
                             scoring={"scorer": "glyphdraw2"})
    with pytest.raises(AqaError, match="no generated images"):
        DiffusionHarness()._result_for(recipe, "org/m", tmp_path / "out", [], [])


# ── vendored data ────────────────────────────────────────────────────────────
def test_vendored_prompts():
    from aqa.eval.glyphdraw2 import DATA_DIR

    records = load_records(DATA_DIR / "prompts.jsonl")
    assert len(records) == 440
    assert [r["idx"] for r in records] == list(range(440))
    kinds = {(r["kind"], r["lang"]) for r in records}
    assert kinds == {("complex", "en"), ("complex", "zh"), ("poster", "en"), ("poster", "zh")}
    assert sum(1 for r in records if r["kind"] == "complex") == 200
    assert sum(1 for r in records if r["kind"] == "poster") == 240
    assert all(r["gt_texts"] for r in records)

"""GenEval scoring tests -- no GPU/network; detector + CLIP are injected fakes."""

from __future__ import annotations

import json

import pytest

from aqa.eval.geneval import (
    COUNTING_THRESHOLD,
    MAX_OBJECTS,
    THRESHOLD,
    GenevalScorer,
    compute_iou,
    evaluate,
    filter_detections,
    load_classnames,
    relative_position,
    summarize,
)
from aqa.eval.registry import BenchmarkRegistry
from aqa.logging_util import AqaError


def _det(x0, y0, x1, y1, score):
    """A ([x0, y0, x1, y1, score], mask) detection (mask unused by the logic)."""
    return ([x0, y0, x1, y1, score], None)


def _meta(tag, include, exclude=None, prompt="p"):
    m = {"tag": tag, "include": include, "prompt": prompt}
    if exclude:
        m["exclude"] = exclude
    return m


def _classify(colors_by_class):
    """Fake zero-shot color classifier: scripted colors per classname."""
    def classify(image, detections, classname):
        return list(colors_by_class.get(classname, []))[: len(detections)]

    return classify


# ── geometry ─────────────────────────────────────────────────────────────────
def test_compute_iou():
    box = [0, 0, 10, 10, 0.9]
    assert compute_iou(box, box) == 1.0
    assert compute_iou(box, [100, 100, 110, 110, 0.5]) == 0.0
    # +1 pixel area convention: [0,0,1,1] vs [1,1,2,2] share exactly one pixel
    assert compute_iou([0, 0, 1, 1], [1, 1, 2, 2]) == pytest.approx(1 / 7)


def test_relative_position_cardinals_and_dead_zone():
    a = [0, 0, 10, 10]
    b = [100, 0, 110, 10]
    assert relative_position(a, b) == {"left of"}
    assert relative_position(b, a) == {"right of"}
    assert relative_position([0, 0, 10, 10], [0, 100, 10, 110]) == {"above"}
    assert relative_position([0, 100, 10, 110], [0, 0, 10, 10]) == {"below"}
    # diagonal: both relations
    assert relative_position([0, 0, 10, 10], [100, 100, 110, 110]) == {"left of", "above"}
    # dead zone: identical boxes -> centers coincide -> no relation
    assert relative_position([0, 0, 10, 10], [0, 0, 10, 10]) == set()
    # a small offset relative to the boxes' size is shrunk past the 0.5 cutoff
    assert relative_position([0, 0, 10, 10], [5, 0, 15, 10]) == {"left of"}


def test_relative_position_threshold_shrinks_by_dimensions():
    # offset 40, dims sum 200 -> revised 40 - 0.1*200 = 20, norm 40 -> dx = -0.5
    assert relative_position([0, 0, 100, 100], [40, 0, 140, 100]) == set()
    # a wider gap clears the boundary
    assert relative_position([0, 0, 100, 100], [200, 0, 300, 100]) == {"left of"}


# ── detection filtering ──────────────────────────────────────────────────────
def test_filter_detections_threshold_is_strict():
    dets = {"dog": [_det(0, 0, 5, 5, 0.3), _det(0, 0, 5, 5, 0.31)]}
    out = filter_detections(dets, threshold=0.3)
    # score == threshold is dropped (official uses strictly-greater)
    assert len(out["dog"]) == 1 and out["dog"][0][0][4] == pytest.approx(0.31)


def test_filter_detections_counting_threshold():
    dets = {"clock": [_det(0, 0, 5, 5, 0.85), _det(10, 0, 15, 5, 0.95)]}
    assert "clock" in filter_detections(dets, threshold=THRESHOLD)  # 0.85 passes 0.3
    out = filter_detections(dets, threshold=COUNTING_THRESHOLD)
    assert len(out["clock"]) == 1 and out["clock"][0][0][4] == 0.95  # 0.85 dropped at 0.9


def test_filter_detections_sort_cap_and_exact_dup_dedup():
    dets = [_det(i, 0, i + 5, 5, 0.5 + i * 0.01) for i in range(MAX_OBJECTS + 2)]
    out = filter_detections({"dog": dets})
    assert len(out["dog"]) == MAX_OBJECTS
    scores = [d[0][4] for d in out["dog"]]
    assert scores == sorted(scores, reverse=True)
    # identical boxes (IoU == 1.0) are deduped; near-duplicates are kept
    dup = [_det(0, 0, 5, 5, 0.9), _det(0, 0, 5, 5, 0.8), _det(0, 0, 6, 5, 0.7)]
    out = filter_detections({"dog": dup})
    assert [d[0][4] for d in out["dog"]] == [0.9, 0.7]


# ── include/exclude evaluation per tag ───────────────────────────────────────
def test_evaluate_single_object():
    meta = _meta("single_object", [{"class": "dog", "count": 1}])
    ok, reason = evaluate(None, {"dog": [_det(0, 0, 5, 5, 0.9)]}, meta,
                          classify_color=_classify({}))
    assert ok and reason == ""
    ok, reason = evaluate(None, {}, meta, classify_color=_classify({}))
    assert not ok and "expected dog>=1, found 0" in reason


def test_evaluate_two_object_multi_include():
    meta = _meta("two_object", [{"class": "dog", "count": 1}, {"class": "cat", "count": 1}])
    objects = {"dog": [_det(0, 0, 5, 5, 0.9)], "cat": [_det(10, 0, 15, 5, 0.8)]}
    assert evaluate(None, objects, meta, classify_color=_classify({}))[0]
    # one include missing -> the whole AND fails
    assert not evaluate(None, {"dog": objects["dog"]}, meta, classify_color=_classify({}))[0]


def test_evaluate_counting_include_and_exclude():
    meta = _meta("counting", [{"class": "clock", "count": 2}],
                 exclude=[{"class": "clock", "count": 3}])
    clocks = [_det(i * 10, 0, i * 10 + 5, 5, 0.95) for i in range(3)]
    assert evaluate(None, {"clock": clocks[:2]}, meta, classify_color=_classify({}))[0]
    ok, reason = evaluate(None, {"clock": clocks[:1]}, meta, classify_color=_classify({}))
    assert not ok and "expected clock>=2, found 1" in reason
    ok, reason = evaluate(None, {"clock": clocks}, meta, classify_color=_classify({}))
    assert not ok and "expected clock<3, found 3" in reason  # exclude triggers


def test_evaluate_colors_uses_classifier():
    meta = _meta("colors", [{"class": "fire hydrant", "count": 1, "color": "blue"}])
    objects = {"fire hydrant": [_det(0, 0, 5, 5, 0.9)]}
    ok, _ = evaluate(None, objects, meta,
                     classify_color=_classify({"fire hydrant": ["blue"]}))
    assert ok
    ok, reason = evaluate(None, objects, meta,
                          classify_color=_classify({"fire hydrant": ["red"]}))
    assert not ok and "expected blue fire hydrant>=1, found 0 blue" in reason


def test_evaluate_position_against_group_zero():
    # official shape: dog positioned relative to include-group 0 (teddy bear)
    meta = _meta("position", [{"class": "teddy bear", "count": 1},
                              {"class": "dog", "count": 1, "position": ["right of", 0]}])
    bear = {"teddy bear": [_det(0, 0, 10, 10, 0.9)]}
    dog_right = {"dog": [_det(100, 0, 110, 10, 0.9)]}
    ok, _ = evaluate(None, {**bear, **dog_right}, meta, classify_color=_classify({}))
    assert ok
    dog_left = {"dog": [_det(-100, 0, -90, 10, 0.9)]}
    ok, reason = evaluate(None, {**bear, **dog_left}, meta, classify_color=_classify({}))
    assert not ok and "expected dog right of target" in reason
    # no target: group 0 unmatched -> position cannot be checked
    ok, reason = evaluate(None, dog_right, meta, classify_color=_classify({}))
    assert not ok
    assert "no target for dog to be right of" in reason


def test_evaluate_exclude_only():
    meta = _meta("single_object", [{"class": "dog", "count": 1}],
                 exclude=[{"class": "cat", "count": 1}])
    objects = {"dog": [_det(0, 0, 5, 5, 0.9)], "cat": [_det(10, 0, 15, 5, 0.8)]}
    ok, reason = evaluate(None, objects, meta, classify_color=_classify({}))
    assert not ok and "expected cat<1, found 1" in reason


# ── aggregation (summary_scores.py formula) ──────────────────────────────────
def test_summarize_is_unweighted_mean_of_task_means():
    rows = (
        [{"tag": "single_object", "correct": i == 0} for i in range(4)]  # mean 0.25
        + [{"tag": "colors", "correct": True}]                            # mean 1.0
    )
    summary = summarize(rows)
    assert summary["task_means"] == {"single_object": 0.25, "colors": 1.0}
    # unweighted mean of task means, NOT the global mean (2/5 = 0.4)
    assert summary["overall"] == pytest.approx(0.625)
    assert summary["num_images"] == 5


# ── scorer with faked detector / CLIP ────────────────────────────────────────
def _fake_scorer(monkeypatch, per_class, colors=None):
    scorer = GenevalScorer()
    monkeypatch.setattr(GenevalScorer, "detect",
                        lambda self, filepath: (None, per_class))
    if colors is not None:
        monkeypatch.setattr(GenevalScorer, "classify_colors",
                            lambda self, image, dets, classname: colors(dets, classname))
    return scorer


def test_scorer_evaluate_image_row_shape_and_counting_threshold(monkeypatch):
    per_class = {"clock": [_det(0, 0, 5, 5, 0.85), _det(10, 0, 15, 5, 0.95)]}
    scorer = _fake_scorer(monkeypatch, per_class)
    # counting: 0.85 dropped at the 0.9 threshold -> only one clock found
    row = scorer.evaluate_image(
        "x.png", _meta("counting", [{"class": "clock", "count": 2}]))
    assert set(row) == {"filename", "tag", "prompt", "correct", "reason",
                        "metadata", "details"}
    assert row["correct"] is False
    assert "expected clock>=2, found 1" in row["reason"]
    details = json.loads(row["details"])
    assert len(details["clock"]) == 1 and details["clock"][0][4] == 0.95
    assert json.loads(row["metadata"])["tag"] == "counting"
    # same detections pass a single_object prompt at the 0.3 threshold
    row = scorer.evaluate_image(
        "x.png", _meta("single_object", [{"class": "clock", "count": 2}]))
    assert row["correct"] is True


def test_scorer_color_path_uses_clip(monkeypatch):
    per_class = {"fire hydrant": [_det(0, 0, 5, 5, 0.9)]}
    scorer = _fake_scorer(monkeypatch, per_class,
                          colors=lambda dets, classname: ["blue"] * len(dets))
    row = scorer.evaluate_image(
        "x.png", _meta("colors", [{"class": "fire hydrant", "count": 1, "color": "blue"}]))
    assert row["correct"] is True


def test_evaluate_directory_official_layout(monkeypatch, tmp_path):
    scorer = _fake_scorer(monkeypatch, {"dog": [_det(0, 0, 5, 5, 0.9)]})
    for idx in range(2):
        prompt_dir = tmp_path / f"{idx:04d}"
        (prompt_dir / "samples").mkdir(parents=True)
        (prompt_dir / "metadata.jsonl").write_text(json.dumps(
            _meta("single_object", [{"class": "dog", "count": 1}], prompt=f"p{idx}")))
        (prompt_dir / "samples" / "0.png").write_bytes(b"x")
        (prompt_dir / "samples" / "1.png").write_bytes(b"x")
    (tmp_path / "0000" / "samples" / "notes.txt").write_text("skip me")
    (tmp_path / "not-a-prompt").mkdir()

    rows = scorer.evaluate_directory(tmp_path)
    assert len(rows) == 4
    assert all(r["correct"] for r in rows)
    assert rows[0]["filename"].endswith("0000/samples/0.png")
    out = scorer.write_results(rows, tmp_path / "results.jsonl")
    back = [json.loads(x) for x in out.read_text().splitlines()]
    assert back == rows


# ── vendored data + recipe ───────────────────────────────────────────────────
def test_vendored_classnames():
    names = load_classnames()
    assert len(names) == 80
    assert names[0] == "person" and names[-1] == "toothbrush"


def test_geneval_recipe_loads_and_resolves_repo_relative_dataset(tmp_path, monkeypatch):
    recipe = BenchmarkRegistry().get("geneval")
    assert recipe.harness == "diffusion"
    assert recipe.type == "generative"
    assert recipe.images_per_prompt == 4
    assert recipe.prompt_field == "prompt"
    assert recipe.scoring.scorer == "geneval"
    assert recipe.scoring.geneval.detector_model  # defaults filled

    # the repo-relative dataset resolves even from an unrelated cwd
    from aqa.eval.harness.diffusion import DiffusionHarness

    monkeypatch.chdir(tmp_path)
    rows = DiffusionHarness(limit=2)._load_prompt_rows(recipe)
    assert len(rows) == 2
    assert rows[0]["tag"] == "single_object"
    assert rows[0]["prompt"] == "a photo of a bench"
    assert rows[0]["include"] == [{"class": "bench", "count": 1}]


def test_geneval_empty_results_raise(monkeypatch, tmp_path):
    scorer = _fake_scorer(monkeypatch, {})
    assert scorer.evaluate_directory(tmp_path) == []
    from aqa.eval.harness.diffusion import DiffusionHarness
    from aqa.eval.registry import BenchmarkRecipe

    recipe = BenchmarkRecipe(
        name="geneval", harness="diffusion", type="generative",
        dataset=str(tmp_path / "prompts.jsonl"), prompt_field="prompt",
        scoring={"scorer": "geneval"},
    )
    with pytest.raises(AqaError, match="no generated images"):
        DiffusionHarness()._result_geneval(recipe, "org/m", tmp_path / "out")


# ── transformers 5.x get_*_features regression ───────────────────────────────
def test_as_tensor_duck_typing_pooler_output():
    """transformers 5.x returns BaseModelOutputWithPooling (no .norm); older
    versions return a bare tensor. Both must work (colors-tag crash tonight)."""
    from types import SimpleNamespace

    from aqa.eval.geneval import GenevalScorer

    class FakeTensor:  # torch-tensor stand-in: has .norm (numpy arrays do NOT)
        def norm(self, *a, **k):
            return 1.0

    t = FakeTensor()
    assert GenevalScorer._as_tensor(t) is t  # bare-tensor path
    wrapped = SimpleNamespace(pooler_output=t)
    assert GenevalScorer._as_tensor(wrapped) is t  # 5.x output-object path


def test_masked_crop_accepts_float_and_bool_masks():
    """v3 crash regression: binary maps arrive float/bool; PIL composite needs 1/L/F."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    from aqa.eval.geneval import GenevalScorer

    img = Image.new("RGB", (32, 32), color="#102030")
    box = [0, 0, 31, 31, 0.9]
    for m in [np.zeros((32, 32), dtype=np.float32), np.zeros((32, 32), dtype=bool),
              np.zeros((32, 32), dtype=np.uint8) * 255]:
        out = GenevalScorer._masked_crop(img, box, m)
        assert out.size == (32, 32)


def test_classname_aliases_mmdet_to_geneval():
    """HF Mask2Former id2label uses mmdet names (motorbike/tvmonitor/sofa...);
    GenEval expects COCO-official names (motorcycle/tv/couch...)."""
    from types import SimpleNamespace

    from aqa.eval.geneval import GenevalScorer

    scorer = GenevalScorer()
    scorer._detector = SimpleNamespace(config=SimpleNamespace(
        id2label={0: "motorbike", 1: "tvmonitor", 2: "sofa", 3: "pottedplant",
                  4: "keyboard", 5: "mouse", 6: "remote", 7: "aeroplane",
                  8: "diningtable", 9: "person"}))
    assert scorer._classname_for(0) == "motorcycle"
    assert scorer._classname_for(1) == "tv"
    assert scorer._classname_for(2) == "couch"
    assert scorer._classname_for(3) == "potted plant"
    assert scorer._classname_for(4) == "computer keyboard"
    assert scorer._classname_for(5) == "computer mouse"
    assert scorer._classname_for(6) == "tv remote"
    assert scorer._classname_for(7) == "airplane"
    assert scorer._classname_for(8) == "dining table"
    assert scorer._classname_for(9) == "person"  # unchanged

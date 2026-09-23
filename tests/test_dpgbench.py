"""DPG-Bench tests -- no GPU/network; the VQA judge and image loading are faked."""

from __future__ import annotations

import json

import pytest

from aqa.eval.dpgbench import (
    DpgBenchScorer,
    convert_dpg_csv,
    load_records,
    score_image,
    summarize,
)
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError


class _FakeImage:
    """Duck-typed PIL stand-in: crop returns a tagged copy (nothing else needed)."""

    def __init__(self, tag="full"):
        self.tag = tag

    def crop(self, box):
        return _FakeImage(tag=f"crop{box}")


class _FakeJudge:
    """Scripted VQA answers keyed by question text (default 'yes')."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def vqa(self, image, question):
        self.calls.append((image.tag, question))
        return self.answers.get(question, "yes")


def _record(idx, questions, prompt="p"):
    return {"idx": idx, "item_id": f"item{idx}", "prompt": prompt,
            "questions": questions, "category": questions[0]["category"]}


def _q(qid, question, dep, category="entity - whole"):
    return {"qid": qid, "question": question, "answer": f"{category} (x)",
            "dependency": dep, "category": category}


# ── converter ────────────────────────────────────────────────────────────────
def test_convert_dpg_csv(tmp_path):
    csv_path = tmp_path / "dpg_bench.csv"
    csv_path.write_text(
        "item_id,text,keywords,proposition_id,dependency,category_broad,"
        "category_detailed,tuple,question_natural_language\n"
        'a,"prompt A",,1,0,entity,whole,entity - whole (space),Is there space?\n'
        'a,"prompt A",,2,1,entity,whole,entity - whole (man),Is there a man?\n'
        'b,"prompt B",,1,0,attribute,color,attribute - color (red),Is it red?\n'
    )
    records = convert_dpg_csv(csv_path, tmp_path / "prompts.jsonl")
    # the first data row is a real proposition, so it is kept (see the module docstring)
    assert [r["item_id"] for r in records] == ["a", "b"]
    qa = records[0]["questions"]
    assert [q["qid"] for q in qa] == [1, 2]
    assert qa[1]["dependency"] == [1]  # resolvable inside the record
    assert qa[0]["category"] == "entity - whole"
    assert records[0]["prompt"] == "prompt A"
    assert records[1]["questions"][0]["category"] == "attribute - color"
    # round-trip
    assert load_records(tmp_path / "prompts.jsonl") == records


def test_convert_dpg_csv_multi_dependency(tmp_path):
    csv_path = tmp_path / "dpg_bench.csv"
    csv_path.write_text(
        "item_id,text,keywords,proposition_id,dependency,category_broad,"
        "category_detailed,tuple,question_natural_language\n"
        'a,"prompt A",,1,0,entity,whole,entity - whole (x),Q1?\n'
        'a,"prompt A",,2,"1, 3",entity,whole,entity - whole (y),Q2?\n'
    )
    records = convert_dpg_csv(csv_path, tmp_path / "p.jsonl")
    assert [r["item_id"] for r in records] == ["a"]
    assert [q["qid"] for q in records[0]["questions"]] == [1, 2]
    assert records[0]["questions"][1]["dependency"] == [1, 3]


def test_convert_dpg_csv_keeps_the_vendored_data_consistent():
    """No prompt may carry a dependency on a question it does not have.

    This is what the removed first-data-row skip used to break: on the vendored
    CSV every item starts at proposition_id 1, so dropping one row left 8
    dangling edges on the first item and upstream's KeyError skipped its image
    forever.
    """
    from aqa.eval.dpgbench import DATA_DIR, load_records

    records = load_records(DATA_DIR / "prompts.jsonl")
    assert len(records) == 1065
    assert sum(len(r["questions"]) for r in records) == 14392
    dangling = {
        (r["item_id"], parent)
        for r in records
        for q in r["questions"]
        for parent in q["dependency"]
        if parent != 0 and parent not in {qq["qid"] for qq in r["questions"]}
    }
    assert dangling == set()


# ── per-image scoring (fake judge) ───────────────────────────────────────────
def test_score_image_all_yes():
    rec = _record(0, [_q(1, "q1?", [0]), _q(2, "q2?", [1])])
    result = score_image(_FakeJudge(), _FakeImage(), rec, resolution=1024)
    assert result["score"] == 1.0
    assert result["adj_scores"] == {1: 1.0, 2: 1.0}
    assert result["answers"] == {1: "yes", 2: "yes"}
    assert len(result["crop_scores"]) == 1  # pic_num=1: single top-left crop


def test_score_image_dependency_zeroing():
    # parent (q1) answers 'no' -> child (q2) zeroed in adjusted scores, raw kept
    rec = _record(0, [_q(1, "parent?", [0]), _q(2, "child?", [1])])
    result = score_image(_FakeJudge({"parent?": "no"}), _FakeImage(), rec,
                         resolution=1024)
    assert result["raw_scores"] == {1: 0.0, 2: 1.0}
    assert result["adj_scores"] == {1: 0.0, 2: 0.0}
    assert result["score"] == 0.0


def test_score_image_mixed_answers():
    rec = _record(0, [_q(1, "a?", [0]), _q(2, "b?", [0]), _q(3, "c?", [0])])
    result = score_image(_FakeJudge({"a?": "no"}), _FakeImage(), rec,
                         resolution=1024)
    assert result["score"] == pytest.approx(2 / 3)


def test_score_image_dangling_dependency_does_not_zero():
    """A parent the record does not carry cannot be shown to have answered 'no'."""
    rec = _record(0, [_q(2, "child?", [1])])  # parent qid 1 absent
    result = score_image(_FakeJudge(), _FakeImage(), rec, resolution=1024)
    assert result["score"] == 1.0  # judge defaults to 'yes'; no phantom zeroing


def test_score_image_dangling_parent_does_not_rescue_a_no():
    """The surviving parent still zeroes its child (normal dependency behavior)."""
    rec = _record(0, [_q(1, "parent?", [0]), _q(2, "child?", [1, 9])])
    result = score_image(_FakeJudge({"parent?": "no"}), _FakeImage(), rec, resolution=1024)
    assert result["adj_scores"][2] == 0.0
    assert result["raw_scores"][2] == 1.0  # categories use the raw score


def test_score_image_pic_num_quadrants():
    judge = _FakeJudge()
    rec = _record(0, [_q(1, "q?", [0])])
    result = score_image(judge, _FakeImage(), rec, resolution=512, pic_num=4)
    assert len(result["crop_scores"]) == 4
    # all four official quadrant crops were judged
    tags = {tag for tag, _ in judge.calls}
    assert "crop(0, 0, 512, 512)" in tags
    assert "crop(512, 512, 1024, 1024)" in tags


# ── evaluate_rows + summarize ────────────────────────────────────────────────
def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx:04d}.png",
            "seed": "42"}


def test_evaluate_rows_skips_failures_and_gaps():
    recs = [
        _record(0, [_q(1, "ok?", [0])]),
        # a parent the record does not carry is ignored, not a KeyError: the
        # image is scored (regression guard for the old first-row-skip fallout)
        _record(1, [_q(2, "dangling?", [1])]),
        _record(2, [_q(1, "unreadable?", [0])]),
    ]
    judge = _FakeJudge()

    def loader(path):
        if path.endswith("0002.png"):
            raise OSError("unreadable image")
        return _FakeImage()

    scorer = DpgBenchScorer(judge=judge)
    rows = scorer.evaluate_rows(
        [_image_row(0), _image_row(1), _image_row(2), _image_row(3)],  # idx 3: no record
        recs, image_loader=loader,
    )
    assert [r["idx"] for r in rows] == [0, 1]  # idx 2 fails, idx 3 has no record
    assert rows[0]["score"] == 1.0
    assert rows[0]["item_id"] == "item0"
    assert rows[0]["categories"] == {1: "entity - whole"}
    assert rows[1]["score"] == 1.0  # dangling parent ignored -> all-yes image scores
    assert scorer.num_skipped == 2  # the unreadable image + the record-less row


def test_summarize_official_formula_quirk():
    """Categories aggregate RAW (pre-dependency) scores; overall uses adjusted."""
    rec = _record(0, [_q(1, "parent?", [0]), _q(2, "child?", [1])],
                  prompt="p")
    judge = _FakeJudge({"parent?": "no"})
    scorer = DpgBenchScorer(judge=judge)
    rows = scorer.evaluate_rows([_image_row(0)], [rec],
                                image_loader=lambda path: _FakeImage())
    summary = summarize(rows)
    # headline: per-image adjusted mean -> (0 + 0) / 2 = 0
    assert summary["overall"] == 0.0
    # category breakdown: RAW scores -> (0 + 1) / 2 = 0.5 (upstream quirk)
    assert summary["categories"]["entity - whole"] == pytest.approx(0.5)
    assert summary["categories_l1"]["entity"] == pytest.approx(0.5)
    assert summary["num_images"] == 1


def test_summarize_multi_image_mean():
    recs = [_record(0, [_q(1, "a?", [0])], prompt="p0"),
            _record(1, [_q(1, "b?", [0])], prompt="p1")]
    judge = _FakeJudge({"a?": "no"})
    scorer = DpgBenchScorer(judge=judge)
    rows = scorer.evaluate_rows([_image_row(0), _image_row(1)], recs,
                                image_loader=lambda path: _FakeImage())
    summary = summarize(rows)
    assert summary["overall"] == pytest.approx(0.5)


def test_write_results_official_txt_format(tmp_path):
    scorer = DpgBenchScorer(judge=_FakeJudge())
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0, [_q(1, "q?", [0])])],
                                image_loader=lambda path: _FakeImage())
    jsonl, txt = scorer.write_results(rows, tmp_path / "dpgbench_results")
    # official line: "<image_path>, <crop scores>, <average>"
    assert txt.read_text() == "/x/0000.png, 1.0, 1.0\n"
    back = [json.loads(x) for x in jsonl.read_text().splitlines()]
    assert back[0]["idx"] == 0


# ── recipe + harness dispatch ────────────────────────────────────────────────
def test_dpgbench_recipe_loads():
    recipe = BenchmarkRegistry().get("dpgbench")
    assert recipe.harness == "diffusion"
    assert recipe.images_per_prompt == 1
    assert recipe.prompt_field == "prompt"
    assert recipe.scoring.scorer == "dpgbench"
    assert recipe.scoring.dpgbench.vqa_model  # defaults filled
    assert recipe.scoring.dpgbench.resolution == 1024


def test_group_key_separates_dpgbench(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": "x", "questions": []}) + "\n")
    harness = DiffusionHarness()
    clip = BenchmarkRecipe(name="c", harness="diffusion", dataset=str(prompts),
                           prompt_field="prompt")
    dpg = BenchmarkRecipe(name="d", harness="diffusion", dataset=str(prompts),
                          prompt_field="prompt", scoring={"scorer": "dpgbench"})
    assert harness.group_key(dpg) != harness.group_key(clip)


def _dpg_recipe(prompts_file):
    return BenchmarkRecipe(
        name="dpgbench", harness="diffusion", type="generative",
        dataset=str(prompts_file), prompt_field="prompt",
        scoring={"scorer": "dpgbench"},
    )


def test_harness_dispatch_with_fake_scorer(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "a red cube",
                    "questions": [_q(1, "red?", [0]), _q(2, "cube?", [0])]}) + "\n"
    )
    fake_rows = [{
        "filename": "0.png", "idx": 0, "item_id": "i0", "prompt": "a red cube",
        "score": 0.5, "crop_scores": [0.5], "answers": {"1": "yes", "2": "no"},
        "raw_scores": {1: 1.0, 2: 0.0}, "adj_scores": {1: 1.0, 2: 0.0},
        "categories": {1: "entity - whole", 2: "entity - whole"},
    }]
    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: fake_rows)
    result = DiffusionHarness()._result_for(
        _dpg_recipe(prompts), "org/m", tmp_path / "out",
        [{"idx": "0", "prompt": "a red cube", "image_path": "0.png", "seed": "42"}],
        [{"prompt": "a red cube", "questions": []}],
    )
    assert result.metric == "dpgbench"
    assert result.score == 0.5
    assert result.num_samples == 1
    assert result.per_sample == [1.0, 0.0]  # adjusted per-question 0/1
    assert result.raw["categories_l1"] == {"entity": 0.5}
    assert result.raw["results"].endswith("dpgbench_results.jsonl")
    assert "dpgbench(" in result.command


def test_harness_dispatch_no_scorable_images_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [])
    with pytest.raises(AqaError, match="no generated images"):
        DiffusionHarness()._result_for(_dpg_recipe(tmp_path / "p.jsonl"), "org/m",
                                       tmp_path / "out", [], [])


def test_vendored_prompts_jsonl():
    from aqa.eval.dpgbench import DATA_DIR

    records = load_records(DATA_DIR / "prompts.jsonl")
    assert len(records) == 1065
    assert sum(len(r["questions"]) for r in records) == 14392
    assert records[0]["item_id"] == "partiprompts97"

"""DPG-Bench scorer: dense-prompt-following judged by mPLUG VQA.

Port of TencentQQGYLab/ELLA's ``dpg_bench/compute_dpg_bench.py`` (Apache-2.0 --
see NOTICE; data vendored under ``recipes/data/dpgbench/``).  Per image, each
proposition question is answered by VQA (``yes``/``no``); a question's score is
zeroed when any parent proposition (dependency) scored 0.  Aggregation follows
upstream exactly, including its quirks:

* the headline DPG score is the mean over images of the per-image
  DEPENDENCY-ADJUSTED mean question score;
* the per-category breakdown instead aggregates the RAW (pre-dependency)
  per-question scores (upstream returns ``qid2scores_orig`` for categories);
* the first CSV data row is skipped in data prep (upstream
  ``if i == 0: continue``), which leaves dangling dependencies on the first
  item -- upstream's per-image ``try/except`` then skips that image, and so
  does :meth:`DpgBenchScorer.evaluate_rows`.

Heavy deps (PIL / torch / modelscope) import lazily so the module is importable
without a GPU stack; the VQA judge is only constructed when scoring runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Protocol

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.dpgbench")

DEFAULT_VQA_MODEL = "damo/mplug_visual-question-answering_coco_large_en"
DEFAULT_RESOLUTION = 1024  # matches the harness's default generation size

DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "dpgbench"


# ── data prep: CSV -> prompts.jsonl ──────────────────────────────────────────
def convert_dpg_csv(csv_path: str | Path, out_path: str | Path) -> list[dict]:
    """Convert the official ``dpg_bench.csv`` into one-record-per-prompt JSONL.

    Records: ``{idx, item_id, prompt, questions: [{qid, question, answer,
    dependency, category}], category}`` -- ``idx`` aligns with the diffusion
    harness's flat generation layout (``<idx>.png``).  Port of the official
    ``prepare_dpg_data`` (including its first-data-row skip, for score parity).
    """
    items: dict[str, dict] = {}
    order: list[str] = []
    with Path(csv_path).open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i == 0:
                continue  # upstream quirk (kept): the first data row is dropped
            item_id = row["item_id"]
            if item_id not in items:
                items[item_id] = {"prompt": row["text"], "questions": []}
                order.append(item_id)
            items[item_id]["questions"].append({
                "qid": int(row["proposition_id"]),
                "question": row["question_natural_language"],
                "answer": row["tuple"],
                "dependency": [int(d.strip()) for d in row["dependency"].split(",")],
                # official category derivation (from the tuple column)
                "category": row["tuple"].split("(")[0].strip(),
            })
    records = []
    for idx, item_id in enumerate(order):
        questions = items[item_id]["questions"]
        records.append({
            "idx": idx,
            "item_id": item_id,
            "prompt": items[item_id]["prompt"],
            "questions": questions,
            "category": questions[0]["category"],
        })
    out_path = Path(out_path)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    return records


def load_records(path: str | Path) -> list[dict]:
    """Read a converted prompts.jsonl."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── VQA judge seam ────────────────────────────────────────────────────────────
class VQAJudge(Protocol):
    """Answers a yes/no question about an image (mPLUG in production, faked in tests)."""

    def vqa(self, image: Any, question: str) -> str: ...


class MPlugJudge:
    """mPLUG VQA via modelscope (``damo/mplug_visual-question-answering_*``).

    The modelscope pipeline (and its old pinned deps) is built lazily on first
    use; a construction failure surfaces as a clear :class:`AqaError`.
    """

    def __init__(self, model_id: str = DEFAULT_VQA_MODEL, device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _build(self) -> None:
        if self._pipe is not None:
            return
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as exc:
            raise AqaError(
                "dpgbench scoring needs the modelscope mPLUG VQA stack: "
                "pip install modelscope (plus mplug's pinned deps)"
            ) from exc
        device = self.device
        if device == "auto":
            try:
                import torch
                device = "gpu" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        try:
            self._pipe = pipeline(
                Tasks.visual_question_answering, model=self.model_id, device=device
            )
        except Exception as exc:
            raise AqaError(
                f"dpgbench: failed to load VQA model {self.model_id}: {exc}"
            ) from exc

    def vqa(self, image: Any, question: str) -> str:
        self._build()
        return self._pipe({"image": image, "question": question})["text"]


# ── scoring (compute_dpg_bench.py port) ──────────────────────────────────────
def score_image(
    judge: VQAJudge,
    image: Any,
    record: dict,
    *,
    resolution: int = DEFAULT_RESOLUTION,
    pic_num: int = 1,
) -> dict[str, Any]:
    """Score one generated image against its prompt record (official logic).

    ``pic_num=1`` keeps the top-left ``resolution``-sized crop only (upstream
    default); higher values add the remaining quadrants.  Returns the per-image
    average plus both score variants (raw/pre-dependency and adjusted).
    """
    crop_tuples = [
        (0, 0, resolution, resolution),
        (resolution, 0, resolution * 2, resolution),
        (0, resolution, resolution, resolution * 2),
        (resolution, resolution, resolution * 2, resolution * 2),
    ][:pic_num]
    questions = record["questions"]
    crop_scores: list[float] = []
    raw_scores: dict[int, float] = {}
    adj_scores: dict[int, float] = {}
    answers: dict[int, str] = {}
    for crop_tuple in crop_tuples:
        cropped = image.crop(crop_tuple)
        qid2scores: dict[int, float] = {}
        for q in questions:
            answer = judge.vqa(cropped, q["question"])
            answers[q["qid"]] = answer
            qid2scores[q["qid"]] = float(answer == "yes")
        raw_scores = dict(qid2scores)  # pre-dependency (upstream: qid2scores_orig)
        # zero out a question's score when any parent proposition answered 'no'
        for q in questions:
            parents = [p for p in q["dependency"] if p != 0]
            if any(qid2scores[parent] == 0 for parent in parents):
                qid2scores[q["qid"]] = 0.0
        adj_scores = qid2scores
        crop_scores.append(sum(qid2scores.values()) / len(qid2scores))
    return {
        "score": sum(crop_scores) / len(crop_scores),
        "crop_scores": crop_scores,
        "answers": answers,
        "raw_scores": raw_scores,
        "adj_scores": adj_scores,
    }


def _question_categories(record: dict) -> dict[int, str]:
    return {q["qid"]: q["category"] for q in record["questions"]}


def summarize(image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image rows into overall + L1/L2 category scores (official).

    Overall = mean of per-image (dependency-adjusted) scores; categories use the
    RAW per-question scores (upstream quirk, kept for parity).
    """
    overall = (sum(r["score"] for r in image_rows) / len(image_rows)) if image_rows else 0.0
    cat_scores: dict[str, list[float]] = {}
    for row in image_rows:
        for qid, cat in row["categories"].items():
            cat_scores.setdefault(cat, []).append(row["raw_scores"][qid])
    l2 = {cat: sum(v) / len(v) for cat, v in cat_scores.items()}
    l1_scores: dict[str, list[float]] = {}
    for cat, vals in cat_scores.items():
        l1_scores.setdefault(cat.split("-")[0].strip(), []).extend(vals)
    l1 = {cat: sum(v) / len(v) for cat, v in l1_scores.items()}
    return {"overall": overall, "categories": l2, "categories_l1": l1,
            "num_images": len(image_rows)}


# ── scorer (harness-facing) ───────────────────────────────────────────────────
class DpgBenchScorer:
    """Evaluate a flat generated-image dir against DPG-Bench prompt records."""

    def __init__(
        self,
        *,
        vqa_model: str = DEFAULT_VQA_MODEL,
        resolution: int = DEFAULT_RESOLUTION,
        device: str = "auto",
        pic_num: int = 1,
        judge: VQAJudge | None = None,  # test seam: inject a fake judge
    ):
        self.vqa_model = vqa_model
        self.resolution = resolution
        self.device = device
        self.pic_num = pic_num
        self._judge = judge

    def judge(self) -> VQAJudge:
        """The VQA judge, constructed only when scoring actually runs."""
        if self._judge is None:
            self._judge = MPlugJudge(self.vqa_model, device=self.device)
        return self._judge

    def evaluate_rows(
        self, rows: list[dict], records: list[dict], *, image_loader=None,
    ) -> list[dict[str, Any]]:
        """Score every generated image row; returns official-diffable rows.

        ``rows`` are the harness's metadata.csv rows (idx/image_path); ``records``
        are the converted prompt records (``questions`` per prompt), aligned by
        ``idx``.  A failed image is logged and skipped (upstream try/except parity).
        ``image_loader`` defaults to PIL's ``Image.open`` (lazy import).
        """
        if image_loader is None:
            from PIL import Image
            image_loader = Image.open

        judge = self.judge()
        by_idx = {int(r["idx"]): r for r in records}
        out: list[dict[str, Any]] = []
        for row in rows:
            idx = int(row["idx"])
            record = by_idx.get(idx)
            image_path = row["image_path"]
            if record is None:
                _log.warning("dpgbench: no prompt record for idx %d; skipped", idx)
                continue
            try:
                result = score_image(
                    judge, image_loader(image_path), record,
                    resolution=self.resolution, pic_num=self.pic_num,
                )
            except Exception as exc:  # noqa: BLE001 - upstream: 'Failed filename'
                _log.warning("dpgbench: failed to score %s: %s", image_path, exc)
                continue
            out.append({
                "filename": str(image_path),
                "idx": idx,
                "item_id": record.get("item_id", ""),
                "prompt": record["prompt"],
                "score": result["score"],
                "crop_scores": result["crop_scores"],
                "answers": {str(k): v for k, v in result["answers"].items()},
                "raw_scores": result["raw_scores"],
                "adj_scores": result["adj_scores"],
                "categories": _question_categories(record),
            })
        return out

    def write_results(self, rows: list[dict[str, Any]], out_prefix: Path) -> tuple[Path, Path]:
        """Write results.jsonl + the official-format results txt lines."""
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        jsonl = out_prefix.with_suffix(".jsonl")
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        txt = out_prefix.with_suffix(".txt")
        # official line: "<image_path>, <crop score>..., <average score>"
        txt.write_text("".join(
            f"{r['filename']}, " + ", ".join(str(s) for s in r["crop_scores"])
            + f", {r['score']}\n"
            for r in rows
        ))
        return jsonl, txt

"""DPG-Bench scorer: dense-prompt-following judged by mPLUG VQA.

Port of TencentQQGYLab/ELLA's ``dpg_bench/compute_dpg_bench.py`` (Apache-2.0 --
see NOTICE; data vendored under ``recipes/data/dpgbench/``).  Per image, each
proposition question is answered by VQA (``yes``/``no``); a question's score is
zeroed when any parent proposition (dependency) scored 0.  Aggregation follows
upstream exactly, including its quirks:

* the headline DPG score is the mean over images of the per-image
  DEPENDENCY-ADJUSTED mean question score;
* the per-category breakdown instead aggregates the RAW (pre-dependency)
  per-question scores (upstream returns ``qid2scores_orig`` for categories).

Deliberate deviations from upstream (verified against ELLA's
``dpg_bench/compute_dpg_bench.py``):

* ``prepare_dpg_data`` really does skip the first CSV data row
  (``for i, line in data.iterrows(): if i == 0: continue``).  On the vendored
  ``dpg_bench.csv`` that row is a genuine proposition -- every one of the 1,065
  items starts at ``proposition_id`` 1 -- so upstream drops question 1 of the
  first item and leaves 8 dependency edges dangling on it; its per-image
  ``try/except`` then skips that image forever (``KeyError(1)``).  We do **not**
  skip the row, so the vendored data has zero dangling dependencies (14,392
  questions) and we score one more image than upstream out of 1,065.
* Consequently :func:`score_image` ignores a dependency on a question the record
  does not carry: a parent that is absent cannot be shown to have answered
  ``no``, so it must not zero a score (upstream raises ``KeyError`` instead and
  loses the whole image).

Everything else follows upstream exactly, including its quirks:

* the headline DPG score is the mean over images of the per-image
  DEPENDENCY-ADJUSTED mean question score;
* the per-category breakdown instead aggregates the RAW (pre-dependency)
  per-question scores (upstream reports ``qid2scores_orig`` for categories).

Heavy deps (PIL / torch / modelscope) import lazily so the module is importable
without a GPU stack; the VQA judge is only constructed when scoring runs.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.dpgbench")

DEFAULT_VQA_MODEL = "damo/mplug_visual-question-answering_coco_large_en"
DEFAULT_RESOLUTION = 1024  # matches the harness's default generation size
# vLLM-served judge (the mPLUG substitution; see the recipe description).
DEFAULT_VLLM_JUDGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"

DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "dpgbench"


# ── data prep: CSV -> prompts.jsonl ──────────────────────────────────────────
def convert_dpg_csv(csv_path: str | Path, out_path: str | Path) -> list[dict]:
    """Convert the official ``dpg_bench.csv`` into one-record-per-prompt JSONL.

    Records: ``{idx, item_id, prompt, questions: [{qid, question, answer,
    dependency, category}], category}`` -- ``idx`` aligns with the diffusion
    harness's flat generation layout (``<idx>.png``).  Port of the official
    ``prepare_dpg_data`` **without** its first-data-row skip: on this CSV that
    row is a real proposition (see the module docstring).
    """
    items: dict[str, dict] = {}
    order: list[str] = []
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
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


# ── vLLM-served VQA judge (mPLUG substitution) ──────────────────────────────
_VQA_SYSTEM_PROMPT = (
    "You are a visual question answering judge for the DPG-Bench benchmark. "
    "Answer the question about the image with a single word: 'yes' or 'no'."
)
_VQA_USER_TEMPLATE = (
    "Question: {question}\n"
    "Answer with exactly one word, either 'yes' or 'no', and nothing else."
)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def build_vqa_messages(question: str, image_base64: str) -> list[dict]:
    """Multimodal chat messages for one DPG-Bench proposition (mPLUG substitution).

    Uses the same question text the official ``compute_dpg_bench.py`` sends to
    mPLUG; the image is attached as a data URI so the judge can answer yes/no.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": _VQA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": _VQA_USER_TEMPLATE.format(question=question)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]},
    ]


def extract_yes_no(text: str) -> str:
    """Normalize a judge reply to ``'yes'``/``'no'`` (else the cleaned raw text).

    The scorer compares answers with ``== "yes"``, so anything that is not an
    explicit yes/no folds to a non-matching string and scores 0 -- matching how
    mPLUG's free-text answer is compared.
    """
    cleaned = re.sub(r" thinking.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    match = _YES_NO_RE.search(cleaned)
    return match.group(1).lower() if match else cleaned


def encode_image(image: Any) -> str:
    """Base64-encode a PIL image (or an image path) as PNG."""
    if isinstance(image, (str, Path)):
        return base64.b64encode(Path(image).read_bytes()).decode()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VllmVQAJudge:
    """DPG-Bench VQA judge over a vLLM OpenAI-compatible chat endpoint (stdlib only).

    Substitutes for the official mPLUG/modelscope judge, whose dependency chain
    (fairseq et al.) is uninstallable on Python 3.12 -- see the recipe
    description.  No judge weights are loaded in-process: a vLLM server must be
    running (``vllm serve <judge_model>``).  Mirrors
    :class:`aqa.eval.wise.VllmWiseJudge`: ``temperature=0.0``, a small
    ``max_tokens``, ``chat_template_kwargs = {"enable_thinking": False}``, and a
    clear :class:`AqaError` on any request failure.

    ``enable_thinking: False`` is load-bearing for a reasoning judge: with
    thinking on, Qwen3.5 answers a yes/no question with its reasoning trace and
    then a bare ``1`` ("The user wants to know if there is a book in the
    image.\\n\\n1" was the observed reply), which can never normalize to
    ``"yes"`` -- every proposition would score 0 as an artifact of the judge
    setup rather than of the image.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_VLLM_JUDGE_MODEL,
        base_url: str = DEFAULT_JUDGE_BASE_URL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 300.0,
        max_tokens: int = 16,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def vqa(self, image: Any, question: str) -> str:  # pragma: no cover - network
        payload = json.dumps({
            "model": self.model,
            "messages": build_vqa_messages(question, encode_image(image)),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            # Raw HTTP, so the kwargs stay top-level (as in vllm_eval.py).
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:
            raise AqaError(
                f"dpgbench: VQA judge request to {self.base_url} failed: {exc} "
                f"(is `vllm serve {self.model}` running and reachable?)"
            ) from exc
        return extract_yes_no(body["choices"][0]["message"]["content"])


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
        # zero out a question's score when any parent proposition answered 'no';
        # a parent the record does not carry cannot be shown to have answered
        # 'no', so it is ignored rather than raising (which would skip the image)
        for q in questions:
            parents = [p for p in q["dependency"] if p != 0 and p in qid2scores]
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


def summarize(
    image_rows: list[dict[str, Any]], *, num_skipped: int = 0, num_env_failed: int = 0
) -> dict[str, Any]:
    """Aggregate per-image rows into overall + L1/L2 category scores (official).

    Overall = mean of per-image (dependency-adjusted) scores; categories use the
    RAW per-question scores (upstream quirk, kept for parity).

    ``num_skipped`` counts generated images that produced no row (upstream
    try/except parity); ``num_env_failed`` is the subset caused by an
    unavailable VQA judge.  Both are carried through as ``num_missing`` /
    ``num_env_failed`` with ``complete`` for the written results.
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
    num_missing = num_skipped + num_env_failed
    return {"overall": overall, "categories": l2, "categories_l1": l1,
            "num_images": len(image_rows), "num_scored": len(image_rows),
            "num_missing": num_missing, "num_env_failed": num_env_failed,
            "complete": num_missing == 0}


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
        judge_kind: str = "mplug",  # "mplug" (official) | "vllm" (substitution)
        judge_model: str = DEFAULT_VLLM_JUDGE_MODEL,
        base_url: str = DEFAULT_JUDGE_BASE_URL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 300.0,
        max_tokens: int = 16,
        judge: VQAJudge | None = None,  # test seam: inject a fake judge
    ):
        if judge_kind not in ("mplug", "vllm"):
            raise AqaError(f"dpgbench: unknown judge {judge_kind!r}; choose 'mplug' or 'vllm'")
        self.vqa_model = vqa_model
        self.resolution = resolution
        self.device = device
        self.pic_num = pic_num
        self.judge_kind = judge_kind
        self.judge_model = judge_model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._judge = judge
        # Counts set by evaluate_rows (written into the results for the harness).
        self.num_skipped = 0
        self.num_env_failed = 0

    def judge(self) -> VQAJudge:
        """The VQA judge, constructed only when scoring actually runs."""
        if self._judge is None:
            if self.judge_kind == "vllm":
                self._judge = VllmVQAJudge(
                    model=self.judge_model, base_url=self.base_url,
                    api_key_env=self.api_key_env, timeout=self.timeout,
                    max_tokens=self.max_tokens,
                )
            else:
                self._judge = MPlugJudge(self.vqa_model, device=self.device)
        return self._judge

    def judge_label(self) -> str:
        """Human-readable identifier of the active judge (for the command string)."""
        if self.judge_kind == "vllm":
            return f"vllm:{self.judge_model}@{self.base_url}"
        return f"mplug:{self.vqa_model}"

    def evaluate_rows(
        self, rows: list[dict], records: list[dict], *, image_loader=None,
    ) -> list[dict[str, Any]]:
        """Score every generated image row; returns official-diffable rows.

        ``rows`` are the harness's metadata.csv rows (idx/image_path); ``records``
        are the converted prompt records (``questions`` per prompt), aligned by
        ``idx``.  A failed image is logged and skipped (upstream try/except parity).

        An :class:`AqaError` from the judge means the judge/model is unavailable:
        it is counted in :attr:`num_env_failed` (so the harness can fail the run
        instead of reporting zeros), while other failures (e.g. the upstream
        dangling-dependency ``KeyError``) are ordinary skips in
        :attr:`num_skipped`.  ``image_loader`` defaults to PIL's ``Image.open``.
        """
        if image_loader is None:
            from PIL import Image
            image_loader = Image.open

        judge = self.judge()
        by_idx = {int(r["idx"]): r for r in records}
        out: list[dict[str, Any]] = []
        self.num_skipped = 0
        self.num_env_failed = 0
        for row in rows:
            idx = int(row["idx"])
            record = by_idx.get(idx)
            image_path = row["image_path"]
            if record is None:
                _log.warning("dpgbench: no prompt record for idx %d; skipped", idx)
                self.num_skipped += 1
                continue
            try:
                result = score_image(
                    judge, image_loader(image_path), record,
                    resolution=self.resolution, pic_num=self.pic_num,
                )
            except AqaError as exc:
                _log.warning("dpgbench: judge unavailable for %s: %s", image_path, exc)
                self.num_env_failed += 1
                continue
            except Exception as exc:  # noqa: BLE001 - upstream: 'Failed filename'
                _log.warning("dpgbench: failed to score %s: %s", image_path, exc)
                self.num_skipped += 1
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
        text = "".join(
            f"{r['filename']}, " + ", ".join(str(s) for s in r["crop_scores"])
            + f", {r['score']}\n"
            for r in rows
        )
        num_missing = self.num_skipped + self.num_env_failed
        if num_missing:
            text += (
                f"missing: {num_missing} "
                f"(environmental/unavailable-judge: {self.num_env_failed})\n"
            )
        txt.write_text(text)
        return jsonl, txt

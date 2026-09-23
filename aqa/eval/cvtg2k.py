"""CVTG-2K scorer: complex visual text generation (Word Accuracy / NED / CLIPScore).

Port of the TextCrafter project's ``TextCrafter_Eval/unified_metrics_eval.py``
(NJU-PCALab/TextCrafter, arXiv:2503.23461 -- see NOTICE; prompts vendored under
``recipes/data/cvtg2k/``).  CVTG-2K is a 2,000-prompt English benchmark (the
"C" stands for **Complex**, not Chinese -- a common misreading): a plain
``CVTG`` subset (~1,002 prompts) plus a ``CVTG-Style`` subset (~998 prompts
carrying size/color/font attributes), each split by the number of text regions
(2-5).  The task's scoring rules are:

* **Word Accuracy** -- OCR-detected words exactly matching the ground-truth
  words, where ground truth is extracted from the single-quoted strings in the
  prompt (``extract_words_from_prompt``) and OCR is PaddleOCR with
  ``lang="en"``.  Aggregation is word-count weighted (``sum(correct) /
  sum(total)``), exactly as the reference ``evaluate_full_dataset`` does.
* **NED** -- per ground-truth word, ``1 - levenshtein(gt, best_pred) /
  max(len(gt), len(best_pred)) + 1e-5`` where ``best_pred`` is the OCR word
  ``difflib.get_close_matches`` ranks closest (cutoff 0); overall NED is the
  mean over **all** ground-truth words (not a per-image mean).
* **CLIPScore** -- official OpenAI CLIP ``ViT-L/14`` (``clip.load``), text
  prefixed with ``"A photo depicts "``; embeddings are L2-normalized and scored
  by cosine similarity, reported on the 0-1 scale.

Deliberate substitutions / faithfulness notes
---------------------------------------------
* CLIPScore scale: the TextCrafter reference multiplies the clipped cosine by
  ``2.5`` (the CLIPScore-paper constant), and **that is what we report by
  default** -- the third-party HunyuanImage-3.0 anchor 0.8121 is on the 2.5x
  scale, so matching it is what makes our number comparable.  Set
  ``clipscore_x2_5=False`` to get the bare 0-1 cosine instead.  (Note: the
  repo's own ``image_metrics`` CLIP scorer uses a different convention, x100.)
* Levenshtein: the reference imports ``Levenshtein`` (python-Levenshtein).  We
  prefer that package (and ``rapidfuzz``) when importable, but fall back to an
  equivalent pure-Python dynamic-programming distance so the scorer -- and its
  tests -- run without the extra dependency.  The distance is identical.
* PaddleOCR: the reference calls ``PaddleOCR(lang="en").ocr(..., cls=True)``
  (2.x API); PaddleOCR 3.x dropped that call, so the engine is built and called
  through :mod:`aqa.eval._paddleocr`, which speaks both API generations.
  PaddleOCR itself is imported lazily, so importing this module never needs it.

Heavy deps (PaddleOCR / torch / openai-clip / PIL) import lazily; the OCR and
CLIP components are injectable so tests stay GPU- and network-free.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from ..logging_util import AqaError, get_logger
from ._paddleocr import PaddleOCREngine

_log = get_logger("eval.cvtg2k")

DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "cvtg2k"

SUBSETS = ("CVTG", "CVTG-Style")
AREAS = (2, 3, 4, 5)
CLIP_PREFIX = "A photo depicts "
DEFAULT_CLIP_MODEL = "ViT-L/14"

_QUOTE_RE = re.compile(r"'(.*?)'")


# ── data prep: official dirs -> prompts.jsonl ────────────────────────────────
def extract_words_from_prompt(prompt: str) -> list[str]:
    """Ground-truth words: single-quoted substrings, lowercased and split.

    Port of the reference ``extract_words_from_prompt`` (``re.findall(r"'(.*?)'",
    prompt)`` then ``match.lower().split()``).  Punctuation inside a quoted
    string stays attached; the same tokenization is applied to OCR output.
    """
    words: list[str] = []
    for match in _QUOTE_RE.findall(prompt):
        words.extend(match.lower().split())
    return words


def convert_cvtg_dir(source_dir: str | Path, out_path: str | Path) -> list[dict]:
    """Convert an extracted ``CVTG-2K/`` tree into one-record-per-prompt JSONL.

    Records: ``{idx, subset, area, index, prompt, gt_words}`` in the reference's
    evaluation order (``CVTG`` then ``CVTG-Style``, areas 2..5, ``data_list``
    order within each file).  ``idx`` aligns with the diffusion harness's flat
    generation layout (``<idx>.png``).
    """
    source_dir = Path(source_dir)
    records: list[dict] = []
    for subset in SUBSETS:
        for area in AREAS:
            path = source_dir / subset / f"{area}.json"
            if not path.exists():
                raise AqaError(f"cvtg2k: missing official prompt file {path}")
            data = json.loads(path.read_text())
            for item in data.get("data_list", []):
                prompt = item["prompt"]
                records.append({
                    "idx": len(records),
                    "subset": subset,
                    "area": area,
                    "index": item["index"],
                    "prompt": prompt,
                    "gt_words": extract_words_from_prompt(prompt),
                })
    out_path = Path(out_path)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    return records


def load_records(path: str | Path) -> list[dict]:
    """Read a converted prompts.jsonl."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── metric primitives (reference ports) ──────────────────────────────────────
def _levenshtein(a: str, b: str) -> int:
    """Edit distance; prefers python-Levenshtein / rapidfuzz, else pure Python DP."""
    try:
        import Levenshtein  # type: ignore

        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    try:
        from rapidfuzz.distance import Levenshtein as _RF  # type: ignore

        return int(_RF.distance(a, b))
    except ImportError:
        pass
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(gt: str, pred: str) -> float:
    """``1 - levenshtein(gt, pred) / (max(len(gt), len(pred)) + 1e-5)`` (reference)."""
    return 1 - _levenshtein(gt, pred) / (max(len(gt), len(pred)) + 1e-5)


def word_scores(gt_words: list[str], pred_words: list[str]) -> dict[str, Any]:
    """Per-image OCR metrics against the reference algorithm.

    ``correct`` counts ground-truth words present verbatim in ``pred_words``;
    ``ned`` is one normalized edit distance per ground-truth word, comparing
    against the ``difflib`` closest OCR word.  An empty OCR result is treated as
    a single ``''`` prediction (reference behavior).
    """
    preds = list(pred_words) or [""]
    correct = sum(1 for gt in gt_words if gt in preds)
    neds = []
    for gt in gt_words:
        best = difflib.get_close_matches(gt, preds, n=1, cutoff=0)
        neds.append(normalized_edit_distance(gt, best[0]) if best else 0.0)
    return {"total_words": len(gt_words), "correct_words": correct, "word_neds": neds}


def clipscore_from_features(image_features, text_features, *, x2_5: bool = True) -> float:
    """CLIP cosine score from a single image/text feature pair.

    Reproduces the reference normalization (``/ sqrt(sum(x**2))``, the numpy
    >= 1.21 branch) and similarity, then applies the CLIPScore-paper
    ``2.5 * max(cos, 0)`` constant by default (``x2_5``) -- the scale the
    reference and the published anchors use.  Pass ``x2_5=False`` for the bare
    0-1 cosine.
    """
    import numpy as np

    img = np.asarray(image_features, dtype=np.float64).reshape(-1)
    txt = np.asarray(text_features, dtype=np.float64).reshape(-1)
    img = img / np.sqrt(np.sum(img**2))
    txt = txt / np.sqrt(np.sum(txt**2))
    similarity = float(np.sum(img * txt))
    clipped = max(similarity, 0.0)
    return 2.5 * clipped if x2_5 else clipped


# ── OCR / CLIP seams ─────────────────────────────────────────────────────────
class OCR(Protocol):
    """Recognizes the text lines in an image (PaddleOCR in production, faked in tests)."""

    def recognize(self, image: Any) -> list[str]: ...


class ClipScorer(Protocol):
    """Image-text CLIP score (official OpenAI CLIP in production, faked in tests)."""

    def score(self, image_path: Any, text: str) -> float: ...


class OpenAIClipScorer:
    """Official ``ViT-L/14`` via ``clip.load`` (``openai-clip``/``clip-anytorch``).

    The text prompt is prefixed with ``"A photo depicts "`` and the score is the
    normalized image/text cosine (optionally x2.5).  ``clip`` and ``torch`` are
    imported lazily.
    """

    def __init__(
        self, model_name: str = DEFAULT_CLIP_MODEL, device: str = "auto",
        *, x2_5: bool = True, prefix: str = CLIP_PREFIX,
    ):
        self.model_name = model_name
        self.device = device
        self.x2_5 = x2_5
        self.prefix = prefix
        self._model = None
        self._preprocess = None

    def _build(self) -> None:
        if self._model is not None:
            return
        try:
            import clip
            import torch
        except ImportError as exc:
            raise AqaError(
                "cvtg2k CLIPScore needs the official OpenAI CLIP: "
                "pip install openai-clip (or clip-anytorch)"
            ) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model, preprocess = clip.load(self.model_name, device=device, jit=False)
        except Exception as exc:
            raise AqaError(f"cvtg2k: failed to load CLIP {self.model_name}: {exc}") from exc
        model.eval()
        self._model, self._preprocess, self.device = model, preprocess, device

    def score(self, image_path: Any, text: str) -> float:
        self._build()
        import clip
        import torch
        from PIL import Image

        with torch.no_grad():
            image_input = self._preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([self.prefix + text], truncate=True).to(self.device)
            image_features = self._model.encode_image(image_input).cpu().numpy()
            text_features = self._model.encode_text(text_input).cpu().numpy()
        return clipscore_from_features(image_features[0], text_features[0], x2_5=self.x2_5)


# ── per-image scoring ────────────────────────────────────────────────────────
def ocr_words(ocr: OCR, image: Any) -> list[str]:
    """The OCR lines, lowercased and split into words (reference tokenization)."""
    words: list[str] = []
    for line in ocr.recognize(image):
        words.extend(str(line).lower().split())
    return words


# ── aggregation (reference ``evaluate_full_dataset`` port) ───────────────────
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Word-Accuracy (word-weighted), NED (mean over all words) + per-area breakdown.

    Overall CLIPScore is the image-count-weighted mean of the per-area means,
    which equals the plain mean over images (reference ``evaluate_full_dataset``).

    Rows whose scoring infrastructure was unavailable (``scored is False``) are
    excluded from every metric -- reporting them as zeros would corrupt the
    number -- and counted in ``num_missing``/``num_env_failed`` with
    ``complete`` set accordingly.  A row missing the key (e.g. hand-built test
    data) is treated as scored.
    """
    scored_rows = [r for r in rows if r.get("scored", True)]
    total_words = sum(r["total_words"] for r in scored_rows)
    correct_words = sum(r["correct_words"] for r in scored_rows)
    all_neds = [n for r in scored_rows for n in r["word_neds"]]
    clips = [r["clip_score"] for r in scored_rows]
    area_results: dict[str, dict[str, Any]] = {}
    for key in sorted({f"{r['subset']}/{r['area']}" for r in scored_rows}):
        subset, area = key.split("/")
        group = [r for r in scored_rows if r["subset"] == subset and str(r["area"]) == area]
        g_words = sum(r["total_words"] for r in group)
        g_neds = [n for r in group for n in r["word_neds"]]
        area_results[key] = {
            "word_accuracy": (sum(r["correct_words"] for r in group) / g_words) if g_words else 0.0,
            "ned": (sum(g_neds) / len(g_neds)) if g_neds else 0.0,
            "clipscore": (sum(r["clip_score"] for r in group) / len(group)) if group else 0.0,
            "num_images": len(group),
        }
    num_missing = len(rows) - len(scored_rows)
    return {
        "overall": {
            "word_accuracy": (correct_words / total_words) if total_words else 0.0,
            "ned": (sum(all_neds) / len(all_neds)) if all_neds else 0.0,
            "clipscore": (sum(clips) / len(clips)) if clips else 0.0,
            "total_words": total_words,
            "correct_words": correct_words,
        },
        "area_results": area_results,
        "num_images": len(rows),
        "num_scored": len(scored_rows),
        "num_missing": num_missing,
        # every unscored row here is an unavailable OCR/CLIP engine, not a miss
        "num_env_failed": num_missing,
        "complete": num_missing == 0,
    }


# ── scorer (harness-facing) ──────────────────────────────────────────────────
class Cvtg2kScorer:
    """Evaluate a flat generated-image dir against CVTG-2K prompt records."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        ocr_lang: str = "en",
        clip_model: str = DEFAULT_CLIP_MODEL,
        clipscore_x2_5: bool = True,
        device: str = "auto",
        ocr: OCR | None = None,          # test seam: inject a fake OCR
        clip: ClipScorer | None = None,  # test seam: inject a fake CLIP
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.ocr_lang = ocr_lang
        self.clip_model = clip_model
        self.clipscore_x2_5 = clipscore_x2_5
        self.device = device
        self._ocr = ocr
        self._clip = clip

    def ocr_engine(self) -> OCR:
        if self._ocr is None:
            self._ocr = PaddleOCREngine("cvtg2k", self.ocr_lang, device=self.device)
        return self._ocr

    def clip_scorer(self) -> ClipScorer:
        if self._clip is None:
            self._clip = OpenAIClipScorer(
                self.clip_model, device=self.device, x2_5=self.clipscore_x2_5
            )
        return self._clip

    def score_image(self, image: Any, record: dict, image_path: Any = None) -> dict[str, Any]:
        """Score one image against its record; OCR/CLIP failures score zero (reference).

        A failure to build the OCR/CLIP engine raises :class:`AqaError` and is
        recorded as ``scored=False`` (an environmental failure: the component is
        simply unavailable) so the harness can refuse to report a metric that
        would be an environment artifact.  Any other per-image exception keeps the
        reference's zero-score behavior and counts as a scored image.
        """
        ocr = self.ocr_engine()
        clip = self.clip_scorer()
        gt_words = extract_words_from_prompt(record["prompt"])
        ocr_scored = True
        try:
            pred_words = ocr_words(ocr, image)
        except AqaError as exc:
            _log.warning("cvtg2k: OCR unavailable for %s: %s", image_path, exc)
            pred_words = []
            ocr_scored = False
        except Exception as exc:  # noqa: BLE001 - reference: OCR failure -> 0 words correct
            _log.warning("cvtg2k: OCR failed for %s: %s", image_path, exc)
            pred_words = []
        scores = word_scores(gt_words, pred_words)
        clip_scored = True
        try:
            clip_score = float(clip.score(
                image_path if image_path is not None else image, record["prompt"]
            ))
        except AqaError as exc:
            _log.warning("cvtg2k: CLIP unavailable for %s: %s", image_path, exc)
            clip_score = 0.0
            clip_scored = False
        except Exception as exc:  # noqa: BLE001 - reference: CLIP failure -> 0.0
            _log.warning("cvtg2k: CLIP failed for %s: %s", image_path, exc)
            clip_score = 0.0
        return {
            "prompt": record["prompt"],
            "subset": record["subset"],
            "area": record["area"],
            "index": record.get("index"),
            "gt_words": gt_words,
            "pred_words": pred_words,
            "total_words": scores["total_words"],
            "correct_words": scores["correct_words"],
            "word_neds": scores["word_neds"],
            "clip_score": clip_score,
            "ocr_scored": ocr_scored,
            "clip_scored": clip_scored,
            "scored": ocr_scored and clip_scored,
        }

    def evaluate_rows(
        self, rows: list[dict], records: list[dict], *, image_loader=None,
    ) -> list[dict[str, Any]]:
        """Score every generated image row (``idx`` aligns with ``records``).

        ``rows`` are the harness metadata rows (idx/image_path); ``records`` the
        converted CVTG-2K prompt records.  ``image_loader`` defaults to PIL's
        ``Image.open`` (lazy import).
        """
        if image_loader is None:
            from PIL import Image
            image_loader = Image.open
        by_idx = {int(r["idx"]): r for r in records}
        out: list[dict[str, Any]] = []
        for row in rows:
            idx = int(row["idx"])
            record = by_idx.get(idx)
            if record is None:
                _log.warning("cvtg2k: no prompt record for idx %d; skipped", idx)
                continue
            image_path = row["image_path"]
            image = image_loader(image_path)
            result = self.score_image(image, record, image_path=image_path)
            out.append({"filename": str(image_path), "idx": idx, "prompt_id": idx, **result})
        return out

    def write_results(self, rows: list[dict[str, Any]], out_prefix: Path) -> tuple[Path, Path]:
        """Write results.jsonl plus a human-readable per-area summary txt."""
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        jsonl = out_prefix.with_suffix(".jsonl")
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary = summarize(rows)
        txt = out_prefix.with_suffix(".txt")
        lines = [
            f"{key}: word_accuracy={m['word_accuracy']:.4f} ned={m['ned']:.4f} "
            f"clipscore={m['clipscore']:.4f} images={m['num_images']}"
            for key, m in summary["area_results"].items()
        ]
        overall = summary["overall"]
        lines.append(
            f"overall: word_accuracy={overall['word_accuracy']:.4f} ned={overall['ned']:.4f} "
            f"clipscore={overall['clipscore']:.4f} images={summary['num_images']}"
        )
        if summary["num_missing"]:
            lines.append(
                f"missing: {summary['num_missing']} "
                f"(environmental/unavailable-scorer: {summary['num_env_failed']})"
            )
        txt.write_text("\n".join(lines) + "\n")
        return jsonl, txt

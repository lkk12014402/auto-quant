"""GlyphDraw2 scorer: text-rendering accuracy (Acc / NED) judged by PP-OCR.

GlyphDraw2 (OPPO-Mente-Lab, MIT -- see NOTICE) releases prompts and OCR
recognizer weights but **no runnable Acc/NED scorer**.  This module therefore
adopts the scoring semantics of the public AnyText evaluator
(``tyxsspa/AnyText``, ``eval/eval_dgocr.py``, Apache-2.0 -- see NOTICE): per
ground-truth text line,

* ``Acc`` -- 1 iff the recognized text equals the ground-truth string
  (AnyText's ``sen_acc``: fraction of correctly rendered lines), and
* ``NED`` -- ``1 - levenshtein(pred, gt) / (max(len(pred), len(gt)) + 1e-5)``
  (AnyText's ``get_ld``, character-id sequences).

The metric definition is borrowed from AnyText **because GlyphDraw2 ships no
scorer**; it is not byte-parity with an unpublished implementation.

Deliberate substitutions / faithfulness notes
---------------------------------------------
* Region crops: AnyText crops each ground-truth text region (it has region
  boxes) and recognizes the crop.  GlyphDraw2 prompts carry no boxes, so we run
  PP-OCR over the **whole image** and, per ground-truth line, take the OCR line
  with the smallest edit distance as the prediction (``Acc`` then tests exact
  membership).  This is the standard region-free adaptation, not the official
  crop-based protocol.
* OCR engine: GlyphDraw2 ships ``third_party/ocr_weights/ppv3_rec*.pth`` (a
  recognition-only model) plus char dicts but no detector, so the released
  weights cannot produce whole-image predictions.  We use PaddleOCR's
  PP-OCR detection+recognition pipeline as the stand-in (``lang="en"`` for the
  English files, ``lang="ch"`` for the Chinese ones) -- family-compatible with
  the released PP-OCRv3 recognizer, but not those exact weights.
* Char dict: NED is computed on raw strings by default; pass ``char2id`` (e.g.
  from GlyphDraw2's ``ppocr_keys_v1.txt``) to reproduce AnyText's out-of-vocab
  collapsing.  Without it the two agree whenever all characters are in-vocab.
* Levenshtein: python-Levenshtein / rapidfuzz are preferred when importable,
  with an equivalent pure-Python fallback so tests need no extra dependency.

Heavy deps (PaddleOCR / PIL) import lazily; OCR is injectable so tests stay
GPU- and network-free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from ..logging_util import AqaError, get_logger
from ._paddleocr import PaddleOCREngine

_log = get_logger("eval.glyphdraw2")

DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "glyphdraw2"

# (file stem, language, kind) in the order used for the flat, idx-aligned layout.
PROMPT_FILES = (
    ("complex_en", "en", "complex"),
    ("complex_zh", "zh", "complex"),
    ("poster_en", "en", "poster"),
    ("poster_zh", "zh", "poster"),
)

# GlyphDraw2 prompts quote their target strings with curly quotes (“ ”); accept
# straight quotes too so hand-written records still parse.
_QUOTE_RE = re.compile(r"[“\"](.*?)[”\"]")


def extract_gt_texts(prompt: str) -> list[str]:
    """Ground-truth text lines: quoted substrings of the GlyphDraw2 prompt."""
    return [m for m in _QUOTE_RE.findall(prompt)]


def convert_glyphdraw_prompts(source_dir: str | Path, out_path: str | Path) -> list[dict]:
    """Convert the repo's ``prompts/*.txt`` into one-record-per-prompt JSONL.

    Records: ``{idx, kind, lang, prompt, gt_texts}`` in ``PROMPT_FILES`` order
    (complex_en, complex_zh, poster_en, poster_zh).  ``idx`` aligns with the
    diffusion harness's flat generation layout (``<idx>.png``).
    """
    source_dir = Path(source_dir)
    records: list[dict] = []
    for stem, lang, kind in PROMPT_FILES:
        path = source_dir / f"{stem}.txt"
        if not path.exists():
            raise AqaError(f"glyphdraw2: missing prompt file {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if not prompt:
                continue
            records.append({
                "idx": len(records),
                "kind": kind,
                "lang": lang,
                "prompt": prompt,
                "gt_texts": extract_gt_texts(prompt),
            })
    out_path = Path(out_path)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    return records


def load_records(path: str | Path) -> list[dict]:
    """Read a converted prompts.jsonl."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_char_ids(dict_path: str | Path) -> dict[str, int]:
    """Build a compact ``char -> id`` map from a PP-OCR char dict (one char per line).

    Blank lines are dropped before numbering so ids stay dense; the OOV sentinel
    used by :func:`normalized_edit_distance` is ``len(char2id)`` and must not
    collide with a real character's id.
    """
    chars = [ch for ch in Path(dict_path).read_text(encoding="utf-8").splitlines() if ch]
    return {ch: i for i, ch in enumerate(chars)}


# ── metric primitives (AnyText ``eval_dgocr.py`` port) ───────────────────────
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


def normalized_edit_distance(gt: str, pred: str, *, char2id: dict[str, int] | None = None) -> float:
    """``1 - levenshtein(pred, gt) / (max(len(pred), len(gt)) + 1e-5)`` (AnyText).

    With ``char2id`` both strings are first mapped to character-id sequences
    (unknown characters collapse to the same out-of-vocab id), matching
    AnyText's ``get_ld``; without it the raw strings are compared.
    """
    if char2id is not None:
        oov = len(char2id)
        gt_ids = [char2id.get(c, oov) for c in gt]
        pred_ids = [char2id.get(c, oov) for c in pred]
        a, b = pred_ids, gt_ids
    else:
        a, b = pred, gt
    return 1 - _levenshtein(a, b) / (max(len(a), len(b)) + 1e-5)


def line_scores(
    gt_texts: list[str], pred_texts: list[str], *, char2id: dict[str, int] | None = None
) -> dict[str, Any]:
    """Per-prompt Accuracy + NED over ground-truth lines (AnyText semantics).

    Each ground-truth line is matched to the closest OCR line (minimum edit
    distance); ``correct`` is 1 iff some OCR line equals it exactly after
    stripping surrounding whitespace.  Returns per-line values so aggregation can
    mirror AnyText's flat mean over lines.
    """
    preds = [p.strip() for p in pred_texts if p and p.strip()]
    line_acc: list[float] = []
    line_neds: list[float] = []
    matched: list[str] = []
    for gt in gt_texts:
        gt = gt.strip()
        best = min(preds, key=lambda p: _levenshtein(gt, p)) if preds else ""
        matched.append(best)
        line_acc.append(1.0 if gt in preds else 0.0)
        line_neds.append(normalized_edit_distance(gt, best, char2id=char2id))
    return {"num_lines": len(gt_texts), "line_acc": line_acc, "line_neds": line_neds,
            "matched": matched}


# ── OCR seam ─────────────────────────────────────────────────────────────────
class OCR(Protocol):
    """Recognizes the text lines in an image (PaddleOCR in production, faked in tests)."""

    def recognize(self, image: Any, lang: str = "en") -> list[str]: ...


# ── aggregation ──────────────────────────────────────────────────────────────
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Flat mean over ground-truth lines for Acc and NED, plus per-kind/lang splits.

    Rows whose OCR engine was unavailable (``scored is False``) are excluded from
    the metrics -- not counted as zeros -- and reported via ``num_missing`` /
    ``num_env_failed`` with ``complete`` set accordingly.  A row missing the key
    is treated as scored.
    """
    def _means(group: list[dict[str, Any]]) -> dict[str, Any]:
        accs = [a for r in group for a in r["line_acc"]]
        neds = [n for r in group for n in r["line_neds"]]
        return {
            "acc": (sum(accs) / len(accs)) if accs else 0.0,
            "ned": (sum(neds) / len(neds)) if neds else 0.0,
            "num_lines": len(accs),
            "num_images": len(group),
        }

    scored_rows = [r for r in rows if r.get("scored", True)]
    breakdown: dict[str, dict[str, Any]] = {}
    for key in sorted({f"{r['kind']}/{r['lang']}" for r in scored_rows}):
        kind, lang = key.split("/")
        group = [r for r in scored_rows if r["kind"] == kind and r["lang"] == lang]
        breakdown[key] = _means(group)
    num_missing = len(rows) - len(scored_rows)
    return {
        "overall": _means(scored_rows),
        "breakdown": breakdown,
        "num_images": len(rows),
        "num_scored": len(scored_rows),
        "num_missing": num_missing,
        "num_env_failed": num_missing,  # unavailable PP-OCR engine, not a miss
        "complete": num_missing == 0,
    }


# ── scorer (harness-facing) ──────────────────────────────────────────────────
class GlyphDraw2Scorer:
    """Evaluate a flat generated-image dir against GlyphDraw2 prompt records."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        ocr_lang: str = "en",
        device: str = "auto",
        char2id: dict[str, int] | None = None,
        ocr: OCR | None = None,  # test seam: inject a fake OCR
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.ocr_lang = ocr_lang
        self.device = device
        self.char2id = char2id
        self._ocr = ocr

    def ocr_engine(self) -> OCR:
        if self._ocr is None:
            self._ocr = PaddleOCREngine("glyphdraw2", device=self.device)
        return self._ocr

    def score_image(self, image: Any, record: dict, image_path: Any = None) -> dict[str, Any]:
        """Score one image against its record; an OCR failure scores zero lines.

        A failure to build the OCR engine raises :class:`AqaError` and is recorded
        as ``scored=False`` (environmental: the OCR component is unavailable);
        other per-image exceptions keep the zero-line reference behavior and
        count as scored.
        """
        ocr = self.ocr_engine()
        gt_texts = record.get("gt_texts") or extract_gt_texts(record["prompt"])
        ocr_scored = True
        try:
            pred_texts = ocr.recognize(image, record.get("lang") or self.ocr_lang)
        except AqaError as exc:
            _log.warning("glyphdraw2: OCR unavailable for %s: %s", image_path, exc)
            pred_texts = []
            ocr_scored = False
        except Exception as exc:  # noqa: BLE001 - failed OCR -> nothing recognized
            _log.warning("glyphdraw2: OCR failed for %s: %s", image_path, exc)
            pred_texts = []
        scores = line_scores(gt_texts, pred_texts, char2id=self.char2id)
        return {
            "prompt": record["prompt"],
            "kind": record.get("kind", ""),
            "lang": record.get("lang") or self.ocr_lang,
            "gt_texts": gt_texts,
            "pred_texts": list(pred_texts),
            "matched": scores["matched"],
            "num_lines": scores["num_lines"],
            "line_acc": scores["line_acc"],
            "line_neds": scores["line_neds"],
            "scored": ocr_scored,
        }

    def evaluate_rows(
        self, rows: list[dict], records: list[dict], *, image_loader=None,
    ) -> list[dict[str, Any]]:
        """Score every generated image row (``idx`` aligns with ``records``)."""
        if image_loader is None:
            from PIL import Image
            image_loader = Image.open
        by_idx = {int(r["idx"]): r for r in records}
        out: list[dict[str, Any]] = []
        for row in rows:
            idx = int(row["idx"])
            record = by_idx.get(idx)
            if record is None:
                _log.warning("glyphdraw2: no prompt record for idx %d; skipped", idx)
                continue
            image_path = row["image_path"]
            result = self.score_image(image_loader(image_path), record, image_path=image_path)
            out.append({"filename": str(image_path), "idx": idx, **result})
        return out

    def write_results(self, rows: list[dict[str, Any]], out_prefix: Path) -> tuple[Path, Path]:
        """Write results.jsonl plus a human-readable per-split summary txt."""
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        jsonl = out_prefix.with_suffix(".jsonl")
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary = summarize(rows)
        txt = out_prefix.with_suffix(".txt")
        lines = [
            f"{key}: acc={m['acc']:.4f} ned={m['ned']:.4f} lines={m['num_lines']} "
            f"images={m['num_images']}"
            for key, m in summary["breakdown"].items()
        ]
        overall = summary["overall"]
        lines.append(
            f"overall: acc={overall['acc']:.4f} ned={overall['ned']:.4f} "
            f"lines={overall['num_lines']} images={summary['num_images']}"
        )
        if summary["num_missing"]:
            lines.append(
                f"missing: {summary['num_missing']} "
                f"(environmental/unavailable-scorer: {summary['num_env_failed']})"
            )
        txt.write_text("\n".join(lines) + "\n")
        return jsonl, txt

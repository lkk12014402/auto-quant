"""GenEval benchmark: detection-based text-to-image alignment scoring.

Port of djghosh13/geneval's ``evaluation/evaluate_images.py`` +
``summary_scores.py`` (MIT, (c) 2023 Dhruba Ghosh -- see NOTICE; vendored
prompts/classes live in ``recipes/data/geneval/``).  The one deliberate
substitution: the official Mask2Former runs via mmdet; we use transformers'
``Mask2FormerForUniversalSegmentation`` instead (no mmdet dependency).  Its
instance post-processing yields per-segment masks + scores but *no boxes*, so
boxes are derived as mask bounding rects -- everything downstream (IoU
dedup, relative position, masked color crops) then matches the official math.

Official pipeline per image: per class keep detections with score >
``THRESHOLD`` (``COUNTING_THRESHOLD`` for the counting tag), cap at
``MAX_OBJECTS`` sorted by score, drop exact-duplicate boxes (IoU == 1.0);
``include`` clauses are AND-ed (count / zero-shot CLIP color on masked crops /
relative position), ``exclude`` clauses OR-ed.  Aggregation: per-tag mean of
per-image correctness, overall = unweighted mean of the six task means.

Heavy deps (torch / transformers / PIL / numpy) import lazily so the pure
scoring logic (IoU, position, evaluate, summarize) stays unit-testable without
a GPU stack.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.geneval")

# Repo-vendored official data (prompts/evaluation_metadata.jsonl +
# evaluation/object_names.txt from djghosh13/geneval).
DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "geneval"

DEFAULT_DETECTOR = "facebook/mask2former-swin-small-coco-instance"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

# Official options (evaluate_images.py defaults).
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0  # only exact (IoU == 1.0) duplicates are dropped
POSITION_THRESHOLD = 0.1

COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]
# Zero-shot color templates (official: classname substituted, {c} the color).
_COLOR_TEMPLATES = [
    "a photo of a {c} {classname}",
    "a photo of a {c}-colored {classname}",
    "a photo of a {c} object",
]

TASKS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")


def load_classnames(data_dir: Path | None = None) -> list[str]:
    """The 80 COCO class names (vendored ``object_names.txt``), in order."""
    path = (data_dir or DATA_DIR) / "object_names.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ── pure geometry (ported verbatim in behavior) ──────────────────────────────
def compute_iou(box_a, box_b) -> float:
    """IoU of two ``[x0, y0, x1, y1, ...]`` boxes (+1 pixel area convention)."""
    def area(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    inter = area([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
    ])
    union = area(box_a) + area(box_b) - inter
    return inter / union if union else 0


def relative_position(obj_a, obj_b, *, position_threshold: float = POSITION_THRESHOLD) -> set[str]:
    """Position of A relative to B, factoring in object dimensions.

    Port of the official rule: shrink the center offset by
    ``POSITION_THRESHOLD * (dim_a + dim_b)`` per axis (dead-zone -> no
    relation), then normalise by the *original* offset norm; |dx|/|dy| beyond
    0.5 select left/right and above/below.
    """
    boxes = [obj_a[:4], obj_b[:4]]
    center_a = ((boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2)
    center_b = ((boxes[1][0] + boxes[1][2]) / 2, (boxes[1][1] + boxes[1][3]) / 2)
    dim_a = (abs(boxes[0][2] - boxes[0][0]), abs(boxes[0][3] - boxes[0][1]))
    dim_b = (abs(boxes[1][2] - boxes[1][0]), abs(boxes[1][3] - boxes[1][1]))
    offset = (center_a[0] - center_b[0], center_a[1] - center_b[1])
    revised = tuple(
        max(abs(o) - position_threshold * (da + db), 0.0) * (1.0 if o > 0 else -1.0 if o < 0 else 0.0)
        for o, da, db in zip(offset, dim_a, dim_b)
    )
    if all(abs(r) < 1e-3 for r in revised):
        return set()
    norm = math.hypot(*offset)
    dx, dy = revised[0] / norm, revised[1] / norm
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def filter_detections(
    per_class: dict[str, list],
    *,
    threshold: float = THRESHOLD,
    max_objects: int = MAX_OBJECTS,
    nms_threshold: float = NMS_THRESHOLD,
) -> dict[str, list]:
    """Threshold -> sort by score desc -> cap -> drop exact duplicates.

    ``per_class`` maps a class name to raw ``([x0, y0, x1, y1, score], mask)``
    detections.  Mirrors the per-class filtering of the official
    ``evaluate_image`` (score strictly greater than the threshold; cap applied
    before dedup; with the default ``NMS_THRESHOLD == 1.0`` only boxes with
    IoU == 1.0 against an already-kept box are dropped).
    """
    detected: dict[str, list] = {}
    for classname, dets in per_class.items():
        dets = sorted(
            (d for d in dets if d[0][4] > threshold),
            key=lambda d: d[0][4], reverse=True,
        )[:max_objects]
        kept = []
        for det in dets:
            if all(compute_iou(det[0], prev[0]) < nms_threshold for prev in kept):
                kept.append(det)
        if kept:
            detected[classname] = kept
    return detected


# ── include/exclude evaluation (ported) ──────────────────────────────────────
def evaluate(
    image,
    objects: dict[str, list],
    metadata: dict[str, Any],
    *,
    classify_color,
    position_threshold: float = POSITION_THRESHOLD,
) -> tuple[bool, str]:
    """Evaluate detected ``objects`` against the prompt's ``metadata``.

    ``classify_color(image, detections, classname) -> list[str]`` is the
    zero-shot color classifier (CLIP-backed in production, faked in tests).
    Assumptions match upstream: include clauses combine with AND, exclude with
    OR; color/position only run on the most confident (top-``count``) objects.
    """
    correct = True
    reason = []
    matched_groups: list[list | None] = []
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[: req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                colors = classify_color(image, found_objects, classname)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(
                                obj[0], target_obj[0], position_threshold=position_threshold
                            )
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(
                f"expected {classname}<{req['count']}, found {len(objects[classname])}"
            )
    return correct, "\n".join(reason)


# ── aggregation (summary_scores.py port) ─────────────────────────────────────
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tag mean of per-image correctness; overall = unweighted task mean."""
    by_tag: dict[str, list[bool]] = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(bool(row["correct"]))
    task_means = {tag: sum(vals) / len(vals) for tag, vals in by_tag.items()}
    overall = sum(task_means.values()) / len(task_means) if task_means else 0.0
    return {"overall": overall, "task_means": task_means, "num_images": len(rows)}


# ── detector + CLIP-backed scorer ────────────────────────────────────────────
class GenevalScorer:
    """Mask2Former detection + CLIP color classification over generated images.

    Models load lazily on first use so constructing the scorer (and importing
    this module) stays cheap and GPU-free; tests inject fakes by monkeypatching
    :meth:`detect` / :meth:`classify_colors`.
    """

    def __init__(
        self,
        *,
        detector_model: str = DEFAULT_DETECTOR,
        clip_model: str = DEFAULT_CLIP_MODEL,
        device: str = "auto",
        data_dir: str | Path | None = None,
        threshold: float = THRESHOLD,
        counting_threshold: float = COUNTING_THRESHOLD,
        max_objects: int = MAX_OBJECTS,
        nms_threshold: float = NMS_THRESHOLD,
        position_threshold: float = POSITION_THRESHOLD,
    ):
        self.detector_model = detector_model
        self.clip_model = clip_model
        self.device = device
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.threshold = threshold
        self.counting_threshold = counting_threshold
        self.max_objects = max_objects
        self.nms_threshold = nms_threshold
        self.position_threshold = position_threshold
        self.classnames = load_classnames(self.data_dir)
        self._detector = None
        self._det_processor = None
        self._clip = None
        self._clip_processor = None
        self._color_classifiers: dict[str, Any] = {}

    # -- lazy model loading --------------------------------------------------
    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_detector(self):
        if self._detector is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise AqaError(
                "geneval scoring needs torch + transformers: pip install transformers torch"
            ) from exc
        device = self._resolve_device()
        self._detector = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.detector_model
        ).to(device)
        self._detector.eval()
        self._det_processor = AutoImageProcessor.from_pretrained(self.detector_model)

    def _load_clip(self):
        if self._clip is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise AqaError(
                "geneval color scoring needs torch + transformers: pip install transformers torch"
            ) from exc
        device = self._resolve_device()
        self._clip = CLIPModel.from_pretrained(self.clip_model).to(device)
        self._clip.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model)

    # -- detection -----------------------------------------------------------
    # The HF/mmdet-converted Mask2Former checkpoint uses mmdet-style class names
    # while GenEval's object_names.txt uses COCO-official names. Alias to the
    # GenEval names (verified diff of the two 80-class lists, 2026-08-26).
    _CLASS_ALIASES: ClassVar[dict[str, str]] = {
        "aeroplane": "airplane",
        "diningtable": "dining table",
        "keyboard": "computer keyboard",
        "motorbike": "motorcycle",
        "mouse": "computer mouse",
        "pottedplant": "potted plant",
        "remote": "tv remote",
        "sofa": "couch",
        "tvmonitor": "tv",
    }

    def _classname_for(self, label_id: int) -> str | None:
        """Map a detector label id to the vendored COCO class name."""
        id2label = getattr(self._detector.config, "id2label", None) or {}
        name = id2label.get(label_id, id2label.get(str(label_id)))
        name = self._CLASS_ALIASES.get(name, name)
        if name is None and 0 <= label_id < len(self.classnames):
            name = self.classnames[label_id]
        return name if name in set(self.classnames) else None

    def detect(self, filepath) -> tuple[Any, dict[str, list]]:
        """Run the detector; return ``(image, per_class)`` raw detections.

        Query-level extraction from the raw outputs (bypasses the HF
        post-processor): score = **class probability** (mmdet-parity semantics --
        HF's segment score is class_prob x mask_quality, which deflates scores and
        breaks GenEval's calibrated thresholds, esp. counting@0.9), mask = the
        query's own binary map (no argmax pixel-stealing), box = mask bounding
        rect in the official ``[x0, y0, x1, y1, score]`` shape.
        """
        self._load_detector()
        import torch
        import torch.nn.functional as F
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(filepath))
        inputs = self._det_processor(images=image, return_tensors="pt").to(
            self._resolve_device()
        )
        with torch.no_grad():
            outputs = self._detector(**inputs)
        class_probs = outputs.class_queries_logits[0].softmax(-1)[:, :-1]  # (Q, C)
        scores, labels = class_probs.max(-1)  # (Q,)
        masks_logits = outputs.masks_queries_logits[0]  # (Q, h, w)
        masks = F.interpolate(
            masks_logits[None], size=image.size[::-1], mode="bilinear",
            align_corners=False,
        )[0]
        per_class: dict[str, list] = {}
        for q in range(masks.shape[0]):
            classname = self._classname_for(int(labels[q]))
            if classname is None:
                continue
            mask = masks[q].sigmoid() > 0.5
            ys, xs = mask.nonzero(as_tuple=True)
            if xs.numel() == 0:
                continue
            box = [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item(),
                   float(scores[q])]
            per_class.setdefault(classname, []).append((box, mask.cpu().numpy()))
        return image, per_class

    # -- zero-shot color classification (CLIP) --------------------------------
    @staticmethod
    def _as_tensor(feats):
        """transformers 5.x returns BaseModelOutputWithPooling from get_*_features;
        older versions return a bare tensor. Duck-typed (no torch import needed)."""
        if hasattr(feats, "norm"):
            return feats
        pooler = getattr(feats, "pooler_output", None)
        return pooler if pooler is not None else feats

    def _color_weights(self, classname: str):
        """Normalized mean text embedding per color (cached per classname)."""
        if classname in self._color_classifiers:
            return self._color_classifiers[classname]
        self._load_clip()
        import torch

        templates = [t.replace("{classname}", classname) for t in _COLOR_TEMPLATES]
        embs = []
        with torch.no_grad():
            for color in COLORS:
                texts = [t.format(c=color) for t in templates]
                inputs = self._clip_processor(
                    text=texts, return_tensors="pt", padding=True, truncation=True
                ).to(self._resolve_device())
                feats = self._as_tensor(self._clip.get_text_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats.mean(dim=0))
        weights = torch.stack(embs)
        weights = weights / weights.norm(dim=-1, keepdim=True)
        self._color_classifiers[classname] = weights
        return weights

    @staticmethod
    def _masked_crop(image, box, mask):
        """Composite the object on a #999 background by its mask, then crop."""
        from PIL import Image

        if mask is not None:
            import numpy as np

            # PIL composite only accepts mode 1/L/F masks; binary maps arrive as
            # float/bool arrays -> normalize to uint8 (0/255).
            if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
                mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
            blank = Image.new("RGB", image.size, color="#999")
            image = Image.composite(image, blank, Image.fromarray(mask))
        return image.crop(box[:4])

    def classify_colors(self, image, detections, classname: str) -> list[str]:
        """Zero-shot color label per detection (score-sorted) via CLIP."""
        if not detections:
            return []
        self._load_clip()
        import torch

        weights = self._color_weights(classname)
        crops = [self._masked_crop(image, box, mask) for box, mask in detections]
        inputs = self._clip_processor(images=crops, return_tensors="pt").to(
            self._resolve_device()
        )
        with torch.no_grad():
            feats = self._as_tensor(self._clip.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        pred = (feats @ weights.T).argmax(dim=-1)
        return [COLORS[i] for i in pred.tolist()]

    # -- per-image / per-directory evaluation ---------------------------------
    def evaluate_image(self, filepath, metadata: dict[str, Any]) -> dict[str, Any]:
        """Score one image; returns the official results.jsonl row shape."""
        image, per_class = self.detect(filepath)
        # counting prompts use the stricter confidence threshold
        threshold = (
            self.counting_threshold if metadata["tag"] == "counting" else self.threshold
        )
        detected = filter_detections(
            per_class, threshold=threshold, max_objects=self.max_objects,
            nms_threshold=self.nms_threshold,
        )
        is_correct, reason = evaluate(
            image, detected, metadata,
            classify_color=self.classify_colors,
            position_threshold=self.position_threshold,
        )
        return {
            "filename": str(filepath),
            "tag": metadata["tag"],
            "prompt": metadata["prompt"],
            "correct": is_correct,
            "reason": reason,
            "metadata": json.dumps(metadata),
            "details": json.dumps({
                key: [list(map(float, box)) for box, _ in value]
                for key, value in detected.items()
            }),
        }

    def evaluate_directory(self, imagedir: str | Path) -> list[dict[str, Any]]:
        """Score an official-layout image dir (``<idx>/metadata.jsonl`` +
        ``<idx>/samples/<j>.png``); port of the official ``main()`` walk."""
        imagedir = Path(imagedir)
        if not imagedir.is_dir():
            return []
        results = []
        total = sum(
            len([p for p in (d / "samples").iterdir()
                 if p.is_file() and re.match(r"\d+\.png", p.name)])
            for d in imagedir.iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "samples").is_dir()
        )
        for folderpath in sorted(imagedir.iterdir()):
            if not folderpath.is_dir() or not folderpath.name.isdigit():
                continue
            metadata = json.loads((folderpath / "metadata.jsonl").read_text())
            samples = folderpath / "samples"
            for imagepath in sorted(samples.iterdir()):
                if not imagepath.is_file() or not re.match(r"\d+\.png", imagepath.name):
                    continue
                results.append(self.evaluate_image(imagepath, metadata))
                if len(results) % 100 == 0:
                    _log.info("geneval scoring: %d/%d images", len(results), total)
        return results

    def write_results(self, rows: list[dict[str, Any]], outfile: str | Path) -> Path:
        """Write rows as results.jsonl (same shape as the official tool)."""
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        return outfile

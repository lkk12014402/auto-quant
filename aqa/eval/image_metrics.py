"""Image-metric scorers for the diffusion harness (DrawBench-style suites).

Three scorers, all pluggable per recipe:

* ``clip``       -- CLIP image-text cosine x100 (``logits_per_image`` scale),
  the historical DrawBench score.
* ``clip_iqa``   -- CLIP-IQA no-reference quality: for each image, softmax over
  the image's similarity to antonym prompt pairs (positive vs negative),
  reported x100.  Reuses the same CLIP checkpoint, no extra dependency.
* ``imagereward`` -- ImageReward-v1.0 human-preference reward model.

ImageReward isolation (verified 2026-08-28)
-------------------------------------------
The ``image-reward`` PyPI package (1.5, 2023-era) is deeply incompatible with
transformers v5: it imports v4-only symbols that were removed
(``apply_chunking_to_forward``, ``find_pruneable_heads_and_indices``,
``prune_linear_layer``, ``additional_special_tokens_ids``,
``additional_special_tokens`` -> ``extra_special_tokens``,
``all_tied_weights_keys``).  Monkey-patching those back means patching model
loading internals -- a silently-wrong score is worse than no score.  Instead
the scorer runs in a subprocess with ``PYTHONPATH`` pointed at an isolated
package dir holding the verified golden combination:

    transformers==4.44.2  diffusers==0.29.2  image-reward  clip(zip archive)

``transformers>=4.45`` is needed by new diffusers but dropped the v4 symbols;
``transformers==4.45..4.56`` all removed them progressively, so 4.44.2 is the
sweet spot; new diffusers (>=0.30) needs ``Dinov2WithRegistersConfig`` (4.45+),
hence the diffusers pin.  ``clip`` comes from the GitHub zip archive because
minimal containers ship no git.  Verified: model loads and ranks a matching
prompt above a mismatching one (1.27 vs -2.00).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.image_metrics")

# ── ImageReward isolated environment ─────────────────────────────────────────
IMAGEREWARD_ENV_DIR = Path(
    os.environ.get("AQA_IMAGEREWARD_ENV", "~/.cache/aqa/envs/imagereward")
).expanduser()
IMAGEREWARD_PINS = [
    "transformers==4.44.2",
    "diffusers==0.29.2",
    "image-reward",
    "clip @ https://github.com/openai/CLIP/archive/refs/heads/main.zip",
]
IMAGEREWARD_MODEL = "ImageReward-v1.0"

# ``pip install --target`` drags in image-reward's transitive torch/torchvision
# (+ their nvidia/cuda/triton wheels). Because the scorer subprocess prepends the
# isolated dir to PYTHONPATH, that torch would shadow the host's CUDA-correct
# build -- and the pulled-in torch can be an ABI-mismatched build whose op table
# is incomplete (``'_OpNamespace' 'aten' object has no attribute 'cholesky'``).
# We purge the whole torch ecosystem from the isolated dir so ``import torch`` /
# ``import torchvision`` fall through to the host interpreter's working install,
# while keeping the isolated transformers/diffusers/timm/clip pins.
_TORCH_ECOSYSTEM_ROOTS = (
    "torch", "torchvision", "torchgen", "functorch", "triton",
    "pytorch_triton", "nvidia", "cuda", "cuda_bindings", "cuda_pathfinder",
    "cuda_toolkit",
)

# CLIP-IQA antonym prompt pairs (positive, negative), CLIP-IQA paper style.
CLIP_IQA_PROMPT_PAIRS: list[tuple[str, str]] = [
    ("Good photo.", "Bad photo."),
    ("Good quality.", "Bad quality."),
    ("Aesthetically pleasing.", "Aesthetically displeasing."),
]

METRICS = ("clip", "clip_iqa", "imagereward")


def _purge_host_torch_from_target(target: Path) -> None:
    """Remove torch-ecosystem packages from the isolated --target dir.

    The subprocess prepends ``target`` to PYTHONPATH, so any torch/torchvision
    installed here would shadow (and break, via ABI mismatch) the host's
    CUDA-correct torch. Deleting them makes ``import torch`` resolve to the host.
    """
    import shutil

    for entry in target.iterdir():
        stem = entry.name.split("-", 1)[0].split(".", 1)[0]
        if stem in _TORCH_ECOSYSTEM_ROOTS:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def ensure_imagereward_env() -> Path:
    """Provision the isolated ImageReward package dir (idempotent).

    Uses the *current* interpreter's pip with ``--target`` so no git/sudo is
    needed; the dir then shadows nothing (it is only ever used via PYTHONPATH).
    """
    if (IMAGEREWARD_ENV_DIR / "ImageReward").is_dir():
        # A previously-provisioned env may predate the torch purge below; make
        # the idempotent path self-healing so stale torch wheels get removed.
        _purge_host_torch_from_target(IMAGEREWARD_ENV_DIR)
        return IMAGEREWARD_ENV_DIR
    IMAGEREWARD_ENV_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--target", str(IMAGEREWARD_ENV_DIR), *IMAGEREWARD_PINS]
    _log.info("provisioning isolated ImageReward env at %s", IMAGEREWARD_ENV_DIR)
    rc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if rc.returncode != 0:
        raise AqaError(
            f"ImageReward env provisioning failed (rc={rc.returncode}):\n{rc.stderr[-2000:]}"
        )
    if not (IMAGEREWARD_ENV_DIR / "ImageReward").is_dir():
        raise AqaError(f"ImageReward env provisioning produced no package at {IMAGEREWARD_ENV_DIR}")
    _purge_host_torch_from_target(IMAGEREWARD_ENV_DIR)
    return IMAGEREWARD_ENV_DIR


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ── CLIP-family scorers (in-process, transformers CLIP) ──────────────────────
def _load_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def _clip_inputs(processor, text: str, image, device: str):
    # DrawBench prompts can exceed CLIP's 77-token context; truncation is the
    # standard CLIP-score convention (transformers 5.x hard-errors otherwise).
    return processor(
        text=[text], images=[image], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)


def clip_score(rows: list[dict], model_id: str, device: str = "auto") -> list[float]:
    """CLIP image-text cosine x100 per row (CLIP's logits_per_image scale)."""
    import torch
    from PIL import Image

    device = _device(device)
    model, processor = _load_clip(model_id, device)
    scores: list[float] = []
    for row in rows:
        image = Image.open(row["image_path"])
        inputs = _clip_inputs(processor, row["prompt"], image, device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores.append(float(outputs.logits_per_image[0, 0]))
    return scores


def clip_iqa(rows: list[dict], model_id: str, device: str = "auto") -> list[float]:
    """CLIP-IQA: mean over antonym pairs of softmax(pos, neg)[pos] x100."""
    import torch
    from PIL import Image

    device = _device(device)
    model, processor = _load_clip(model_id, device)
    scores: list[float] = []
    for row in rows:
        image = Image.open(row["image_path"])
        pair_scores: list[float] = []
        for positive, negative in CLIP_IQA_PROMPT_PAIRS:
            inputs = processor(
                text=[positive, negative], images=[image, image],
                return_tensors="pt", padding=True,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # logits_per_image is (2 texts, 2 images); take each text vs its image
            sims = outputs.logits_per_image[[0, 1], [0, 1]].softmax(dim=0)
            pair_scores.append(float(sims[0]) * 100.0)
        scores.append(sum(pair_scores) / len(pair_scores))
    return scores


# ── ImageReward scorer (subprocess, isolated v4 env) ─────────────────────────
_IMAGEREWARD_WORKER = r"""
import json, sys
from PIL import Image

def main():
    payload = json.loads(sys.argv[1])
    import ImageReward
    rm = ImageReward.load(payload["model"])
    out = []
    for row in payload["rows"]:
        out.append(float(rm.score(row["prompt"], [Image.open(row["image_path"])])))
    print(json.dumps(out))

if __name__ == "__main__":
    main()
"""


def imagereward_score(rows: list[dict], model_id: str = IMAGEREWARD_MODEL,
                      device: str = "auto") -> list[float]:
    """ImageReward-v1.0 score per row, computed in the isolated v4 subprocess."""
    import tempfile

    if not rows:
        return []
    env_dir = ensure_imagereward_env()
    with tempfile.TemporaryDirectory() as tmp:
        worker = Path(tmp) / "worker.py"
        worker.write_text(_IMAGEREWARD_WORKER)
        payload = json.dumps({
            "model": model_id,
            "rows": [{"prompt": r["prompt"], "image_path": r["image_path"]} for r in rows],
        })
        env = dict(os.environ)
        env["PYTHONPATH"] = str(env_dir) + os.pathsep + env.get("PYTHONPATH", "")
        # ImageReward reads CUDA itself; scorer_device=cpu forces it off-GPU.
        if device == "cpu" or (device == "auto" and not _cuda_available()):
            env["CUDA_VISIBLE_DEVICES"] = ""
        rc = subprocess.run(
            [sys.executable, str(worker), payload],
            capture_output=True, text=True, env=env, check=False,
        )
        if rc.returncode != 0:
            raise AqaError(
                f"ImageReward scorer subprocess failed (rc={rc.returncode}):\n{rc.stderr[-2000:]}"
            )
        scores = json.loads(rc.stdout.strip().splitlines()[-1])
    if len(scores) != len(rows):
        raise AqaError(f"ImageReward returned {len(scores)} scores for {len(rows)} rows")
    return scores


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        return False


def score_rows(rows: list[dict], metrics: list[str], *, clip_model: str,
               imagereward_model: str = IMAGEREWARD_MODEL,
               device: str = "auto") -> dict[str, list[float]]:
    """Score rows with each requested metric; return {metric: [scores]}."""
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise AqaError(f"unknown image metric(s) {unknown}; choose from {METRICS}")
    out: dict[str, list[float]] = {}
    for metric in metrics:
        if metric == "clip":
            out[metric] = clip_score(rows, clip_model, device)
        elif metric == "clip_iqa":
            out[metric] = clip_iqa(rows, clip_model, device)
        elif metric == "imagereward":
            out[metric] = imagereward_score(rows, imagereward_model, device)
    return out

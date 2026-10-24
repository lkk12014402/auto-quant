"""Architecture-agnostic model structure from *safetensors headers* (RFC 0002).

Both ANALYZE (the true module→params structure map) and CLOSE
(``quantization_details`` ground truth) need to know how many parameters live in
each module family of a model — for *any* architecture (dense, MoE, multimodal,
hybrid linear-attention), not just a canonical dense transformer.

This module reads only the safetensors **header** (the leading JSON block that
records each tensor's ``dtype``/``shape``), never the weight tensors themselves,
so it is cheap, **local-only, no GPU, no network**:

* It operates purely on files already on disk.  If the model is a remote HF id
  that has not been downloaded, or has no ``*.safetensors`` shards, every entry
  point returns ``None`` and the caller falls back to a config-dims estimate.
* ``classify_module`` buckets each tensor into a coarse, architecture-agnostic
  family by name, so new architectures degrade to ``other`` rather than breaking.

Nothing here downloads, loads weights, or allocates accelerators.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, NamedTuple

from ..logging_util import get_logger

_log = get_logger("report.weight_map")

# Families that never carry a quantizable Linear weight (precision-critical or
# non-Linear): embeddings, the MoE router (gate), norms, misc buffers.
_ALWAYS_KEEP_FAMILIES = frozenset({"embed", "router", "norms", "other"})


class ModuleInfo(NamedTuple):
    family: str
    label: str
    quantizable: bool   # family *can* hold a quantizable Linear weight
    is_lm_head: bool


_FAMILY_LABELS: dict[str, str] = {
    "visual": "Vision tower",
    "mtp": "Multi-token prediction",
    "linear-attn": "Linear attention",
    "shared-expert": "Shared expert projections",
    "moe-experts": "MoE expert projections",
    "router": "MoE router (gate)",
    "self-attn": "Attention projections",
    "mlp": "MLP projections",
    "embed": "Token embedding",
    "lm-head": "LM head",
    "norms": "LayerNorms",
    "other": "Other parameters",
}


def classify_module(name: str) -> ModuleInfo:
    """Bucket a tensor name into a coarse, architecture-agnostic module family.

    Order matters: container families (``visual``/``mtp``) are matched first so
    their inner ``attn``/``mlp`` submodules group with the container, and the
    MoE router (``mlp.gate``) is distinguished from ``mlp.gate_proj``.
    """
    low = name.lower()
    if "visual" in low or "vision_model" in low or ".vision." in low:
        fam = "visual"
    elif "mtp" in low:
        fam = "mtp"
    elif "linear_attn" in low or "mamba" in low or ".ssm" in low:
        fam = "linear-attn"
    elif "shared_expert" in low:
        fam = "shared-expert"
    elif "experts" in low:
        fam = "moe-experts"
    elif re.search(r"\.(mlp|block_sparse_moe)\.gate\.(weight|bias)$", low):
        fam = "router"
    elif "self_attn" in low or re.search(r"\.attn\.", low):
        fam = "self-attn"
    elif ".mlp." in low or low.endswith(".mlp"):
        fam = "mlp"
    elif "embed_tokens" in low or low.endswith("embeddings.weight") or ".wte." in low:
        fam = "embed"
    elif "lm_head" in low:
        fam = "lm-head"
    elif "norm" in low:
        fam = "norms"
    else:
        fam = "other"
    quantizable = fam not in _ALWAYS_KEEP_FAMILIES
    return ModuleInfo(fam, _FAMILY_LABELS[fam], quantizable, fam == "lm-head")


def module_leaf(name: str) -> str:
    """A compact representative leaf for a tensor (last non-numeric components).

    ``model.language_model.layers.5.self_attn.q_proj.weight`` → ``self_attn.q_proj``.
    """
    stem = name[:-len(".weight")] if name.endswith(".weight") else name
    stem = stem[:-len(".bias")] if stem.endswith(".bias") else stem
    parts = [p for p in stem.split(".") if not p.isdigit()]
    return ".".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else stem)


def _layer_index(name: str) -> int | None:
    m = re.search(r"\.(?:layers|blocks|h)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _tensor_params(shape: list[int]) -> int:
    out = 1
    for x in shape:
        out *= int(x)
    return out


def _resolve_model_dir(model_source: str | Path | None) -> Path | None:
    """Local directory holding a model's weights, resolving cached HF repo ids.

    A path that exists on disk is returned as-is.  A bare HF repo id
    (``org/name``) already present in the HF cache is resolved to its snapshot
    directory — first via ``huggingface_hub`` with ``local_files_only=True``
    (verifies the revision), then, when that stricter check fails for a lazily
    populated cache (``from_pretrained`` does not write the completeness markers
    ``snapshot_download`` expects), by scanning the cache for a snapshot whose
    ``*.safetensors`` shards are all present per its ``index.json``.  Either way
    this only reads what is already on disk and **never triggers a download**.
    Returns ``None`` when nothing complete is locally available.
    """
    if not model_source:
        return None
    p = Path(model_source)
    if p.is_dir():
        return p
    if p.exists():  # a file, not a model dir
        return None
    s = str(model_source)
    if "/" not in s:
        return None
    try:
        from huggingface_hub import snapshot_download
        cached = snapshot_download(
            s, local_files_only=True, allow_patterns=["*.safetensors", "*.json"])
        return Path(cached)
    except Exception:  # noqa: BLE001 - stricter check failed; try a lenient scan
        return _scan_cache_snapshot(s)


def _scan_cache_snapshot(repo_id: str) -> Path | None:
    """Newest cached snapshot dir whose safetensors shards are fully present.

    Handles a cache populated by ``from_pretrained`` (no ``snapshot_download``
    completeness markers): a snapshot is usable only when every shard its
    ``model.safetensors.index.json`` references is on disk (so param totals stay
    exact).  A shardless single-file export counts as complete.  Never downloads.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        snaps = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if not snaps.is_dir():
            return None
        best: tuple[float, Path] | None = None
        for d in snaps.iterdir():
            if not d.is_dir():
                continue
            shards = {p.name for p in d.glob("*.safetensors")}
            if not shards:
                continue
            index = d / "model.safetensors.index.json"
            if index.is_file():
                try:
                    referenced = set(
                        json.loads(index.read_text()).get("weight_map", {}).values())
                except (OSError, ValueError, json.JSONDecodeError):
                    referenced = set()
                if referenced and not referenced <= shards:
                    continue  # incomplete download — fall back to config dims
            mtime = d.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, d)
        return best[1] if best else None
    except Exception:  # noqa: BLE001 - cache layout unavailable
        return None


def _read_header(path: Path) -> dict[str, Any] | None:
    """Parse one safetensors file header (the leading JSON), no tensor data."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 200_000_000:  # guard against non-safetensors files
                return None
            return json.loads(f.read(n))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def read_param_shapes(model_source: str | Path | None) -> dict[str, list[int]] | None:
    """Map ``tensor name → shape`` from a local model's safetensors headers.

    Accepts a local directory, or a bare HF repo id that has **already been
    downloaded** — the latter resolves to its cached snapshot via
    ``huggingface_hub`` with ``local_files_only=True`` (never triggers a
    download).  Returns ``None`` when nothing is locally available, there are no
    ``*.safetensors`` shards, or none could be parsed — the signal for callers to
    fall back to config dims.  Never downloads and never loads weights.
    """
    root = _resolve_model_dir(model_source)
    if root is None:
        return None
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        return None
    shapes: dict[str, list[int]] = {}
    for shard in shards:
        header = _read_header(shard)
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            shape = meta.get("shape")
            if isinstance(shape, list):
                shapes[name] = shape
    return shapes or None


def summarize_structure(shapes: dict[str, list[int]]) -> dict[str, Any]:
    """Aggregate ``name → shape`` into a family-level structure map (ANALYZE).

    Produces the deterministic *true* parameter breakdown per module family with
    tensor and distinct-layer counts, ordered by descending parameter share.
    """
    fams: dict[str, dict[str, Any]] = {}
    total = 0
    for name, shape in shapes.items():
        params = _tensor_params(shape)
        total += params
        info = classify_module(name)
        f = fams.setdefault(info.family, {
            "id": info.family, "label": info.label,
            "params": 0, "tensors": 0, "_layers": set()})
        f["params"] += params
        f["tensors"] += 1
        li = _layer_index(name)
        if li is not None:
            f["_layers"].add(li)

    families: list[dict[str, Any]] = []
    for f in sorted(fams.values(), key=lambda x: -x["params"]):
        entry: dict[str, Any] = {
            "id": f["id"], "label": f["label"],
            "params": f["params"],
            "params_ratio": round(f["params"] / total, 4) if total else None,
            "tensors": f["tensors"],
        }
        if f["_layers"]:
            entry["layers"] = len(f["_layers"])
        families.append(entry)
    return {"source": "weight_headers", "total_params": total, "families": families}


def model_structure(model_source: str | Path | None) -> dict[str, Any] | None:
    """Convenience: true structure map from a model's weight headers, or ``None``."""
    shapes = read_param_shapes(model_source)
    if not shapes:
        return None
    try:
        return summarize_structure(shapes)
    except Exception as exc:  # noqa: BLE001 - structure map is best-effort
        _log.warning("weight-header structure summary failed: %s", exc)
        return None


# Suffixes a compressed-tensors / auto-round export attaches to a *quantized*
# Linear weight.  Their presence is the authoritative "this module was quantized"
# signal — more reliable than the ``ignore`` list, which omits modules handled
# outside the quant graph entirely (e.g. MTP tensors copied verbatim as BF16).
_QUANT_ARTIFACT_SUFFIXES = (".weight_scale", ".weight_packed",
                            ".weight_zero_point")


def read_quantized_modules(quant_dir: str | Path | None) -> set[str] | None:
    """Module stems actually quantized in an export (from its weight headers).

    Reads the exported ``*.safetensors`` headers and returns the set of module
    paths that carry a quantization artifact (``weight_scale`` / ``weight_packed``).
    Returns ``None`` when the export is absent or has no such artifacts (e.g. a
    plain BF16 dir) — the caller then falls back to the ``ignore``-list heuristic.
    """
    if not quant_dir:
        return None
    shapes = read_param_shapes(quant_dir)
    if not shapes:
        return None
    quantized: set[str] = set()
    for name in shapes:
        for suf in _QUANT_ARTIFACT_SUFFIXES:
            if name.endswith(suf):
                quantized.add(name[: -len(suf)])
                break
    return quantized or None

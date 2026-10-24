"""``quantization_details`` — the per-module precision & coverage map (RFC 0002).

The INC recipe manifest records *what command ran* and *what score came out*, but
not **which modules of the final model ended up at which precision**.  This module
synthesises that coverage, deterministically and offline:

* **Comprehensive module groups** — every parameter-bearing module family of the
  architecture (attention projections, dense/MoE MLP, router, embeddings,
  ``lm_head``, norms), so the groups together cover the whole parameter budget.
* **Per-group scheme** — the applied precision (e.g. ``MXFP8``) or ``BF16`` when
  the group was kept in high precision, resolved from the *exported* quant config
  (auto-round ``extra_config`` / compressed-tensors ``ignore`` + ``config_groups``)
  falling back to the declared quant scope.
* **Parameter counts & ratios** — ``effective_bits`` is the param-weighted average
  width (nominal, excluding scale bits — matching INC's ``avg_bits`` convention).

Two parameter-counting bases, best to worst fidelity:

1. **Weight-shape ground truth** (``derived_from: weight_shapes``) — when the base
   model's ``*.safetensors`` are on disk, every real tensor is counted from its
   header shape and classified by module family (``weight_map``).  This is exact
   for *any* architecture — multimodal vision towers, hybrid linear-attention,
   MoE — because it never assumes a canonical dense transformer.  Quantized-vs-kept
   is decided per tensor against the exported ``ignore`` list.
2. **Analytical fallback** — when weights are not local (remote id / offline), the
   canonical dense/MoE module groups are synthesised from config dims.  Accurate
   for plain dense/MoE LLMs; approximate for hybrid/multimodal models.

Everything is best-effort: unknown dims degrade to ``null`` params while still
emitting the group+scheme coverage, and any failure returns a minimal block
rather than aborting the run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..logging_util import get_logger
from .weight_map import (
    classify_module,
    module_leaf,
    read_param_shapes,
    read_quantized_modules,
)

_log = get_logger("report.quant_details")

# Module families that are *never* quantized (precision-critical): embeddings,
# the routed-expert gate (router) and norms.  ``lm_head`` is conditional
# (``quant_lm_head``) and handled separately.
_ALWAYS_BF16 = {"embed", "router", "norms"}

# Nominal weight bit-width per scheme label (scale bits excluded).  ``W<w>A<a>``
# takes the weight group; the MX/NV families take their trailing width.
_SCHEME_BITS: dict[str, int] = {
    "BF16": 16, "FP16": 16, "FP32": 32,
    "FP8": 8, "MXFP8": 8, "W8A8": 8, "W8A16": 8,
    "FP6": 6, "MXFP6": 6,
    "FP4": 4, "MXFP4": 4, "NVFP4": 4, "INT4": 4, "W4A16": 4,
}


def _scheme_weight_bits(scheme: str | None) -> int | None:
    """Nominal weight bits for a scheme label (``None`` when unrecognised)."""
    if not scheme:
        return None
    s = scheme.strip().upper()
    if s in _SCHEME_BITS:
        return _SCHEME_BITS[s]
    m = re.match(r"^W(\d+)A\d+$", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:FP|INT)(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def _is_quantizable_weight(name: str, shape: list[int], info: Any) -> bool:
    """Whether a tensor is a quantizable Linear-like weight matrix.

    Standard Linear weights are 2-D and end in ``.weight``.  MoE experts are
    commonly exported *fused* as a 3-D stack ``[num_experts, in, out]`` with no
    ``.weight`` suffix — auto-round still quantizes these (the ``mlp.experts``
    ``layer_config`` targets them), so a 3-D tensor in the experts family counts.
    Conv/SSM weights (e.g. ``conv1d.weight`` ``[C, 1, k]``) and 1-D
    buffers/biases/norms never count.
    """
    if not info.quantizable:
        return False
    nd = len(shape)
    if info.family == "moe-experts":
        return nd == 3 or (nd == 2 and name.endswith(".weight"))
    return nd == 2 and name.endswith(".weight")


def _layer_config_label(spec: dict[str, Any], fallback: str) -> str:
    """Scheme label for a ``layer_config`` override entry (``{bits, data_type}``).

    Mirrors the convention in :mod:`aqa.report.names`: ``mx_fp``+4 → ``MXFP4``,
    ``nv_fp``+4 → ``NVFP4``, a plain float type → ``FP<bits>``, else an integer
    weight-only type → ``INT<bits>``.  An override with ``bits >= 16`` keeps the
    module in BF16.  Falls back to the global scheme when ``bits`` is unusable.
    """
    bits = spec.get("bits")
    if not isinstance(bits, int):
        return fallback
    if bits >= 16:
        return "BF16"
    dtype = str(spec.get("data_type", "")).lower()
    if dtype.startswith("mx_fp") or dtype.startswith("mxfp"):
        return {4: "MXFP4", 6: "MXFP6", 8: "MXFP8"}.get(bits, f"MXFP{bits}")
    if dtype.startswith("nv_fp") or dtype.startswith("nvfp"):
        return {4: "NVFP4", 8: "NVFP8"}.get(bits, f"NVFP{bits}")
    if "fp" in dtype or dtype in ("float", ""):
        return {4: "FP4", 6: "FP6", 8: "FP8"}.get(bits, f"FP{bits}")
    return {4: "INT4", 8: "INT8"}.get(bits, f"INT{bits}")


def _declared_quantized_scheme(
    module: str, declared_scheme: str, layer_config: dict[str, Any] | None,
) -> tuple[str, int | None, str | None, str]:
    """Applied scheme for a module that *is* quantized, honoring ``layer_config``.

    Returns ``(scheme_label, weight_bits, data_type, matched_by)``.  A
    ``layer_config`` key that substring-matches the module path (auto-round's own
    matching semantics) overrides the global ``--scheme`` — this is what pushes
    e.g. ``mlp.experts`` to ``MXFP4`` while the rest of the model stays ``MXFP8``.
    """
    for key, spec in (layer_config or {}).items():
        if not (key and isinstance(spec, dict) and key in module):
            continue
        label = _layer_config_label(spec, declared_scheme)
        matched = f"layer_config:{key}" + (" (keep)" if label == "BF16" else "")
        return label, _scheme_weight_bits(label), spec.get("data_type"), matched
    return declared_scheme, _scheme_weight_bits(declared_scheme), None, "scheme"


def _kept_reason(name: str, shape: list[int], info: Any, kept_subs: set[str],
                 quant_lm_head: bool) -> str:
    """Why a tensor stays BF16 (the per-module disposition ``matched_by``)."""
    if not info.quantizable:
        return f"precision-critical ({info.family})"
    if not _is_quantizable_weight(name, shape, info):
        return "non-Linear weight"
    if info.is_lm_head and not quant_lm_head:
        return "lm_head kept (quant_lm_head=false)"
    module = name[:-len(".weight")] if name.endswith(".weight") else name
    for sub in sorted((s for s in kept_subs if s), key=len, reverse=True):
        if sub in module:
            return f"ignore:{sub}"
    return "kept BF16"


def _num_layers(cfg: dict[str, Any]) -> int | None:
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        if isinstance(cfg.get(key), int):
            return cfg[key]
    text = cfg.get("text_config")
    if isinstance(text, dict):
        return _num_layers(text)
    return None


def _dims(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the parameter-counting dims, tolerating multimodal ``text_config``."""
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(key: str, default: Any = None) -> Any:
        v = cfg.get(key)
        if v is None:
            v = text.get(key)
        return default if v is None else v

    H = g("hidden_size")
    nh = g("num_attention_heads")
    nkv = g("num_key_value_heads", nh)
    hd = g("head_dim") or ((H // nh) if (H and nh) else None)
    ne = g("num_experts") or g("num_local_experts") or g("n_routed_experts")
    moe_i = g("moe_intermediate_size") or g("intermediate_size")
    n_shared = g("n_shared_experts")
    shared_i = g("shared_expert_intermediate_size")
    if shared_i is None and moe_i and n_shared:
        shared_i = moe_i * n_shared
    return {
        "H": H, "L": _num_layers(cfg), "V": g("vocab_size"),
        "nh": nh, "nkv": nkv, "hd": hd,
        "I": g("intermediate_size"),
        "ne": ne, "moe_i": moe_i, "shared_i": shared_i,
        "tie": bool(g("tie_word_embeddings", False)),
        "is_moe": bool(ne),
    }


def _mul(*vals: Any) -> int | None:
    """Multiply, returning ``None`` if any factor is missing."""
    out = 1
    for v in vals:
        if not v:
            return None
        out *= int(v)
    return out


def _group_specs(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Canonical, architecture-aware module groups with analytical param counts.

    ``rep`` is a representative full module path used for substring ``ignore``
    matching (mirrors auto-round's own substring semantics).
    """
    H, L, V = d["H"], d["L"], d["V"]
    nh, nkv, hd = d["nh"], d["nkv"], d["hd"]
    specs: list[dict[str, Any]] = []

    specs.append({
        "id": "embed", "label": "Token embedding",
        "modules": ["model.embed_tokens"], "rep": "model.embed_tokens",
        "params": _mul(V, H), "always_keep": True, "quantizable": False})

    # Attention projections: q,o over nh*hd; k,v over nkv*hd.
    attn = None
    qo = _mul(L, H, nh, hd)
    kv = _mul(L, H, nkv, hd)
    if qo is not None and kv is not None:
        attn = 2 * qo + 2 * kv
    specs.append({
        "id": "attn-proj", "label": "Attention projections",
        "modules": ["self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj"],
        "rep": "model.layers.0.self_attn.q_proj", "params": attn,
        "layers": L, "always_keep": False, "quantizable": True})

    if d["is_moe"]:
        experts = _mul(L, d["ne"], 3, H, d["moe_i"])
        specs.append({
            "id": "moe-experts", "label": "MoE expert projections",
            "modules": ["mlp.experts.*.gate_proj", "mlp.experts.*.up_proj",
                        "mlp.experts.*.down_proj"],
            "rep": "model.layers.0.mlp.experts.0.down_proj", "params": experts,
            "layers": L, "always_keep": False, "quantizable": True})
        if d["shared_i"]:
            shared = _mul(L, 3, H, d["shared_i"])
            specs.append({
                "id": "shared-expert", "label": "Shared expert projections",
                "modules": ["mlp.shared_expert.gate_proj",
                            "mlp.shared_expert.up_proj",
                            "mlp.shared_expert.down_proj"],
                "rep": "model.layers.0.mlp.shared_expert.down_proj",
                "params": shared, "layers": L,
                "always_keep": False, "quantizable": True})
        specs.append({
            "id": "router", "label": "MoE router (gate)",
            "modules": ["mlp.gate"], "rep": "model.layers.0.mlp.gate",
            "params": _mul(L, H, d["ne"]), "layers": L,
            "always_keep": True, "quantizable": False})
    else:
        specs.append({
            "id": "mlp", "label": "MLP projections",
            "modules": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "rep": "model.layers.0.mlp.down_proj", "params": _mul(L, 3, H, d["I"]),
            "layers": L, "always_keep": False, "quantizable": True})

    lm_head_params = 0 if d["tie"] else _mul(V, H)
    specs.append({
        "id": "lm-head", "label": "LM head",
        "modules": ["lm_head"], "rep": "lm_head", "params": lm_head_params,
        "always_keep": False, "quantizable": True, "is_lm_head": True})

    norms = None
    if L and H:
        norms = 2 * L * H + H  # per-layer input+post-attn norms + final norm
    specs.append({
        "id": "norms", "label": "LayerNorms",
        "modules": ["input_layernorm", "post_attention_layernorm", "model.norm"],
        "rep": "model.layers.0.input_layernorm", "params": norms,
        "always_keep": True, "quantizable": False})
    return specs


def read_applied_quant(quant_dir: str | Path | None) -> dict[str, Any]:
    """Normalise the *exported* ``quantization_config`` (best-effort, offline).

    Handles both auto-round (``extra_config`` per-module overrides, where
    ``bits >= 16`` / ``data_type == "float"`` means a module kept high precision)
    and compressed-tensors (``ignore`` list + ``config_groups``).  Returns
    ``{}`` when no exported config is available.
    """
    if quant_dir is None:
        return {}
    cfg_path = Path(quant_dir) / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        qcfg = json.loads(cfg_path.read_text()).get("quantization_config")
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(qcfg, dict):
        return {}

    kept: set[str] = set()
    # compressed-tensors: explicit ignore list.
    for name in qcfg.get("ignore") or []:
        if isinstance(name, str) and name:
            kept.add(name)
    # auto-round: per-module overrides kept at >= 16 bits.
    extra = qcfg.get("extra_config")
    if isinstance(extra, dict):
        for name, ov in extra.items():
            if not isinstance(ov, dict):
                continue
            bits = ov.get("bits")
            dtype = str(ov.get("data_type", "")).lower()
            if (isinstance(bits, int) and bits >= 16) or dtype == "float":
                kept.add(name)
    default_bits = qcfg.get("bits")
    return {
        "quant_method": qcfg.get("quant_method"),
        "default_bits": default_bits if isinstance(default_bits, int) else None,
        "kept_modules": kept,
    }


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "scheme"


def _tensor_kept(name: str, shape: list[int], info: Any, kept_subs: set[str],
                 quant_lm_head: bool) -> bool:
    """Whether a real tensor stays high precision (BF16) rather than quantized.

    A tensor is quantized only if it is a quantizable Linear-like weight (a 2-D
    ``.weight`` or a fused 3-D MoE expert stack), its ``lm_head`` gate is open, and
    its module path is not in the exported/declared keep set.  Biases, norms, 1-D
    buffers and non-Linear weights (conv, SSM state) always stay high precision.
    """
    if not _is_quantizable_weight(name, shape, info):
        return True
    if info.is_lm_head and not quant_lm_head:
        return True
    module = name[:-len(".weight")] if name.endswith(".weight") else name
    return any(sub and sub in module for sub in kept_subs)


def _ground_truth_details(shapes: dict[str, list[int]], declared_scheme: str,
                          linear_op: str, kept_subs: set[str],
                          quant_lm_head: bool,
                          quantized_modules: set[str] | None = None,
                          layer_config: dict[str, Any] | None = None,
                          ) -> dict[str, Any] | None:
    """Per-module coverage from *real* tensor shapes (any architecture).

    Quantized-vs-kept is decided per tensor: authoritatively from the export's
    quantization artifacts (``quantized_modules``) when available, else from the
    exported/declared keep set.  The *applied scheme* among quantized tensors is
    resolved from the declared command — a ``layer_config`` key that matches the
    module path overrides the global ``--scheme`` (e.g. ``mlp.experts`` → MXFP4
    while the rest stays MXFP8).  Groups by ``(family, scheme)`` so a mixed family
    (hybrid linear-attention, or shared-expert gate-kept vs projections-quantized,
    or experts at a lower bit-width) splits into one group per distinct precision,
    each tagged with the command token (``matched_by``) that produced it.  Returns
    ``None`` if the shapes yield no parameters.
    """
    def _resolve(name: str, shape: list[int], info: Any
                 ) -> tuple[str, int | None, str | None, str]:
        if quantized_modules is not None:
            stem = name[:-len(".weight")] if name.endswith(".weight") else name
            quantized = stem in quantized_modules
        else:
            quantized = not _tensor_kept(name, shape, info, kept_subs, quant_lm_head)
        if not quantized:
            return "BF16", 16, None, _kept_reason(
                name, shape, info, kept_subs, quant_lm_head)
        module = name[:-len(".weight")] if name.endswith(".weight") else name
        return _declared_quantized_scheme(module, declared_scheme, layer_config)

    agg: dict[tuple[str, str], dict[str, Any]] = {}
    total = 0
    for name, shape in shapes.items():
        params = 1
        for x in shape:
            params *= int(x)
        total += params
        info = classify_module(name)
        scheme, bits, dtype, matched = _resolve(name, shape, info)
        key = (info.family, scheme)
        g = agg.setdefault(key, {
            "family": info.family, "label": info.label, "scheme": scheme,
            "bits": bits, "data_type": dtype, "params": 0, "tensors": 0,
            "_leaves": set(), "_layers": set(), "_matched": set()})
        g["params"] += params
        g["tensors"] += 1
        g["_leaves"].add(module_leaf(name))
        g["_matched"].add(matched)
        m = re.search(r"\.(?:layers|blocks|h)\.(\d+)\.", name)
        if m:
            g["_layers"].add(int(m.group(1)))
    if total <= 0:
        return None

    # A family split across >1 distinct scheme gets a disambiguating suffix.
    fam_scheme_count: dict[str, int] = {}
    for fam, _scheme in agg:
        fam_scheme_count[fam] = fam_scheme_count.get(fam, 0) + 1
    split_families = {fam for fam, c in fam_scheme_count.items() if c > 1}

    groups: list[dict[str, Any]] = []
    for (_fam, scheme), g in sorted(agg.items(), key=lambda kv: -kv[1]["params"]):
        quantized = scheme != "BF16"
        gid = g["family"]
        label = g["label"]
        if g["family"] in split_families:
            gid = f"{g['family']}-{_slug(scheme)}"
            label = f"{g['label']} ({scheme})"
        matched_by = (next(iter(g["_matched"])) if len(g["_matched"]) == 1
                      else ", ".join(sorted(g["_matched"])))
        entry: dict[str, Any] = {
            "id": gid, "label": label,
            "modules": sorted(g["_leaves"])[:8],
            "operation": f"{_slug(scheme)}-linear" if quantized else None,
            "scheme": scheme,
            "bits": g["bits"],
            "quantized": quantized,
            "matched_by": matched_by,
            "params": g["params"],
            "params_ratio": round(g["params"] / total, 4),
        }
        if g["data_type"]:
            entry["data_type"] = g["data_type"]
        if g["_layers"]:
            entry["layers"] = len(g["_layers"])
        groups.append(entry)

    quantized_params = sum(g["params"] for g in groups if g["scheme"] != "BF16")
    details: dict[str, Any] = {
        "derived_from": "weight_shapes",
        "params_basis": "safetensors_headers",
        "quantized_basis": ("export_artifacts" if quantized_modules is not None
                            else "declared_command"),
        "module_groups": groups,
        "total_params": total,
        "quantized_params": quantized_params,
        "quantized_ratio": round(quantized_params / total, 4),
    }
    weighted = 0.0
    bits_known = True
    for g in groups:
        b = _scheme_weight_bits(g["scheme"])
        if b is None:
            bits_known = False
            break
        weighted += g["params"] * b
    if bits_known:
        details["effective_bits"] = round(weighted / total, 3)
    return details


def compute_quantization_details(
    *,
    model_cfg: dict[str, Any],
    declared_scheme: str,
    quant_dir: str | Path | None = None,
    model_source: str | Path | None = None,
    declared_ignore: str | None = None,
    quant_lm_head: bool = False,
    layer_config: dict[str, Any] | None = None,
    kv_scheme: str | None = None,
    attn_scheme: str | None = None,
) -> dict[str, Any]:
    """Build the ``quantization_details`` block for one quantized variant.

    Prefers weight-shape ground truth (reads ``model_source``'s safetensors
    headers, exact for any architecture); falls back to the analytical dense/MoE
    estimate from ``model_cfg`` dims when weights are not local.  ``layer_config``
    (the declared per-module mixed-precision overrides) is honored in both paths
    so e.g. routed experts show up at ``MXFP4`` rather than the global scheme.

    The *same* function serves ANALYZE (predicted: no ``quant_dir`` → the plan
    from the declared command) and CLOSE (actual: the exported checkpoint drives
    quantized-vs-kept), so the two tables share a schema and are directly
    comparable.
    """
    applied = read_applied_quant(quant_dir)
    family = _slug(declared_scheme)
    linear_op = f"{family}-linear"

    kept_subs: set[str] = set(applied.get("kept_modules", set()))
    for tok in (declared_ignore or "").split(","):
        tok = tok.strip()
        if tok:
            kept_subs.add(tok)

    # Preferred path: real tensor shapes classify every module exactly.
    shapes = read_param_shapes(model_source)
    if shapes:
        quantized_modules = read_quantized_modules(quant_dir)
        gt = _ground_truth_details(shapes, declared_scheme, linear_op, kept_subs,
                                   quant_lm_head, quantized_modules, layer_config)
        # If the export's module names didn't line up with the base model's (no
        # overlap → nothing marked quantized), retry with the ignore heuristic.
        if (gt is not None and quantized_modules is not None
                and gt.get("quantized_params", 0) == 0):
            gt = _ground_truth_details(shapes, declared_scheme, linear_op,
                                       kept_subs, quant_lm_head, None, layer_config)
        if gt is not None:
            _append_runtime_schemes(gt, kv_scheme, attn_scheme)
            return gt

    d = _dims(model_cfg)
    groups: list[dict[str, Any]] = []
    total = 0
    total_known = True
    for spec in _group_specs(d):
        if spec.get("is_lm_head") and not quant_lm_head:
            scheme, bits, dtype, matched_by = "BF16", 16, None, (
                "lm_head kept (quant_lm_head=false)")
        elif spec["always_keep"]:
            scheme, bits, dtype, matched_by = "BF16", 16, None, (
                f"precision-critical ({spec['id']})")
        else:
            kept_sub = next((sub for sub in sorted(
                (s for s in kept_subs if s), key=len, reverse=True)
                if sub in spec["rep"]), None)
            if kept_sub:
                scheme, bits, dtype, matched_by = "BF16", 16, None, f"ignore:{kept_sub}"
            else:
                scheme, bits, dtype, matched_by = _declared_quantized_scheme(
                    spec["rep"], declared_scheme, layer_config)
        quantized = scheme != "BF16"
        g: dict[str, Any] = {
            "id": spec["id"], "label": spec["label"], "modules": spec["modules"],
            "operation": (f"{_slug(scheme)}-linear"
                          if (spec["quantizable"] and quantized) else None),
            "scheme": scheme, "bits": bits, "quantized": quantized,
            "matched_by": matched_by, "params": spec["params"],
        }
        if dtype:
            g["data_type"] = dtype
        if "layers" in spec:
            g["layers"] = spec["layers"]
        groups.append(g)
        if spec["params"] is None:
            total_known = False
        else:
            total += spec["params"]

    details: dict[str, Any] = {
        "derived_from": "exported_quant_config" if applied else "declared_scope",
        "module_groups": groups,
    }
    if total_known and total > 0:
        quantized = sum(g["params"] for g in groups if g["scheme"] != "BF16")
        details["total_params"] = total
        details["quantized_params"] = quantized
        details["quantized_ratio"] = round(quantized / total, 4)
        weighted = 0.0
        bits_known = True
        for g in groups:
            b = _scheme_weight_bits(g["scheme"])
            if b is None:
                bits_known = False
                break
            weighted += g["params"] * b
            g["params_ratio"] = round(g["params"] / total, 4)
        if bits_known:
            details["effective_bits"] = round(weighted / total, 3)
    else:
        details["note"] = ("parameter counts unavailable (config dims missing or "
                           "unsupported architecture); group coverage + schemes only")

    _append_runtime_schemes(details, kv_scheme, attn_scheme)
    return details


def _append_runtime_schemes(details: dict[str, Any], kv_scheme: str | None,
                            attn_scheme: str | None) -> None:
    """Attach optional static KV-cache / attention quant schemes to a block."""
    if kv_scheme:
        details["kv_cache"] = {"scheme": kv_scheme,
                               "element_format": "E4M3" if "8" in kv_scheme else None}
    if attn_scheme:
        details["attention"] = {"scheme": attn_scheme,
                                "element_format": "E4M3" if "8" in attn_scheme else None}

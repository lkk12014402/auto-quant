"""Canonical artifact naming — single source of truth for run/model identity.

Used by both the dataset sink artifact paths (``results/<org>/<artifact>/``)
and the HF model repo default name (``HfModelSink``), so a quantized model has
the same name everywhere.

Format::

    <ModelShort>-<Scheme>[-Mixed][-FP8KV][-FP8Attn][-CT][-RTN]-AutoRound

- ``Mixed``     — ``quant.layer_config`` present (per-layer mixed precision;
                  the leading scheme is the lowest-bit target).
- ``FP8KV`` / ``FP8Attn`` — ``static_kv_fp8`` / ``static_attention_fp8``.
- ``CT``        — ``export_format: llm_compressor`` (compressed-tensors pack).
- ``RTN``       — ``method: model_free`` or ``iters: 0`` (calibration-free /
                  RTN runs), placed right before ``AutoRound`` to match the
                  published convention (``...-W4A16-RTN-AutoRound``).
"""

from __future__ import annotations

from ..config import RunConfig


def _lowest_bit_scheme(quant) -> str:
    """Leading scheme = the lowest-bit target (Mixed 命名以低比特开头）。

    ``layer_config`` 里出现比主 scheme 更低比特的条目时（典型：MXFP8 主体 +
    路由专家 MXFP4），前导段取低比特侧（MXFP4）；无法映射时保守用主 scheme。
    """
    scheme = quant.scheme
    if not quant.layer_config:
        return scheme
    best_bits: int | None = None
    best_dtype = ""
    for entry in quant.layer_config.values():
        if not isinstance(entry, dict):
            continue
        b = entry.get("bits")
        if isinstance(b, int) and (best_bits is None or b < best_bits):
            best_bits, best_dtype = b, str(entry.get("data_type", ""))
    if best_bits is None:
        return scheme
    if best_dtype.startswith("mx_fp"):
        mapped = {4: "MXFP4", 8: "MXFP8"}.get(best_bits)
    else:  # int 系 weight-only
        mapped = {4: "W4A16", 8: "W8A16"}.get(best_bits)
    if not mapped:
        return scheme
    scheme_bits = 4 if "4" in scheme else 8 if "8" in scheme else None
    return mapped if (scheme_bits is None or best_bits < scheme_bits) else scheme


def scheme_display(quant) -> str:
    """Scheme label for human-facing surfaces (the HF model card).

    ``quant.scheme`` is the *base* recipe knob and under-describes a mixed artifact:
    ``scheme=MXFP8`` + ``layer_config={mlp.experts: 4 bit}`` ships routed experts at
    MXFP4, so a card reading "This is a MXFP8 quantization" mislabels the weights.
    Use the same effective-lowest-bit scheme :func:`canonical_artifact_name` leads
    with, plus the ``Mixed`` marker, so card and artifact/repo name agree.
    """
    label = _lowest_bit_scheme(quant)
    return f"{label}-Mixed" if quant.layer_config else label


def canonical_artifact_name(run_cfg: RunConfig) -> str:
    """Canonical name for a quantize run's artifact (dataset path + model repo)."""
    if run_cfg.output.artifact_name:
        return run_cfg.output.artifact_name
    ident = run_cfg.output.report_model or run_cfg.model
    short = ident.split("/")[-1] if ident else "model"
    quant = run_cfg.quant
    if quant is None:
        return f"{short}-BF16-Baseline"
    parts = [short, _lowest_bit_scheme(quant)]
    if quant.layer_config:
        parts.append("Mixed")
    if quant.static_kv_fp8:
        parts.append("FP8KV")
    if quant.static_attention_fp8:
        parts.append("FP8Attn")
    if quant.export_format == "llm_compressor":
        parts.append("CT")
    if quant.method == "model_free" or quant.iters == 0:
        parts.append("RTN")
    parts.append("AutoRound")
    return "-".join(parts)

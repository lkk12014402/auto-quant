"""Tests for the weight-header structure reader (RFC 0002, ``weight_map``) and the
weight-shape ground-truth path of ``quantization_details``."""

from __future__ import annotations

import numpy as np
from safetensors.numpy import save_file

from aqa.report.quant_details import compute_quantization_details
from aqa.report.weight_map import (
    classify_module,
    model_structure,
    read_param_shapes,
    read_quantized_modules,
    summarize_structure,
)


def _w(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float16)


def _v(n: int) -> np.ndarray:
    return np.zeros((n,), dtype=np.float16)


def _write_base_model(d) -> None:
    """A tiny *hybrid multimodal* base model: full-attn + linear-attn + MoE-free
    MLP + a vision tower + an MTP head + embeddings/lm_head/norms."""
    tensors = {
        "model.embed_tokens.weight": _w(100, 16),
        # layer 0: full attention
        "model.language_model.layers.0.self_attn.q_proj.weight": _w(16, 16),
        "model.language_model.layers.0.self_attn.o_proj.weight": _w(16, 16),
        "model.language_model.layers.0.self_attn.q_norm.weight": _v(16),
        "model.language_model.layers.0.mlp.down_proj.weight": _w(16, 32),
        "model.language_model.layers.0.input_layernorm.weight": _v(16),
        # layer 1: linear attention (SSM-style: some proj + non-Linear state)
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": _w(48, 16),
        "model.language_model.layers.1.linear_attn.out_proj.weight": _w(16, 48),
        "model.language_model.layers.1.linear_attn.in_proj_a.weight": _w(8, 16),
        "model.language_model.layers.1.linear_attn.conv1d.weight": _w(16, 4),
        "model.language_model.layers.1.linear_attn.A_log": _v(16),
        "model.language_model.layers.1.mlp.down_proj.weight": _w(16, 32),
        # vision tower
        "model.visual.blocks.0.attn.qkv.weight": _w(48, 16),
        "model.visual.merger.linear_fc1.weight": _w(16, 16),
        # MTP head (copied verbatim, never quantized)
        "mtp.layers.0.mlp.down_proj.weight": _w(16, 32),
        "mtp.norm.weight": _v(16),
        # tied-free lm head + final norm
        "lm_head.weight": _w(100, 16),
        "model.norm.weight": _v(16),
    }
    save_file(tensors, str(d / "model.safetensors"))


def _write_export(d) -> None:
    """An export that quantized only the LM backbone Linear weights: full-attn,
    linear-attn projections (not ``in_proj_a``), and MLP — NOT vision/MTP."""
    quantized_stems = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.1.mlp.down_proj",
    ]
    tensors: dict[str, np.ndarray] = {}
    for stem in quantized_stems:
        tensors[stem + ".weight_packed"] = np.zeros((4, 4), dtype=np.int32)
        tensors[stem + ".weight_scale"] = _v(4)
    # kept-BF16 modules survive as plain weights
    tensors["model.visual.blocks.0.attn.qkv.weight"] = _w(48, 16)
    tensors["mtp.layers.0.mlp.down_proj.weight"] = _w(16, 32)
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(
        '{"quantization_config": {"quant_method": "compressed-tensors", '
        '"config_groups": {"group_0": {"targets": ["Linear"], '
        '"weights": {"num_bits": 4}}}, "ignore": ["lm_head", '
        '"re:.*visual.*", "re:.*in_proj_a.*"]}}')


# ───────────────────────── classifier ─────────────────────────


def test_classify_module_families():
    assert classify_module("model.visual.blocks.0.attn.qkv.weight").family == "visual"
    assert classify_module("mtp.layers.0.mlp.down_proj.weight").family == "mtp"
    assert classify_module(
        "model.language_model.layers.1.linear_attn.out_proj.weight").family == "linear-attn"
    assert classify_module(
        "model.layers.0.self_attn.q_proj.weight").family == "self-attn"
    assert classify_module("model.layers.0.mlp.gate.weight").family == "router"
    assert classify_module("model.layers.0.mlp.gate_proj.weight").family == "mlp"
    assert classify_module("model.layers.0.mlp.experts.3.up_proj.weight").family == "moe-experts"
    assert classify_module("model.embed_tokens.weight").family == "embed"
    assert classify_module("lm_head.weight").family == "lm-head"
    assert classify_module("model.norm.weight").family == "norms"
    # lm_head and embed are not "quantizable families" by default coverage
    assert classify_module("lm_head.weight").is_lm_head is True
    assert classify_module("model.embed_tokens.weight").quantizable is False


# ───────────────────────── header reader ─────────────────────────


def test_read_param_shapes_and_structure(tmp_path):
    _write_base_model(tmp_path)
    shapes = read_param_shapes(tmp_path)
    assert shapes is not None
    assert shapes["lm_head.weight"] == [100, 16]
    s = summarize_structure(shapes)
    assert s["source"] == "weight_headers"
    fams = {f["id"]: f for f in s["families"]}
    # Hybrid + multimodal families must all surface (none dropped).
    assert {"self-attn", "linear-attn", "mlp", "visual", "mtp",
            "embed", "lm-head", "norms"} <= set(fams)
    # ratios sum ~1 and total matches the sum of tensors.
    assert abs(sum(f["params_ratio"] for f in s["families"]) - 1.0) < 1e-3
    assert s["total_params"] == sum(np.prod(v) for v in shapes.values())


def test_read_param_shapes_degrades_off_disk(tmp_path):
    # No safetensors present → None (remote/offline signal).
    assert read_param_shapes(tmp_path) is None
    assert read_param_shapes("Qwen/Qwen3-8B") is None
    assert read_param_shapes(None) is None
    assert model_structure(tmp_path) is None


def test_read_quantized_modules(tmp_path):
    _write_export(tmp_path)
    q = read_quantized_modules(tmp_path)
    assert q is not None
    assert "model.language_model.layers.0.self_attn.q_proj" in q
    assert "model.language_model.layers.1.linear_attn.out_proj" in q
    # kept-BF16 modules carry no quant artifact
    assert "mtp.layers.0.mlp.down_proj" not in q
    assert "model.visual.blocks.0.attn.qkv" not in q


# ───────────────────── ground-truth quant_details ─────────────────────


def test_ground_truth_uses_export_artifacts(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    exp = tmp_path / "export"; exp.mkdir()
    _write_base_model(base)
    _write_export(exp)
    d = compute_quantization_details(
        model_cfg={}, declared_scheme="W4A16", quant_dir=exp, model_source=base,
        declared_ignore="lm_head,visual")
    assert d["derived_from"] == "weight_shapes"
    assert d["quantized_basis"] == "export_artifacts"
    by = {g["id"]: g for g in d["module_groups"]}
    # MTP + vision + embed + lm_head are kept BF16 (never in the quant graph).
    assert by["mtp"]["scheme"] == "BF16"
    assert by["visual"]["scheme"] == "BF16"
    assert by["embed"]["scheme"] == "BF16"
    assert by["lm-head"]["scheme"] == "BF16"
    # Backbone Linear weights are quantized.
    assert by["mlp"]["scheme"] == "W4A16" and by["mlp"]["operation"] == "w4a16-linear"
    # linear-attn is mixed → split into one group per scheme (scheme-suffixed id).
    assert "linear-attn-w4a16" in by and "linear-attn-bf16" in by
    assert by["linear-attn-w4a16"]["scheme"] == "W4A16"
    assert by["linear-attn-bf16"]["scheme"] == "BF16"
    # Totals are exact and consistent.
    assert d["total_params"] == sum(
        int(np.prod(v)) for v in read_param_shapes(base).values())
    assert 0.0 < d["quantized_ratio"] < 1.0
    assert 4.0 < d["effective_bits"] < 16.0
    assert abs(sum(g["params_ratio"] for g in d["module_groups"]) - 1.0) < 1e-3


def test_ground_truth_falls_back_to_analytical_without_weights():
    # No model_source → analytical path (config dims), unchanged behavior.
    d = compute_quantization_details(
        model_cfg=dict(hidden_size=512, num_hidden_layers=4, num_attention_heads=8,
                       num_key_value_heads=8, intermediate_size=1024,
                       vocab_size=1000, tie_word_embeddings=False),
        declared_scheme="W4A16", declared_ignore="lm_head")
    assert d["derived_from"] in ("declared_scope", "exported_quant_config")
    assert "params_basis" not in d


def test_ground_truth_without_export_uses_ignore_heuristic(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    _write_base_model(base)
    # No quant_dir → no export artifacts → declared-command basis.
    d = compute_quantization_details(
        model_cfg={}, declared_scheme="W4A16", model_source=base,
        declared_ignore="lm_head,visual,in_proj_a")
    assert d["derived_from"] == "weight_shapes"
    assert d["quantized_basis"] == "declared_command"
    by = {g["id"]: g for g in d["module_groups"]}
    assert by["visual"]["scheme"] == "BF16"


def test_fused_3d_experts_quantize_via_layer_config(tmp_path):
    """Fused MoE experts (3-D ``[E,in,out]``, no ``.weight``) must be quantized —
    and honor a ``layer_config`` override — not treated as non-Linear BF16."""
    base = tmp_path / "base"; base.mkdir()
    save_file({
        "model.embed_tokens.weight": _w(100, 16),
        # Fused experts: 3-D stacks with NO `.weight` suffix.
        "model.layers.0.mlp.experts.gate_up_proj": np.zeros((8, 16, 32), np.float16),
        "model.layers.0.mlp.experts.down_proj": np.zeros((8, 16, 16), np.float16),
        # A conv (3-D ``[C,1,k]``) in a non-experts family must stay BF16.
        "model.layers.0.linear_attn.conv1d.weight": np.zeros((16, 1, 4), np.float16),
        "model.layers.0.self_attn.q_proj.weight": _w(16, 16),
        "lm_head.weight": _w(100, 16),
        "model.norm.weight": _v(16),
    }, str(base / "model.safetensors"))
    d = compute_quantization_details(
        model_cfg={}, declared_scheme="MXFP8", model_source=base,
        declared_ignore="lm_head",
        layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}})
    by = {g["id"]: g for g in d["module_groups"]}
    assert by["moe-experts"]["scheme"] == "MXFP4"
    assert by["moe-experts"]["matched_by"] == "layer_config:mlp.experts"
    assert by["moe-experts"]["quantized"] is True
    assert by["self-attn"]["scheme"] == "MXFP8"
    # conv1d is 3-D but not an expert stack -> kept BF16.
    assert by["linear-attn"]["scheme"] == "BF16"
    # Experts dominate the Linear budget -> quantized ratio is substantial.
    assert d["quantized_ratio"] > 0.6


def test_exclude_layers_lists_quantizable_kept_families():
    """`exclude_layers` must list quantization-candidate families kept fully BF16
    (visual/mtp/lm-head), skip always-keep families (embed/norms), and skip the
    `-bf16` remainder of a partially-quantized family."""
    from aqa.report.recipe_manifest import _exclude_layers
    details = {"module_groups": [
        {"id": "mlp", "scheme": "W4A16"},
        {"id": "self-attn-quant", "scheme": "W4A16"},
        {"id": "linear-attn-quant", "scheme": "W4A16"},
        {"id": "linear-attn-bf16", "scheme": "BF16"},   # partial keep → skip
        {"id": "visual", "scheme": "BF16"},              # candidate kept → list
        {"id": "mtp", "scheme": "BF16"},                 # candidate kept → list
        {"id": "lm-head", "scheme": "BF16"},             # candidate kept → list
        {"id": "embed", "scheme": "BF16"},               # always-keep → skip
        {"id": "norms", "scheme": "BF16"},               # always-keep → skip
    ]}
    assert _exclude_layers(details) == ["visual", "mtp", "lm-head"]


def test_run_log_tee_writes_pipeline_log(tmp_path):
    import logging

    from aqa.logging_util import attach_run_log, detach_run_log, get_logger
    log = get_logger("pipeline")
    before = list(logging.getLogger("aqa").handlers)
    h = attach_run_log(tmp_path)
    log.info("== enter QUANT ==")
    detach_run_log(h)
    p = tmp_path / "logs" / "pipeline.log"
    assert p.is_file()
    assert "== enter QUANT ==" in p.read_text()
    # handler is removed cleanly (no leak across runs)
    assert list(logging.getLogger("aqa").handlers) == before


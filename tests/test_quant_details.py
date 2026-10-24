"""Tests for the ``quantization_details`` calculator (RFC 0002)."""

from __future__ import annotations

from aqa.report.quant_details import (
    _scheme_weight_bits,
    compute_quantization_details,
    read_applied_quant,
)

DENSE_CFG = dict(
    hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
    num_key_value_heads=8, intermediate_size=28672, vocab_size=128256,
    tie_word_embeddings=False)

MOE_CFG = dict(
    hidden_size=4096, num_hidden_layers=61, num_attention_heads=32,
    num_key_value_heads=8, n_routed_experts=256, moe_intermediate_size=2048,
    n_shared_experts=1, vocab_size=129280, intermediate_size=18432,
    tie_word_embeddings=False)


def test_scheme_weight_bits():
    assert _scheme_weight_bits("MXFP8") == 8
    assert _scheme_weight_bits("MXFP4") == 4
    assert _scheme_weight_bits("W4A16") == 4
    assert _scheme_weight_bits("BF16") == 16
    assert _scheme_weight_bits("totally-unknown") is None
    assert _scheme_weight_bits(None) is None


def test_dense_coverage_and_ratios():
    d = compute_quantization_details(
        model_cfg=DENSE_CFG, declared_scheme="MXFP8", declared_ignore="lm_head")
    ids = {g["id"] for g in d["module_groups"]}
    # Comprehensive coverage: every dense family present.
    assert {"embed", "attn-proj", "mlp", "lm-head", "norms"} <= ids
    # lm_head ignored -> kept BF16; attn/mlp quantized to the declared scheme.
    by = {g["id"]: g for g in d["module_groups"]}
    assert by["lm-head"]["scheme"] == "BF16"
    assert by["attn-proj"]["scheme"] == "MXFP8"
    assert by["attn-proj"]["operation"] == "mxfp8-linear"
    assert by["embed"]["scheme"] == "BF16" and by["embed"]["operation"] is None
    # Param ratios sum to 1 and effective_bits is between 4 and 16.
    total_ratio = sum(g.get("params_ratio", 0.0) for g in d["module_groups"])
    assert abs(total_ratio - 1.0) < 1e-3
    assert 7.0 < d["effective_bits"] <= 8.5
    assert 0.0 < d["quantized_ratio"] <= 1.0
    assert d["derived_from"] == "declared_scope"


def test_moe_coverage_router_kept():
    d = compute_quantization_details(
        model_cfg=MOE_CFG, declared_scheme="MXFP4",
        declared_ignore="lm_head,mlp.gate")
    by = {g["id"]: g for g in d["module_groups"]}
    assert {"moe-experts", "shared-expert", "router"} <= set(by)
    # Router (gate) is always kept; experts are quantized.
    assert by["router"]["scheme"] == "BF16"
    assert by["moe-experts"]["scheme"] == "MXFP4"
    # Experts dominate the budget -> effective bits close to 4.
    assert 4.0 <= d["effective_bits"] < 4.5
    total_ratio = sum(g.get("params_ratio", 0.0) for g in d["module_groups"])
    assert abs(total_ratio - 1.0) < 1e-3


def test_layer_config_overrides_experts_to_mxfp4():
    """`layer_config` pushes routed experts to a lower bit-width than the scheme."""
    d = compute_quantization_details(
        model_cfg=MOE_CFG, declared_scheme="MXFP8",
        layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}})
    by = {g["id"]: g for g in d["module_groups"]}
    # Experts matched by the override -> MXFP4; everything else -> global MXFP8.
    assert by["moe-experts"]["scheme"] == "MXFP4"
    assert by["moe-experts"]["bits"] == 4
    assert by["moe-experts"]["matched_by"] == "layer_config:mlp.experts"
    assert by["moe-experts"]["operation"] == "mxfp4-linear"
    assert by["shared-expert"]["scheme"] == "MXFP8"
    assert by["shared-expert"]["matched_by"] == "scheme"
    # Experts dominate -> effective bits sit well below the global 8.
    assert d["effective_bits"] < 5.0


def test_quant_lm_head_flips_scheme():
    d = compute_quantization_details(
        model_cfg=DENSE_CFG, declared_scheme="MXFP8", quant_lm_head=True)
    by = {g["id"]: g for g in d["module_groups"]}
    assert by["lm-head"]["scheme"] == "MXFP8"


def test_unknown_arch_degrades_without_params():
    d = compute_quantization_details(model_cfg={}, declared_scheme="MXFP4")
    # Still comprehensive coverage + schemes, but params are null.
    assert d["module_groups"]
    assert "total_params" not in d
    assert "note" in d
    for g in d["module_groups"]:
        assert g["params"] is None


def test_read_applied_quant_auto_round(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "auto-round", "bits": 4, '
        '"extra_config": {"lm_head": {"bits": 16, "data_type": "float"}, '
        '"model.layers.0.self_attn.k_proj": {"bits": 4, "data_type": "mx_fp"}}}}')
    applied = read_applied_quant(tmp_path)
    assert applied["quant_method"] == "auto-round"
    assert "lm_head" in applied["kept_modules"]
    assert "model.layers.0.self_attn.k_proj" not in applied["kept_modules"]


def test_read_applied_quant_compressed_tensors(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "compressed-tensors", '
        '"ignore": ["lm_head", "re:.*mlp.gate$"], "config_groups": {}}}')
    applied = read_applied_quant(tmp_path)
    assert applied["quant_method"] == "compressed-tensors"
    assert "lm_head" in applied["kept_modules"]


def test_exported_config_drives_derived_from(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "auto-round", "bits": 8, '
        '"extra_config": {"lm_head": {"bits": 16, "data_type": "float"}}}}')
    d = compute_quantization_details(
        model_cfg=DENSE_CFG, declared_scheme="MXFP8", quant_dir=tmp_path)
    by = {g["id"]: g for g in d["module_groups"]}
    assert d["derived_from"] == "exported_quant_config"
    # lm_head kept high precision because the exported config said so.
    assert by["lm-head"]["scheme"] == "BF16"

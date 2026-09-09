"""Canonical artifact naming matrix."""

from aqa.config import parse_run_config
from aqa.report.names import canonical_artifact_name


def _name(**quant):
    return canonical_artifact_name(
        parse_run_config({"model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                          "quant": quant}))


def test_pure_schemes():
    assert _name(scheme="MXFP4", method="model_free") == \
        "Qwen3-30B-A3B-Instruct-2507-MXFP4-CT-RTN-AutoRound"  # 默认 format 即 llm_compressor
    assert _name(scheme="MXFP4", method="model_free", export_format="auto_round") == \
        "Qwen3-30B-A3B-Instruct-2507-MXFP4-RTN-AutoRound"
    assert _name(scheme="W4A16", method="autoround", iters=200) == \
        "Qwen3-30B-A3B-Instruct-2507-W4A16-CT-AutoRound"
    # iters=0 也是 RTN 语义
    assert _name(scheme="W4A16", method="autoround", iters=0) == \
        "Qwen3-30B-A3B-Instruct-2507-W4A16-CT-RTN-AutoRound"


def test_mixed_and_fp8_variants():
    assert _name(scheme="MXFP8", method="model_free",
                 layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}},
                 export_format="llm_compressor") == \
        "Qwen3-30B-A3B-Instruct-2507-MXFP4-Mixed-CT-RTN-AutoRound"
    assert _name(scheme="MXFP8", method="autoround", iters=0,
                 layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}},
                 static_kv_fp8=True, export_format="llm_compressor") == \
        "Qwen3-30B-A3B-Instruct-2507-MXFP4-Mixed-FP8KV-CT-RTN-AutoRound"
    assert _name(scheme="MXFP8", method="autoround", iters=0,
                 layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}},
                 static_kv_fp8=True, static_attention_fp8=True,
                 export_format="llm_compressor") == \
        "Qwen3-30B-A3B-Instruct-2507-MXFP4-Mixed-FP8KV-FP8Attn-CT-RTN-AutoRound"


def test_baseline_and_glm_examples():
    from aqa.pipeline import _artifact_name
    cfg = parse_run_config({"model": "zai-org/GLM-5.3-Flash",
                            "quant": {"scheme": "W4A16", "method": "autoround",
                                      "iters": 0}})
    assert _artifact_name(cfg, "quant") == "GLM-5.3-Flash-W4A16-CT-RTN-AutoRound"
    assert _artifact_name(cfg, "baseline") == "GLM-5.3-Flash-BF16-Baseline"
    assert _artifact_name(cfg, "sweep") == "GLM-5.3-Flash-AutoRound-sweep"

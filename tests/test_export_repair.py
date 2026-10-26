"""Post-export diffusion repair tests (no GPU / real models needed).

auto-round re-saves non-quantized pipeline components itself and can mangle
their weight names (Z-Image's Qwen3 text encoder loses the inner ``model.``
prefix, which breaks vllm-omni's loader).  ``repair_diffusion_export`` must
restore exactly those components from the source model and never touch the
quantized transformer.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from aqa.quantize import repair_diffusion_export, repair_glm5_vllm


def _write_st(dirpath: Path, keys: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text("{}")
    tensors = {k: np.zeros((2, 2), dtype=np.float32) for k in keys}
    save_file(tensors, str(dirpath / "model.safetensors"))


def _read_keys(dirpath: Path) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for shard in sorted(dirpath.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as fh:
            keys.update(fh.keys())
    return keys


def _make_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "transformer").mkdir(parents=True)
    (src / "model_index.json").write_text(json.dumps({"_class_name": "FakePipeline"}))
    _write_st(src / "transformer", ["layers.0.to_q.weight"])
    _write_st(src / "text_encoder", ["model.embed_tokens.weight", "model.layers.0.mlp.weight"])
    _write_st(src / "vae", ["encoder.conv.weight"])
    return src


def test_repairs_mangled_text_encoder_and_keeps_transformer(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    # export: transformer quantized (different keys), text_encoder mangled
    # (`model.` prefix stripped -- the Z-Image failure), vae intact.
    _write_st(out / "transformer", ["layers.0.to_q.weight_packed"])
    _write_st(out / "text_encoder", ["embed_tokens.weight", "layers.0.mlp.weight"])
    _write_st(out / "vae", ["encoder.conv.weight"])

    repaired = repair_diffusion_export(str(src), out)

    assert repaired == ["text_encoder"]
    assert _read_keys(out / "text_encoder") == _read_keys(src / "text_encoder")
    # quantized component must NOT be restored from source
    assert _read_keys(out / "transformer") == {"layers.0.to_q.weight_packed"}
    # matching component untouched (same file object content, no rewrite needed)
    assert _read_keys(out / "vae") == {"encoder.conv.weight"}
    report = json.loads((out / "aqa_export_repair.json").read_text())
    assert report["repaired_components"] == ["text_encoder"]


def test_clean_export_is_untouched(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _write_st(out / "transformer", ["layers.0.to_q.weight_packed"])
    _write_st(out / "text_encoder", ["model.embed_tokens.weight", "model.layers.0.mlp.weight"])
    _write_st(out / "vae", ["encoder.conv.weight"])

    assert repair_diffusion_export(str(src), out) == []
    assert not (out / "aqa_export_repair.json").exists()


def test_missing_component_dir_is_restored(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _write_st(out / "transformer", ["layers.0.to_q.weight_packed"])
    # text_encoder absent entirely from the export

    repaired = repair_diffusion_export(str(src), out)
    assert "text_encoder" in repaired
    assert _read_keys(out / "text_encoder") == _read_keys(src / "text_encoder")


def test_non_diffusion_model_is_noop(tmp_path):
    src = tmp_path / "llm"  # no model_index.json
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    assert repair_diffusion_export(str(src), out) == []


# ── config.json structure restore (LLM tuning-path exports) ─────────────────
from aqa.quantize import restore_config_structure  # noqa: E402


def _make_llm_src(tmp_path: Path) -> Path:
    src = tmp_path / "src_llm"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "text_config": {"layer_types": ["linear_attention", "full_attention"]},
    }))
    return src


def test_restores_drifted_config_in_nested_export(tmp_path):
    """auto-round re-serialization renames layer_types; repair restores source structure."""
    src = _make_llm_src(tmp_path)
    out = tmp_path / "quantized"
    nested = out / "model-w4g128"  # tuning path nests one level
    nested.mkdir(parents=True)
    (nested / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "text_config": {"layer_types": ["linear_attention", "qwen_sparse_attention"]},
        "quantization_config": {"quant_method": "auto-round", "bits": 4},
    }))

    assert restore_config_structure(str(src), out) is True
    repaired = json.loads((nested / "config.json").read_text())
    assert repaired["text_config"]["layer_types"] == ["linear_attention", "full_attention"]
    assert repaired["quantization_config"]["quant_method"] == "auto-round"
    # pre-repair copy preserved
    backup = json.loads((nested / "config.json.autoround.bak").read_text())
    assert backup["text_config"]["layer_types"][1] == "qwen_sparse_attention"


def test_config_restore_noop_when_only_quant_config_differs(tmp_path):
    src = _make_llm_src(tmp_path)
    out = tmp_path / "quantized"
    out.mkdir()
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "auto-round"}
    (out / "config.json").write_text(json.dumps(cfg))

    assert restore_config_structure(str(src), out) is False
    assert not (out / "config.json.autoround.bak").exists()


def test_config_restore_skips_diffusion_and_missing_quant_config(tmp_path):
    src = _make_llm_src(tmp_path)
    (src / "model_index.json").write_text("{}")  # diffusion pipeline marker
    out = tmp_path / "quantized"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"drifted": True}))
    assert restore_config_structure(str(src), out) is False

    src2 = _make_llm_src(tmp_path / "second")
    (out / "config.json").write_text(json.dumps({"drifted": True}))
    assert restore_config_structure(str(src2), out) is False  # no quantization_config
    assert json.loads((out / "config.json").read_text()) == {"drifted": True}



def test_repair_fused_name_gaps(tmp_path):
    """extra_config 里 gate_proj+up_proj 成对存在时，补融合名 gate_up_proj。"""
    from aqa.quantize import repair_fused_name_gaps

    out = tmp_path / "quantized"
    out.mkdir()
    cfg = {
        "quantization_config": {
            "quant_method": "auto-round",
            "extra_config": {
                "model.layers.0.mlp.gate_proj": {"bits": 16, "data_type": "float"},
                "model.layers.0.mlp.up_proj": {"bits": 16, "data_type": "float"},
                "model.layers.0.mlp.down_proj": {"bits": 16, "data_type": "float"},
                "model.layers.1.mlp.shared_experts.gate_proj": {"bits": 16, "data_type": "float"},
                "model.layers.1.mlp.shared_experts.up_proj": {"bits": 16, "data_type": "float"},
            },
        }
    }
    (out / "config.json").write_text(json.dumps(cfg))

    added = repair_fused_name_gaps(out)
    assert set(added) == {
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.1.mlp.shared_experts.gate_up_proj",
    }
    ec = json.loads((out / "config.json").read_text())["quantization_config"]["extra_config"]
    assert ec["model.layers.0.mlp.gate_up_proj"] == {"bits": 16, "data_type": "float"}
    # down_proj 不成对，不补
    assert "model.layers.0.mlp.gate_up_proj.weight" not in ec


def test_repair_fused_name_gaps_noop(tmp_path):
    from aqa.quantize import repair_fused_name_gaps

    out = tmp_path / "quantized"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"quantization_config": {"extra_config": {}}}))
    assert repair_fused_name_gaps(out) == []
    # 无 config / 无 extra_config 也安全
    (out / "config.json").write_text(json.dumps({"a": 1}))
    assert repair_fused_name_gaps(out) == []


def test_glm5_vllm_rewrites_targets_to_runtime_namespace(tmp_path):
    out = tmp_path / "quantized"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({
        "model_type": "glm5_next",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "ignore": ["model.language_model.layers.45.mlp.experts"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float",
                                "group_size": 32, "strategy": "group"},
                }
            },
        },
    }))
    (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.layers.0.mlp.gate_up_proj.weight_packed": "x",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed": "x",
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight": "y",
    }}))

    info = repair_glm5_vllm(out)

    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "mxfp4-pack-quantized"
    assert qc["ignore"] == []
    targets = qc["config_groups"]["group_0"]["targets"]
    assert "language_model.model.layers.0.mlp.gate_up_proj" in targets
    assert "language_model.model.layers.3.mlp.experts.0.gate_proj" in targets
    assert all("layers.45" not in target for target in targets)
    assert info["kind"] == "glm5_vllm" and info["targets"] == 2
    assert (out / ".pre_export_repair_config.json").is_file()
    assert json.loads((out / "quantization_config.json").read_text())["ignore"] == []

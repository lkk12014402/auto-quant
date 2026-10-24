"""Tests for the deterministic quantized-model card (hard-flow step ④)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aqa.config import (
    EvalConfig,
    ModelUploadConfig,
    OutputConfig,
    ProviderConfig,
    QuantConfig,
    RunConfig,
)
from aqa.logging_util import AqaError
from aqa.report.model_card import (
    CARD_FILE,
    render_model_card,
    signoff_problem,
    write_model_card,
)

_ARCH = dict(
    hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, intermediate_size=14336, vocab_size=152064,
    tie_word_embeddings=False, architectures=["LlamaForCausalLM"],
    model_type="llama")


def _cfg(tmp_path: Path, **over):
    base = tmp_path / "base-model"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text(json.dumps(_ARCH))
    quant = QuantConfig(scheme="MXFP8", method="model_free", iters=0,
                        ignore_layers="lm_head")
    upload = ModelUploadConfig(
        repo="intel-ai", repo_name="Base-MXFP8-CT-RTN-AutoRound",
        signoff="Jane Smith <jane.smith@intel.com>")
    cfg = RunConfig(
        model=str(base),
        provider=ProviderConfig(type="quantize", quant=quant, upload=upload),
        eval=EvalConfig(backend="vllm"),
        output=OutputConfig(dir=str(tmp_path / "out"), report_model="org/Base"),
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _seed_run(tmp_path: Path, cfg: RunConfig) -> Path:
    out = tmp_path / "run"
    (out / "eval").mkdir(parents=True)
    (out / "baseline").mkdir()
    (out / "quantized").mkdir()
    (out / "quantized" / "config.json").write_text(json.dumps({
        **_ARCH,
        "quantization_config": {
            "quant_method": "auto-round", "bits": 8,
            "extra_config": {"lm_head": {"bits": 16, "data_type": "float"}},
        },
    }))
    (out / "quant_plan.json").write_text(json.dumps({
        "command": ("auto-round --model_name " + cfg.model +
                    " --scheme MXFP8 --iters 0 --device_map auto "
                    "--output_dir " + str(tmp_path / "quantized"))}))
    (out / "run_manifest.json").write_text(json.dumps({
        "environment": {"packages": {
            "vllm": "0.29.0", "auto-round": "0.15.0",
            "transformers": "5.17.0"}},
        "commands": {
            "quantize": ("auto-round --model_name " + cfg.model +
                         " --scheme MXFP8 --iters 0 --device_map auto "
                         "--output_dir " + str(tmp_path / "quantized")),
            "evaluate": [
                ("lm_eval --model vllm --model_args pretrained="
                 + str(out / "quantized") + ",tensor_parallel_size=1 "
                 "--tasks piqa,mmlu"),
            ],
        },
    }))
    (out / "eval" / "accuracy.json").write_text(json.dumps({
        "model": "quantized", "backend": "vllm",
        "results": {
            "piqa": {"score": 0.82, "metric": "acc", "num_samples": 100},
            "mmlu": {"score": 0.79, "metric": "acc", "num_samples": 100},
        },
        "commands": ["lm_eval ..."],
    }))
    (out / "baseline" / "accuracy.json").write_text(json.dumps({
        "model": "bf16", "backend": "vllm",
        "results": {
            "piqa": {"score": 0.84, "metric": "acc", "num_samples": 100},
            "mmlu": {"score": 0.81, "metric": "acc", "num_samples": 100},
        },
        "commands": ["lm_eval ..."],
    }))
    return out


def test_render_card_follows_template_contract(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = _seed_run(tmp_path, cfg)
    card = render_model_card(
        cfg, out_dir=out, repo_name="Base-MXFP8-CT-RTN-AutoRound",
        quant_dir=out / "quantized", signoff="Jane Smith <jane.smith@intel.com>")

    assert card.startswith("---\n")
    assert "<!--" not in card  # R5: every template comment is deleted
    assert card.rstrip().endswith(
        "Signed-off-by: Jane Smith <jane.smith@intel.com>")
    assert "base_model:\n- org/Base" in card
    assert "pipeline_tag: text-generation" in card

    headings = re.findall(r"^#{1,3} .+$", card, re.MULTILINE)
    assert headings == [
        "## Model Details",
        "## How to Use",
        "### Environment",
        "### vLLM Usage",
        "## Evaluation",
        "## Generate the Model",
        "## Ethical Considerations and Limitations",
        "## Caveats and Recommendations",
        "## Disclaimer",
        "## Cite",
    ]
    # Evaluation table: general rows + average, raw scores x100, recovery.
    assert "| piqa | 84.00 | 82.00 | 97.62% |" in card
    assert "**Average**" in card
    # One deployment command and one quantization command (R2).
    assert card.count("vllm serve ") == 1
    assert card.count("auto-round \\") == 1
    # Local paths and device_map never leak into the card.
    assert str(tmp_path) not in card
    assert "--device_map" not in card
    # Sign-off helper agrees.
    assert signoff_problem(card) is None


def test_local_base_readme_metadata_is_used(tmp_path: Path):
    cfg = _cfg(tmp_path)
    (tmp_path / "base-model" / "README.md").write_text(
        "---\nlicense: apache-2.0\npipeline_tag: image-text-to-text\n---\n")
    out = _seed_run(tmp_path, cfg)
    card = render_model_card(
        cfg, out_dir=out, repo_name="Base-MXFP8-CT-RTN-AutoRound",
        quant_dir=out / "quantized", signoff="Jane Smith <jane.smith@intel.com>")
    assert "license: apache-2.0" in card
    assert "pipeline_tag: image-text-to-text" in card


def test_draft_card_when_signoff_and_repo_id_missing(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.output.report_model = None
    cfg.provider = cfg.provider.model_copy(update={"upload": None})
    out = _seed_run(tmp_path, cfg)
    card = render_model_card(
        cfg, out_dir=out, repo_name="Base-MXFP8-CT-RTN-AutoRound",
        quant_dir=out / "quantized", signoff=None)
    assert card.rstrip().endswith(
        "Signed-off-by: {member name} <{member email}>")
    assert "base_model:\n- {base_repo_id}" in card
    assert "vllm serve {org}/Base-MXFP8-CT-RTN-AutoRound" in card
    assert "| piqa |" in card and "| mmlu |" in card
    assert str(tmp_path) not in card
    assert signoff_problem(card) is not None  # draft is not publishable


def test_invalid_explicit_signoff_is_rejected(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = _seed_run(tmp_path, cfg)
    with pytest.raises(AqaError, match="末行须为"):
        render_model_card(cfg, out_dir=out, repo_name="X",
                          quant_dir=out / "quantized",
                          signoff="Jane Smith")
    with pytest.raises(AqaError, match="占位身份"):
        render_model_card(cfg, out_dir=out, repo_name="X",
                          quant_dir=out / "quantized",
                          signoff="Jane <jane@example.com>")


def test_write_card_atomically(tmp_path: Path):
    cfg = _cfg(tmp_path)
    out = _seed_run(tmp_path, cfg)
    path = write_model_card(
        cfg, out_dir=out, repo_name="Base-MXFP8-CT-RTN-AutoRound",
        quant_dir=out / "quantized", signoff="Jane Smith <jane.smith@intel.com>")
    assert path == out / CARD_FILE
    assert path.is_file()
    assert not (out / "model_card.md.tmp").exists()
    assert signoff_problem(path.read_text()) is None

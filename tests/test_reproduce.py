"""Tests for the reproduce adapter + three-sided verification (RFC 0002 §5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aqa.config import RunConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.logging_util import AqaError
from aqa.reproduce import (
    _actual_quant_details,
    build_reproduction_config,
    expected_results,
    load_manifest,
    plan_reproduction,
    reproduce_manifest,
    resolve_variant,
    verify_reproduction,
)

_ARCH = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, intermediate_size=128, vocab_size=128,
    tie_word_embeddings=False, architectures=["LlamaForCausalLM"],
    model_type="llama")


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_ARCH))
    return model


def _manifest(tmp_path: Path, *, model: Path | None = None) -> dict:
    model = model or _model_dir(tmp_path)
    return {
        "schema_version": "1.0",
        "model": {"id": "tiny", "name": "Tiny", "source": str(model)},
        "environments": {
            "quantization": {"packages": ["auto-round==0.15.0"]},
            "evaluation": {"packages": ["vllm==0.27.0", "lm-eval==0.4.12"]},
        },
        "operations": {
            "mxfp4-linear": {
                "label": "Linear",
                "settings": [
                    {"key": "data_type", "value": "MXFP4"},
                    {"key": "exclude_layers", "value": ["lm_head"]},
                ],
            }
        },
        "variants": [{
            "id": "mxfp4-v1", "family": "mxfp4", "label": "V1 — Linear",
            "recipe": {"approach": "PTQ", "algorithm": "RTN",
                       "operations": ["mxfp4-linear"]},
            "quantization": {
                "environment": "quantization", "shell": "bash",
                "variables": {
                    "MODEL": {"kind": "input_model", "from": "model.source"},
                    "OUTPUT_DIR": {"kind": "output_dir", "default": "quantized"},
                },
                "script": ('auto-round --model_name "${MODEL}" --scheme MXFP4 '
                           '--iters 0 --ignore_layers lm_head '
                           '--output_dir "${OUTPUT_DIR}"'),
            },
            "evaluation": {
                "environment": "evaluation", "shell": "bash",
                "variables": {
                    "MODEL_PATH": {"kind": "checkpoint",
                                   "from": "quantization.output_dir",
                                   "default": "quantized"},
                },
                "script": ('lm_eval --model vllm --model_args '
                           '"pretrained=${MODEL_PATH},tensor_parallel_size=2,'
                           'dtype=bfloat16,enable_thinking=False" '
                           '--tasks piqa,mmlu --batch_size auto --seed 42'),
            },
        }],
        "result_tables": [{
            "id": "mxfp4-accuracy", "title": "Accuracy", "notes": "n",
            "columns": [
                {"id": "piqa", "label": "piqa", "kind": "metric",
                 "task": "piqa", "unit": "fraction"},
                {"id": "mmlu", "label": "mmlu", "kind": "metric",
                 "task": "mmlu", "unit": "fraction"},
            ],
            "rows": [{
                "id": "mxfp4-v1-result", "label": "MXFP4",
                "kind": "variant", "variant": "mxfp4-v1",
                "values": {"piqa": 0.80, "mmlu": 0.79},
            }],
        }],
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    import yaml

    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _seed_run(tmp_path: Path, manifest: dict, cfg: RunConfig) -> Path:
    out = tmp_path / "run"
    (out / "eval").mkdir(parents=True)
    (out / "quantized").mkdir()
    quant_cfg = dict(_ARCH)
    quant_cfg["quantization_config"] = {
        "quant_method": "auto-round", "bits": 4,
        "extra_config": {"lm_head": {"bits": 16, "data_type": "float"}}}
    (out / "quantized" / "config.json").write_text(json.dumps(quant_cfg))
    (out / "run_manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (out / "env_lock.json").write_text(json.dumps({
        "packages": {"auto-round": "0.15.0", "vllm": "0.27.0",
                     "lm-eval": "0.4.12"}}))
    (out / ".quant_command.json").write_text(json.dumps({
        "command": cfg.quant.command_override, "source": "reproduce"}))
    (out / "eval" / "accuracy.json").write_text(json.dumps({
        "model": "q", "backend": "vllm",
        "results": {
            "piqa": {"score": 0.80, "metric": "acc", "num_samples": 10},
            "mmlu": {"score": 0.79, "metric": "acc", "num_samples": 10},
        },
        "commands": [],
    }))
    return out


def test_load_manifest_and_resolve_variant(tmp_path: Path):
    manifest = _manifest(tmp_path)
    path = _write_manifest(tmp_path, manifest)
    loaded = load_manifest(path)
    vid, variant = resolve_variant(loaded)
    assert vid == "mxfp4-v1" and variant["id"] == vid
    with pytest.raises(AqaError, match="not found"):
        resolve_variant(loaded, "nope")


def test_build_full_config_substitutes_variables(tmp_path: Path):
    manifest = _manifest(tmp_path)
    vid, variant = resolve_variant(manifest)
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "out")
    assert cfg.model == str(tmp_path / "model")
    assert cfg.provider.type == "quantize" and cfg.quant is not None
    script = cfg.quant.command_override
    assert str(tmp_path / "out" / "quantized") in script
    assert "${OUTPUT_DIR}" not in script and "${MODEL}" not in script
    assert cfg.quant.scheme == "MXFP4" and cfg.quant.iters == 0
    assert cfg.eval.benchmarks == ["piqa", "mmlu"]
    assert cfg.eval.thinking == "off"
    assert cfg.eval.lm_eval.model_args["dtype"] == "bfloat16"
    assert "tensor_parallel_size" not in cfg.eval.lm_eval.model_args
    assert cfg.reproduce is not None and cfg.reproduce.mode == "full"
    # Reproduce is read-only: no KB pin, no upload, no self-heal.
    assert cfg.provider.upload is None
    assert cfg.agent.recipe_kb.pin is False
    assert cfg.agent.fix_loop is False


def test_build_eval_only_config_uses_checkpoint(tmp_path: Path):
    manifest = _manifest(tmp_path)
    vid, variant = resolve_variant(manifest)
    checkpoint = tmp_path / "ckpt"
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "out", mode="eval-only",
        checkpoint=str(checkpoint))
    assert cfg.model == str(checkpoint)
    assert cfg.provider.type == "pretrained" and cfg.quant is None
    assert cfg.reproduce is not None and cfg.reproduce.mode == "eval-only"


def test_undeclared_variable_is_rejected(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["variants"][0]["evaluation"]["script"] = (
        'lm_eval --model vllm --model_args "pretrained=${NOPE}" --tasks piqa')
    vid, variant = resolve_variant(manifest)
    with pytest.raises(AqaError, match="undeclared variable"):
        build_reproduction_config(
            manifest, variant, manifest_path=tmp_path / "recipe.yaml",
            out_dir=tmp_path / "out")


def test_plan_and_expected_results(tmp_path: Path):
    manifest = _manifest(tmp_path)
    path = _write_manifest(tmp_path, manifest)
    plan = plan_reproduction(path, variant="mxfp4-v1", out_dir=tmp_path / "out")
    assert plan["variant"] == "mxfp4-v1"
    assert plan["evaluation"]["benchmarks"] == ["piqa", "mmlu"]
    assert expected_results(manifest, "mxfp4-v1") == {"piqa": 0.80, "mmlu": 0.79}


def test_verify_reproduced(tmp_path: Path):
    manifest = _manifest(tmp_path)
    vid, variant = resolve_variant(manifest)
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "run")
    out = _seed_run(tmp_path, manifest, cfg)
    # Use the actual per-module map as the expected one (tests the comparator,
    # not the estimator itself).
    manifest["quantization_details"] = {vid: _actual_quant_details(cfg, out)}
    report = verify_reproduction(
        manifest, variant, out_dir=out, mode="full", run_cfg=cfg,
        manifest_path=tmp_path / "recipe.yaml")
    assert report["input_equivalence"]["status"] == "match"
    assert report["quantization_reproduction"]["status"] == "pass"
    assert report["evaluation_reproduction"]["status"] == "pass"
    assert report["verdict"] == "reproduced"
    json.dumps(report, allow_nan=False)  # strict-JSON safe for external agents


def test_verify_numeric_mismatch_is_not_reproduced(tmp_path: Path):
    manifest = _manifest(tmp_path)
    vid, variant = resolve_variant(manifest)
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "run")
    out = _seed_run(tmp_path, manifest, cfg)
    manifest["result_tables"][0]["rows"][0]["values"] = {
        "piqa": 0.60, "mmlu": 0.79}
    report = verify_reproduction(
        manifest, variant, out_dir=out, mode="full", run_cfg=cfg,
        manifest_path=tmp_path / "recipe.yaml")
    assert report["evaluation_reproduction"]["status"] == "fail"
    assert report["verdict"] == "not_reproduced"


def test_input_mismatch_fail_makes_not_reproduced(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["environments"] = {
        "evaluation": {"packages": ["vllm==0.1.0"]}}
    vid, variant = resolve_variant(manifest)
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "run")
    out = _seed_run(tmp_path, manifest, cfg)
    manifest["quantization_details"] = {vid: _actual_quant_details(cfg, out)}
    report = verify_reproduction(
        manifest, variant, out_dir=out, mode="full", run_cfg=cfg,
        on_input_mismatch="fail", manifest_path=tmp_path / "recipe.yaml")
    assert report["input_equivalence"]["status"] == "mismatch"
    assert any("vllm" in d for d in report["input_equivalence"]["diffs"])
    assert report["verdict"] == "not_reproduced"


def test_verify_missing_expected_is_inconclusive(tmp_path: Path):
    manifest = _manifest(tmp_path)
    vid, variant = resolve_variant(manifest)
    cfg = build_reproduction_config(
        manifest, variant, manifest_path=tmp_path / "recipe.yaml",
        out_dir=tmp_path / "run")
    out = _seed_run(tmp_path, manifest, cfg)
    manifest["result_tables"] = []
    manifest["quantization_details"] = {}
    report = verify_reproduction(
        manifest, variant, out_dir=out, mode="full", run_cfg=cfg,
        manifest_path=tmp_path / "recipe.yaml")
    assert report["evaluation_reproduction"]["status"] == "inconclusive"
    assert report["quantization_reproduction"]["status"] == "skipped"
    assert report["verdict"] == "inconclusive"


def test_reproduce_dry_run_does_not_write(tmp_path: Path):
    manifest = _manifest(tmp_path)
    path = _write_manifest(tmp_path, manifest)
    report = reproduce_manifest(
        path, variant="mxfp4-v1", out_dir=tmp_path / "out", dry_run=True)
    assert report["variant"] == "mxfp4-v1"
    assert not (tmp_path / "out").exists()


def test_cli_dry_run(tmp_path: Path):
    from click.testing import CliRunner

    from aqa.cli import main

    manifest = _manifest(tmp_path)
    path = _write_manifest(tmp_path, manifest)
    result = CliRunner().invoke(
        main, ["reproduce", str(path), "--variant", "mxfp4-v1",
               "--out", str(tmp_path / "out"), "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["variant"] == "mxfp4-v1"
    assert plan["evaluation"]["benchmarks"] == ["piqa", "mmlu"]


def test_layout_contract_requires_report_for_reproduce(tmp_path: Path):
    from aqa.report.layout import inspect_run_dir

    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "reproduction": {"manifest": "x", "variant": "v", "mode": "full"}}))
    report = inspect_run_dir(tmp_path)
    assert any(e.path == "reproduction_report.json" for e in report.missing)
    (tmp_path / "reproduction_report.json").write_text("{}")
    report = inspect_run_dir(tmp_path)
    assert any(e.path == "reproduction_report.json" for e in report.present)


def test_pipeline_verify_phase_writes_report(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest["environments"] = {}
    manifest["quantization_details"] = {
        "mxfp4-v1": {"module_groups": [{"id": "mlp", "scheme": "MXFP4"}]}}
    path = _write_manifest(tmp_path, manifest)

    def fake_quant(script: str, quant_dir: Path):
        quant_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(_ARCH)
        cfg["quantization_config"] = {"quant_method": "auto-round", "bits": 4}
        (quant_dir / "config.json").write_text(json.dumps(cfg))
        return quant_dir

    def fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge,
                  run_id, **kwargs):
        result = EvalResult(model=str(model), backend="hf")
        result.add(BenchmarkResult(name="piqa", score=0.80, metric="acc"))
        result.add(BenchmarkResult(name="mmlu", score=0.79, metric="acc"))
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "accuracy.json").write_text(json.dumps(result.to_dict()))
        return result

    monkeypatch.setattr("aqa.pipeline.run_quant_script", fake_quant)
    monkeypatch.setattr("aqa.pipeline._run_evaluate", fake_eval)
    monkeypatch.setattr(
        "aqa.reproduce._actual_quant_details",
        lambda cfg, out: {"module_groups": [{"id": "mlp", "scheme": "MXFP4"}]})

    report = reproduce_manifest(
        path, variant="mxfp4-v1", out_dir=tmp_path / "out")
    assert report["verdict"] == "reproduced", report
    assert report["quantization_reproduction"]["status"] == "pass"
    assert report["evaluation_reproduction"]["status"] == "pass"
    out = tmp_path / "out"
    assert (out / "reproduction_report.json").is_file()
    assert (out / "run.yaml").is_file()  # resolved config snapshot
    # Every quantize run with eval results gets a model card; without a publish
    # sign-off it is a draft with template placeholders.
    card = (out / "model_card.md").read_text()
    assert "| piqa |" in card and "| mmlu |" in card
    assert "Signed-off-by: {member name} <{member email}>" in card
    assert "base_model:\n- {base_repo_id}" in card
    state = json.loads((out / "session" / "state.json").read_text())
    assert state["phases"]["VERIFY"] == "DONE"

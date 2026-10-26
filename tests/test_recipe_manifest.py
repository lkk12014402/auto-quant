"""Tests for the INC recipe.yaml emitter (RFC 0002, protocol 1.0)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from aqa.config import EvalConfig, ProviderConfig, QuantConfig, RunConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.report.recipe_manifest import (
    build_recipe_manifest,
    inc_schema_errors,
    to_inc_strict,
    write_recipe_manifest,
    write_strict_recipe,
)

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_SCHEMA = Path(__file__).parent / "data" / "inc-model-recipe.schema.json"

DENSE_ARCH = dict(
    hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
    num_key_value_heads=8, intermediate_size=28672, vocab_size=128256,
    tie_word_embeddings=False, architectures=["LlamaForCausalLM"],
    model_type="llama")


def _quant_cfg(**over):
    q = dict(scheme="MXFP8", method="model_free", iters=0, ignore_layers="lm_head")
    q.update(over)
    return RunConfig(
        model="meta-llama/Llama-3.3-70B-Instruct",
        provider=ProviderConfig(type="quantize", quant=QuantConfig(**q)),
        eval=EvalConfig(backend="vllm"))


def _result(commands=True, model_path="quantized"):
    r = EvalResult(model="q", backend="vllm")
    piqa = (f"lm_eval --model vllm --model_args pretrained={model_path} "
            "--tasks piqa" if commands else "")
    mmlu = (f"lm_eval --model vllm --model_args pretrained={model_path} "
            "--tasks mmlu" if commands else "")
    r.add(BenchmarkResult(name="piqa", score=0.83, metric="acc", command=piqa))
    r.add(BenchmarkResult(name="mmlu", score=0.80, metric="acc", command=mmlu))
    return r


def _baseline():
    b = EvalResult(model="bf16", backend="vllm")
    b.add(BenchmarkResult(name="piqa", score=0.84, metric="acc",
                          command="lm_eval --tasks piqa --model bf16"))
    b.add(BenchmarkResult(name="mmlu", score=0.82, metric="acc",
                          command="lm_eval --tasks mmlu --model bf16"))
    return b


def _seed_artifacts(out_dir, command=(
        "auto-round --model_name meta-llama/Llama-3.3-70B-Instruct "
        "--scheme MXFP8 --iters 0 --output_dir quantized")):
    (out_dir / "env_lock.json").write_text(json.dumps({
        "packages": {"torch": "2.6.0", "transformers": "4.50.0",
                     "auto_round": "0.15.0", "vllm": "0.8.0", "lm_eval": "0.4.9"},
        "python": "3.11.5",
        "accelerator": {"family": "cuda", "num_devices": 8,
                        "device_name": "NVIDIA H200"}}))
    (out_dir / "quant_plan.json").write_text(json.dumps({"command": command}))


def _seed_quant_dir(quant_dir):
    quant_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DENSE_ARCH)
    cfg["quantization_config"] = {
        "quant_method": "auto-round", "bits": 8,
        "extra_config": {"lm_head": {"bits": 16, "data_type": "float"}}}
    (quant_dir / "config.json").write_text(json.dumps(cfg))


def _assert_inc_structural(core):
    """Mirror the hard INC schema constraints (additionalProperties:false, ids)."""
    assert core["schema_version"] == "1.0"
    assert set(core) <= {"schema_version", "model", "environments", "operations",
                         "variants", "baselines", "result_tables"}
    assert {"schema_version", "model", "environments", "operations",
            "variants", "result_tables"} <= set(core)
    # model
    assert set(core["model"]) <= {"id", "name", "source", "revision", "comments"}
    assert {"id", "name", "source"} <= set(core["model"])
    assert _ID_RE.match(core["model"]["id"])
    # environments
    assert core["environments"]
    for env in core["environments"].values():
        assert env["packages"]
        for pkg in env["packages"]:
            assert "==" in pkg
    # operations
    assert core["operations"]
    for op_id, op in core["operations"].items():
        assert _ID_RE.match(op_id)
        assert op["label"] and op["settings"]
        for s in op["settings"]:
            assert set(s) <= {"key", "label", "value", "unit"}
            assert re.match(r"^[a-z][a-z0-9_.-]*$", s["key"])
    # variants: protocol 1.0 fields only (no status/working_directory/tasks).
    assert core["variants"]
    op_ids = set(core["operations"])
    for v in core["variants"]:
        assert set(v) <= {"id", "family", "label", "issues", "avg_bits", "hf_url",
                          "recommended", "private", "recipe", "quantization",
                          "evaluation"}
        assert {"id", "family", "label", "recipe", "quantization", "evaluation"} <= set(v)
        assert _ID_RE.match(v["id"]) and _ID_RE.match(v["family"])
        assert v["id"].startswith(v["family"] + "-v")
        assert set(v["recipe"]["operations"]) <= op_ids
        for block in (v["quantization"], v["evaluation"]):
            if block is None:
                assert v.get("issues")
                continue
            assert set(block) <= {"environment", "shell", "variables", "script",
                                  "notes", "comments", "script_url"}
            assert block["environment"] and block["shell"] == "bash"
            declared = set(block["variables"])
            used = set(_VAR_RE.findall(block["script"]))
            assert used <= declared
            assert not any(p.startswith("/") for p in block["script"].split())
            for name, binding in block["variables"].items():
                assert set(binding) <= {"kind", "description", "default", "from"}
                assert binding["kind"] in {"input_model", "output_dir",
                                           "checkpoint", "parameter"}
                assert binding.get("default") or binding.get("from")
    # result_tables
    for t in core["result_tables"]:
        assert set(t) <= {"id", "title", "notes", "comments", "hardware",
                          "columns", "rows"}
        assert {"id", "title", "notes", "columns", "rows"} <= set(t)
        for c in t["columns"]:
            assert set(c) <= {"id", "label", "kind", "unit", "task"}
            assert _ID_RE.match(c["id"])
            if c["kind"] == "metric":
                assert c.get("task")
            if c["kind"] == "bits":
                assert c["unit"] == "bits/weight"
        for row in t["rows"]:
            assert _ID_RE.match(row["id"])
            assert row["values"]
            if row["kind"] == "variant":
                assert "variant" in row and "baseline" not in row
            if row["kind"] == "baseline":
                assert "baseline" in row and "variant" not in row


def _assert_schema(core):
    errors = inc_schema_errors(core, _SCHEMA)
    assert errors == [], "\n".join(errors)


def test_build_manifest_on_go(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(model_path=str(qdir)), run_id="r1",
        out_dir=out, quant_dir=qdir, baseline=_baseline(), gate_verdict="GO",
        gate_reasons=[])
    # aqa extensions present at top level.
    assert "quantization_details" in m and "provenance" in m
    vid = m["variants"][0]["id"]
    assert vid == "mxfp8-v1"
    assert vid in m["quantization_details"]
    assert "status" not in m["variants"][0]
    # Commands are template scripts with declared variables, no local paths.
    quant = m["variants"][0]["quantization"]
    assert "${MODEL}" in quant["script"] and "${OUTPUT_DIR}" in quant["script"]
    assert "working_directory" not in quant and "tasks" not in quant
    eval_block = m["variants"][0]["evaluation"]
    assert "${MODEL_PATH}" in eval_block["script"]
    assert "working_directory" not in eval_block and "tasks" not in eval_block
    # Environment split.
    assert "quantization" in m["environments"] and "evaluation" in m["environments"]
    assert "auto-round==0.15.0" in m["environments"]["quantization"]["packages"]
    # Operations describe the actual MXFP8 scheme.
    settings = {s["key"]: s["value"] for s in m["operations"]["mxfp8-linear"]["settings"]}
    assert settings["data_type"] == "MXFP8"
    assert settings["block_size"] == 32
    # Result columns only carry task/unit (no non-schema `metric` field).
    for col in m["result_tables"][0]["columns"]:
        assert "metric" not in col
    # Strict projection drops extensions and validates against the INC schema.
    core = to_inc_strict(m)
    assert "quantization_details" not in core and "provenance" not in core
    _assert_inc_structural(core)
    _assert_schema(core)


def test_gate_not_go_is_an_issue(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, baseline=None, gate_verdict="NO-GO",
        gate_reasons=["mmlu regressed"])
    # No status field; incompleteness is expressed by `issues`.
    assert "status" not in m["variants"][0]
    assert any("Gate verdict was NO-GO" in i for i in m["variants"][0]["issues"])
    _assert_inc_structural(to_inc_strict(m))
    _assert_schema(to_inc_strict(m))


def test_missing_eval_command_degrades(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(commands=False), run_id="r1",
        out_dir=out, quant_dir=qdir, gate_verdict="GO", gate_reasons=[])
    # No eval commands -> evaluation null and an honest issue.
    assert m["variants"][0]["evaluation"] is None
    assert m["variants"][0]["issues"]
    _assert_inc_structural(to_inc_strict(m))
    _assert_schema(to_inc_strict(m))


def test_non_quantize_returns_none(tmp_path):
    cfg = RunConfig(model="org/m", provider=ProviderConfig(type="pretrained"))
    m = build_recipe_manifest(
        run_cfg=cfg, result=_result(), run_id="r1", out_dir=tmp_path)
    assert m is None


def test_write_round_trips(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, baseline=_baseline(), gate_verdict="GO", gate_reasons=[])
    path = write_recipe_manifest(out, m)
    assert path is not None and path.name == "recipe.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["schema_version"] == "1.0"
    assert loaded["variants"][0]["id"] == m["variants"][0]["id"]
    # Multi-line script survives the literal block round-trip.
    assert "\n" in loaded["variants"][0]["evaluation"]["script"]
    # CLOSE also writes the INC-strict sidecar named <model.id>.yaml.
    strict_path = out / f"{m['model']['id']}.yaml"
    assert strict_path.is_file()
    strict = yaml.safe_load(strict_path.read_text())
    assert "quantization_details" not in strict and "provenance" not in strict
    assert inc_schema_errors(strict, _SCHEMA) == []
    assert strict_path.name == "llama-3-3-70b-instruct.yaml"


def test_kv_and_attention_ops_from_command(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out, command=("auto-round --scheme MXFP8 --iters 0 "
                                  "--static_kv_dtype fp8 --static_attention_dtype fp8"))
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, gate_verdict="GO", gate_reasons=[])
    assert "fp8-kv-cache" in m["operations"]
    assert "fp8-attention" in m["operations"]
    vid = m["variants"][0]["id"]
    details = m["quantization_details"][vid]
    assert details.get("kv_cache", {}).get("scheme") == "FP8"
    _assert_inc_structural(to_inc_strict(m))
    _assert_schema(to_inc_strict(m))


def test_local_model_path_is_flagged_not_leaked(tmp_path):
    model_dir = tmp_path / "models" / "Base"
    model_dir.mkdir(parents=True)
    cfg = _quant_cfg()
    cfg.model = str(model_dir)
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out, command=(
        f"auto-round --model_name {model_dir} --scheme MXFP8 --iters 0 "
        f"--device_map auto --output_dir {out / 'quantized'}"))
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    r = _result()
    r.commands = [
        "lm_eval --model vllm --model_args "
        "pretrained=/stale/old/quantized,dtype=bfloat16 "
        f"--tasks piqa --output_path {out / 'eval'}"
    ]
    m = build_recipe_manifest(
        run_cfg=cfg, result=r, run_id="r1", out_dir=out, quant_dir=qdir,
        gate_verdict="GO", gate_reasons=[])
    core = to_inc_strict(m)
    script = core["variants"][0]["quantization"]["script"]
    assert str(model_dir) not in script
    assert str(out) not in script
    assert "${MODEL}" in script and "${OUTPUT_DIR}" in script
    eval_script = core["variants"][0]["evaluation"]["script"]
    assert "/stale/old/quantized" not in eval_script
    assert "${MODEL_PATH}" in eval_script
    assert any("local path" in i for i in core["variants"][0]["issues"])
    _assert_schema(core)


def test_smoke_limit_is_an_issue_and_table_note(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    cfg = _quant_cfg()
    cfg.eval.limit = 8
    m = build_recipe_manifest(
        run_cfg=cfg, result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, gate_verdict="GO", gate_reasons=[])
    assert any("smoke-limited" in i for i in m["variants"][0]["issues"])
    assert "Smoke evaluation" in m["result_tables"][0]["notes"]
    _assert_schema(to_inc_strict(m))


def test_owner_patch_is_recorded_in_issues(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    (out / "owner_actions.jsonl").write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00+00:00", "kind": "patch",
        "text": "vLLM wo_a patch applied"}) + "\n")
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, gate_verdict="GO", gate_reasons=[])
    assert any("vLLM wo_a patch applied" in i for i in m["variants"][0]["issues"])


def test_post_export_repair_is_in_quant_script_and_issues(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _seed_artifacts(out)
    qdir = out / "quantized"
    _seed_quant_dir(qdir)
    (out / ".post_export.json").write_text(json.dumps({
        "kind": "glm5_vllm", "targets": 36297,
        "backup": "/tmp/raw-config.json",
        "command": "python -m aqa.quantize.export_repair /tmp/quantized --kind glm5_vllm"}))
    m = build_recipe_manifest(
        run_cfg=_quant_cfg(), result=_result(), run_id="r1", out_dir=out,
        quant_dir=qdir, gate_verdict="GO", gate_reasons=[])
    script = m["variants"][0]["quantization"]["script"]
    assert "python -m aqa.quantize.export_repair" in script
    assert "--kind glm5_vllm" in script
    assert any("Post-export repair" in i for i in m["variants"][0]["issues"])
    _assert_schema(to_inc_strict(m))


def test_strict_sidecar_requires_model_id(tmp_path):
    assert write_strict_recipe(tmp_path, {"model": {}}) is None


def test_strict_sidecar_falls_back_when_id_is_recipe(tmp_path):
    path = write_strict_recipe(tmp_path, {
        "model": {"id": "recipe"}, "schema_version": "1.0",
        "environments": {}, "operations": {}, "variants": [], "result_tables": []})
    assert path is not None and path.name == "recipe.inc.yaml"

"""Tests for P2 phase-gate artifacts (aqa/report/phase_artifacts.py).

Each writer must emit deterministic, machine-readable JSON from a plain
``RunConfig`` / ``EvalResult`` (no GPU, no network), and ``aqa inspect`` must
require those artifacts once a run went through the session state machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aqa.config import ProviderConfig, QuantConfig, RunConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.report.layout import inspect_run_dir
from aqa.report.manifest import record_quant_command
from aqa.report.phase_artifacts import (
    ANALYSIS_FILE,
    ENV_LOCK_FILE,
    EVAL_REPORT_FILE,
    QUANT_PLAN_FILE,
    write_analysis,
    write_env_lock,
    write_eval_report,
    write_quant_plan,
)


def _quant_cfg(model: str = "org/m") -> RunConfig:
    return RunConfig(model=model, provider=ProviderConfig(
        type="quantize", quant=QuantConfig(scheme="MXFP4", method="autoround",
                                           iters=200, ignore_layers="lm_head")))


def _pretrained_cfg(model: str = "org/m") -> RunConfig:
    return RunConfig(model=model, provider=ProviderConfig(type="pretrained"))


def _eval_result(model: str, scores: dict[str, float]) -> EvalResult:
    r = EvalResult(model=model, backend="hf")
    for name, score in scores.items():
        r.add(BenchmarkResult(name=name, score=score, metric="acc", num_samples=10))
    return r


def test_env_lock_has_frozen_stack(tmp_path):
    p = write_env_lock(tmp_path, _pretrained_cfg(), "run-1")
    assert p == tmp_path / ENV_LOCK_FILE and p.is_file()
    data = json.loads(p.read_text())
    assert data["run_id"] == "run-1"
    assert {"python", "platform", "packages", "accelerator"} <= data.keys()
    assert data["accelerator"]["remote"] is False


def test_analysis_reads_local_model_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["FooForCausalLM"], "model_type": "foo",
        "num_hidden_layers": 48}))
    out = tmp_path / "run"
    out.mkdir()
    p = write_analysis(out, _quant_cfg(str(model_dir)), "run-2")
    assert p == out / ANALYSIS_FILE
    data = json.loads(p.read_text())
    assert data["identity"]["architectures"] == ["FooForCausalLM"]
    assert data["identity"]["num_hidden_layers"] == 48
    assert data["identity"]["multimodal"] is False
    assert data["identity"]["source"] == "config.json (local)"
    assert data["quant_scope"]["scheme"] == "MXFP4"
    # Full declared intent is captured (incl. ignore + mixed-precision knobs).
    assert data["quant_scope"]["ignore_layers"] == "lm_head"
    assert "layer_config" in data["quant_scope"]
    # Predicted per-module disposition is always present.
    assert data["disposition"]["stage"] == "predicted"
    assert data["disposition"]["module_groups"]
    assert data["planned_command"].startswith("auto-round")


def test_analysis_handles_multimodal_nested_layers(tmp_path):
    model_dir = tmp_path / "mm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["BarForConditionalGeneration"], "model_type": "bar",
        "num_hidden_layers": None, "vision_config": {"depth": 24},
        "text_config": {"num_hidden_layers": 64}}))
    out = tmp_path / "run"
    out.mkdir()
    data = json.loads(write_analysis(out, _quant_cfg(str(model_dir)), "r").read_text())
    assert data["identity"]["num_hidden_layers"] == 64
    assert data["identity"]["multimodal"] is True


def test_analysis_disposition_maps_layer_config_to_experts(tmp_path):
    """The experts→MXFP4 override must be visible and attributed in ANALYZE."""
    model_dir = tmp_path / "moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["MoeForCausalLM"], "model_type": "moe",
        "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16,
        "num_key_value_heads": 2, "n_routed_experts": 64, "moe_intermediate_size": 768,
        "vocab_size": 100000, "intermediate_size": 6144}))
    out = tmp_path / "run"
    out.mkdir()
    cfg = RunConfig(model=str(model_dir), provider=ProviderConfig(
        type="quantize", quant=QuantConfig(
            scheme="MXFP8", method="autoround",
            ignore_layers="lm_head",
            layer_config={"mlp.experts": {"bits": 4, "data_type": "mx_fp"}})))
    data = json.loads(write_analysis(out, cfg, "r").read_text())
    by = {g["id"]: g for g in data["disposition"]["module_groups"]}
    assert by["moe-experts"]["scheme"] == "MXFP4"
    assert by["moe-experts"]["matched_by"] == "layer_config:mlp.experts"
    assert by["moe-experts"]["quantized"] is True
    # Non-overridden Linear families stay on the global scheme.
    assert by["attn-proj"]["scheme"] == "MXFP8"
    assert by["attn-proj"]["matched_by"] == "scheme"
    # The override carries its nominal bit-width; mixing pulls effective bits
    # below the global MXFP8 (8).
    assert by["moe-experts"]["bits"] == 4
    assert data["disposition"]["effective_bits"] < 8.0


def test_analysis_quant_scope_none_for_pretrained(tmp_path):
    data = json.loads(write_analysis(tmp_path, _pretrained_cfg(), "r").read_text())
    assert data["quant_scope"] is None
    assert "disposition" not in data


def test_quant_plan_merges_recipe_and_command(tmp_path):
    record_quant_command(tmp_path, "auto-round --scheme MXFP4", source="builder")
    p = write_quant_plan(tmp_path, _quant_cfg(), "run-3")
    assert p == tmp_path / QUANT_PLAN_FILE
    data = json.loads(p.read_text())
    assert data["scheme"] == "MXFP4" and data["method"] == "autoround"
    assert data["iters"] == 200 and data["ignore_layers"] == "lm_head"
    assert data["command"] == "auto-round --scheme MXFP4"
    assert data["command_source"] == "builder"


def test_quant_plan_none_for_pretrained(tmp_path):
    assert write_quant_plan(tmp_path, _pretrained_cfg(), "r") is None
    assert not (tmp_path / QUANT_PLAN_FILE).exists()


def test_eval_report_computes_gap(tmp_path):
    quant = _eval_result("org/m-quant", {"gsm8k": 0.70, "mmlu": 0.60})
    base = _eval_result("org/m", {"gsm8k": 0.75, "mmlu": 0.62})
    p = write_eval_report(tmp_path, _quant_cfg(), "run-4", quant, base,
                          "GO", ["drop within budget"])
    data = json.loads(p.read_text())
    assert data["quant"]["mean_score"] == pytest.approx(0.65)
    assert data["source"]["mean_score"] == pytest.approx(0.685)
    assert data["gap"]["per_benchmark"]["gsm8k"] == pytest.approx(-0.05)
    assert data["gate"] == {"verdict": "GO", "reasons": ["drop within budget"]}


def test_eval_report_without_baseline(tmp_path):
    quant = _eval_result("org/m-quant", {"gsm8k": 0.70})
    data = json.loads(write_eval_report(
        tmp_path, _quant_cfg(), "r", quant, None, None).read_text())
    assert data["source"] is None and data["gap"] is None and data["gate"] is None


def _mk_session_run(tmp_path: Path, *, quantize: bool, done: bool) -> Path:
    (tmp_path / "aqa_run.json").write_text(json.dumps(
        {"model": "org/m", "state": "done" if done else "running"}))
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "session").mkdir(exist_ok=True)  # went through the state machine
    (tmp_path / "session" / "allowed_actions.json").write_text("{}")  # S3 contract (open_session)
    (tmp_path / "run_manifest.json").write_text("{}")
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "accuracy.json").write_text("{}")
    if quantize:
        (tmp_path / "quantized").mkdir(exist_ok=True)
        (tmp_path / "quantized" / "quantize.log").write_text("")
        (tmp_path / ".quant_command.json").write_text("{}")
        (tmp_path / "recipe.yaml").write_text("{}")  # RFC 0002 CLOSE product
    return tmp_path


def test_inspect_requires_session_artifacts(tmp_path):
    run = _mk_session_run(tmp_path, quantize=True, done=True)
    report = inspect_run_dir(run)
    missing = {e.path for e in report.missing}
    assert {ENV_LOCK_FILE, ANALYSIS_FILE, QUANT_PLAN_FILE, EVAL_REPORT_FILE} <= missing
    assert not report.ok

    for name in (ENV_LOCK_FILE, ANALYSIS_FILE, QUANT_PLAN_FILE, EVAL_REPORT_FILE):
        (run / name).write_text("{}")
    assert inspect_run_dir(run).ok


def test_inspect_pretrained_session_excludes_quant_plan(tmp_path):
    run = _mk_session_run(tmp_path, quantize=False, done=True)
    for name in (ENV_LOCK_FILE, ANALYSIS_FILE, EVAL_REPORT_FILE):
        (run / name).write_text("{}")
    report = inspect_run_dir(run)
    # quant_plan.json is session_quantize only => not expected without quantized/
    assert QUANT_PLAN_FILE not in {e.path for e in report.missing}
    assert report.ok


def test_inspect_running_session_defers_eval_report(tmp_path):
    run = _mk_session_run(tmp_path, quantize=False, done=False)
    for name in (ENV_LOCK_FILE, ANALYSIS_FILE):
        (run / name).write_text("{}")
    report = inspect_run_dir(run)
    # eval_report.json is session_done => not expected while still running
    assert EVAL_REPORT_FILE not in {e.path for e in report.missing}


def test_non_session_run_exempt_from_new_artifacts(tmp_path):
    (tmp_path / "aqa_run.json").write_text(json.dumps({"model": "org/m", "state": "done"}))
    (tmp_path / "logs").mkdir()
    (tmp_path / "run_manifest.json").write_text("{}")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "accuracy.json").write_text("{}")
    missing = {e.path for e in inspect_run_dir(tmp_path).missing}
    assert not ({ENV_LOCK_FILE, ANALYSIS_FILE, EVAL_REPORT_FILE} & missing)


def test_pipeline_writes_all_phase_artifacts_and_inspect_ok(tmp_path, monkeypatch):
    """End-to-end: a real run_pipeline must drop every contracted phase artifact
    and the run dir must then pass ``aqa inspect`` (session_quantize + session_done).
    """
    import aqa.pipeline as P

    monkeypatch.setattr(P, "_dispatch_sinks", lambda *a, **k: None)
    monkeypatch.setattr(P, "_make_knowledge", lambda rc: None)
    monkeypatch.setattr(P, "_finalize_knowledge", lambda k, **kw: None)
    monkeypatch.setattr(P, "_maybe_upload_model", lambda *a, **k: None)

    class _Resolved:
        model = "org/m-quant"
        upload = None

    def _fake_resolve(run_cfg, out_dir, *a, **k):
        # Mimic a real quantize provider's on-disk footprint so inspect detects
        # the quantize condition and requires quant_plan.json.
        record_quant_command(out_dir, "auto-round --scheme MXFP4", source="builder")
        (out_dir / "quantized").mkdir(parents=True, exist_ok=True)
        (out_dir / "quantized" / "quantize.log").write_text("done")
        return _Resolved()

    monkeypatch.setattr(P, "_resolve_model", _fake_resolve)

    def _fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge=None,
                   run_id="", phase_name="evaluate", heal=None, session=None):
        score = 0.80 if phase_name == "baseline" else 0.74
        return _eval_result(model, {"piqa": score})

    monkeypatch.setattr(P, "_run_evaluate", _fake_eval)

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": True},
        "gate": {"max_accuracy_drop": 0.1},
        "output": {"dir": str(tmp_path), "sinks": []},
    })
    P.run_pipeline(cfg, tmp_path)

    # run_pipeline writes the phase artifacts + session/; the CLI `run` wrapper
    # supplies the always-on run envelope (aqa_run.json + logs/). Simulate it so
    # inspect can judge the full contract.
    (tmp_path / "aqa_run.json").write_text(
        json.dumps({"model": cfg.model, "state": "done"}))
    (tmp_path / "logs").mkdir(exist_ok=True)

    for name in (ENV_LOCK_FILE, ANALYSIS_FILE, QUANT_PLAN_FILE, EVAL_REPORT_FILE):
        assert (tmp_path / name).is_file(), f"{name} not written by pipeline"

    plan = json.loads((tmp_path / QUANT_PLAN_FILE).read_text())
    assert plan["command"] == "auto-round --scheme MXFP4"
    report = json.loads((tmp_path / EVAL_REPORT_FILE).read_text())
    assert report["gap"]["per_benchmark"]["piqa"] == pytest.approx(-0.06)
    assert report["gate"]["verdict"] == "GO"

    inspected = inspect_run_dir(tmp_path)
    assert inspected.ok, [e.path for e in inspected.missing]


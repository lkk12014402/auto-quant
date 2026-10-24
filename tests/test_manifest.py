"""Tests for the reproducibility manifest (NeMo-Evaluator-style run record)."""

from __future__ import annotations

import json

from aqa.config import RunConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.report.manifest import SCHEMA_VERSION, build_manifest, write_manifest


def _result(scores: dict[str, float]) -> EvalResult:
    r = EvalResult(model="org/m", backend="vllm")
    for name, s in scores.items():
        r.add(BenchmarkResult(name=name, score=s, metric="acc", num_samples=100))
    return r


def _cfg(**over) -> RunConfig:
    base = {
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4", "method": "autoround"}},
        "eval": {"benchmarks": ["piqa", "gsm8k"], "backend": "vllm",
                 "limit": 50, "lm_eval": {"seed": 7}},
    }
    base.update(over)
    return RunConfig.model_validate(base)


def test_manifest_captures_model_quant_eval_and_results():
    cfg = _cfg()
    result = _result({"piqa": 0.8, "gsm8k": 0.5})
    m = build_manifest(cfg, result, run_id="rid-1", gate_verdict="GO")

    d = m.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["run_id"] == "rid-1"
    assert d["model"] == {"id": "org/m", "provider": "quantize"}
    assert d["quant"]["scheme"] == "MXFP4" and d["quant"]["method"] == "autoround"
    # eval config fingerprint
    assert d["eval"]["benchmarks"] == ["piqa", "gsm8k"]
    assert d["eval"]["backend"] == "vllm"
    assert d["eval"]["limit"] == 50
    assert d["eval"]["seed"] == 7
    # results + mean
    assert d["results"]["mean_score"] == 0.65
    assert d["results"]["benchmarks"]["piqa"]["num_samples"] == 100
    assert d["gate"] == {"verdict": "GO"}
    # environment fingerprint always present
    assert "python" in d["environment"] and "packages" in d["environment"]
    assert d["environment"]["accelerator"]["family"] in ("cpu/local", "cuda", "xpu")


def test_manifest_quant_none_for_pretrained_provider():
    cfg = _cfg(provider={"type": "pretrained"})
    m = build_manifest(cfg, _result({"piqa": 1.0}), run_id="r")
    assert m.quant is None
    assert m.model["provider"] == "pretrained"


def test_manifest_includes_baseline_when_given():
    cfg = _cfg()
    m = build_manifest(cfg, _result({"piqa": 0.7}), run_id="r",
                       baseline=_result({"piqa": 0.9}))
    assert m.results["baseline"]["mean_score"] == 0.9


def test_manifest_accelerator_reflects_ssh_family():
    cfg = _cfg(executor={"type": "ssh",
                         "ssh": {"host": "u@h", "gpus": "0,1,2", "accelerator": "xpu"}})
    m = build_manifest(cfg, _result({"piqa": 1.0}), run_id="r")
    acc = m.environment["accelerator"]
    assert acc["family"] == "xpu" and acc["num_devices"] == 3 and acc["remote"] is True


def test_manifest_accelerator_reflects_local_gpu_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    cfg = _cfg(executor={"type": "local", "num_gpus": 8})

    m = build_manifest(cfg, _result({"piqa": 1.0}), run_id="r")

    acc = m.environment["accelerator"]
    assert acc["family"] == "cuda"
    assert acc["num_devices"] == 8
    assert acc["executor"] == "local" and acc["remote"] is False
    # ``device_name`` is an optional probe detail (not present under mocks).
    assert set(acc) <= {"family", "num_devices", "executor", "remote", "device_name"}


def test_write_manifest_roundtrip(tmp_path):
    cfg = _cfg()
    m = build_manifest(cfg, _result({"piqa": 0.8}), run_id="rid-9",
                       config_path="examples/run.yaml")
    path = write_manifest(m, tmp_path)
    assert path.name == "run_manifest.json"
    loaded = json.loads(path.read_text())
    assert loaded["run_id"] == "rid-9"
    assert loaded["reproduce"]["command"] == "aqa run examples/run.yaml"


def test_pipeline_writes_manifest_on_success(tmp_path, monkeypatch):
    import aqa.pipeline as P

    monkeypatch.setattr(P, "_dispatch_sinks", lambda *a, **k: None)
    monkeypatch.setattr(P, "_make_knowledge", lambda rc: None)
    monkeypatch.setattr(P, "_finalize_knowledge", lambda k: None)

    class _Resolved:
        model = "org/m"
        upload = None

    monkeypatch.setattr(P, "_resolve_model", lambda *a, **k: _Resolved())
    monkeypatch.setattr(P, "_run_evaluate", lambda *a, **k: _result({"piqa": 0.8}))
    monkeypatch.setattr(P, "_maybe_upload_model", lambda *a, **k: None)

    cfg = _cfg(output={"dir": str(tmp_path), "sinks": []})
    state = P.run_pipeline(cfg, tmp_path)
    assert state.value == "done"

    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()
    d = json.loads(manifest_path.read_text())
    assert d["results"]["mean_score"] == 0.8
    assert d["quant"]["scheme"] == "MXFP4"


# ── commands block: the ACTUAL executed quant/eval commands ────────────────────
def _result_with_commands() -> EvalResult:
    r = EvalResult(model="org/m", backend="vllm")
    br = BenchmarkResult(name="piqa", score=0.8, metric="acc", num_samples=100)
    br.command = "lm_eval --model vllm --tasks piqa --batch_size 8"
    r.add(br)
    r.healed = ["batch_size 16->8 (oom)"]
    return r


def test_commands_block_captures_eval_and_healed():
    cfg = _cfg()
    m = build_manifest(cfg, _result_with_commands(), run_id="r")
    cmds = m.to_dict()["commands"]
    assert cmds["evaluate"] == ["lm_eval --model vllm --tasks piqa --batch_size 8"]
    assert cmds["healed"] == ["batch_size 16->8 (oom)"]


def test_commands_block_reads_recorded_quant_command(tmp_path):
    from aqa.report.manifest import record_quant_command

    record_quant_command(tmp_path, "auto-round --model_name org/m --scheme MXFP4 "
                         "--device_map auto", source="agent_optimized")
    m = build_manifest(_cfg(), _result({"piqa": 0.8}), run_id="r", out_dir=tmp_path)
    cmds = m.to_dict()["commands"]
    assert cmds["quantize"].startswith("auto-round --model_name org/m")
    assert cmds["quantize_source"] == "agent_optimized"


def test_commands_block_empty_without_provenance():
    m = build_manifest(_cfg(), _result({"piqa": 0.8}), run_id="r")
    assert m.to_dict()["commands"] == {}


def test_eval_result_auto_collects_command_on_add():
    r = EvalResult(model="m", backend="vllm")
    a = BenchmarkResult(name="a", score=1.0, metric="acc")
    a.command = "lm_eval --tasks a+b"
    b = BenchmarkResult(name="b", score=0.5, metric="acc")
    b.command = "lm_eval --tasks a+b"  # same grouped call -> deduped
    r.add(a)
    r.add(b)
    assert r.commands == ["lm_eval --tasks a+b"]


# --------------------------------------------------------------------------
# efficiency block (agent tokens + turns alongside the score)
# --------------------------------------------------------------------------
def test_manifest_no_efficiency_for_pure_lm_eval():
    m = build_manifest(_cfg(), _result({"piqa": 0.8}), run_id="r")
    assert "efficiency" not in m.to_dict()["results"]


def test_manifest_efficiency_from_agent_usage():
    from aqa.agent_eval.loop import AgentEvalResult, TaskOutcome
    from aqa.agent_eval.stats import TokenStats

    usage = AgentEvalResult(name="swebench", outcomes=[
        TaskOutcome("t1", "", 1.0,
                    stats=TokenStats(n_input_tokens=200, n_output_tokens=50), turns=3),
        TaskOutcome("t2", "", 0.0,
                    stats=TokenStats(n_input_tokens=100, n_output_tokens=20), turns=1),
    ])
    m = build_manifest(_cfg(), _result({"piqa": 0.8}), run_id="r", usage=usage)
    eff = m.to_dict()["results"]["efficiency"]
    assert eff["tokens"]["n_input_tokens"] == 300
    assert eff["tokens"]["n_output_tokens"] == 70
    assert eff["total_turns"] == 4
    assert eff["avg_turns"] == 2.0


def test_manifest_efficiency_from_bridged_benchmark():
    from aqa.agent_eval.loop import AgentEvalResult, TaskOutcome
    from aqa.agent_eval.stats import TokenStats

    agent_result = AgentEvalResult(name="swe", outcomes=[
        TaskOutcome("t1", "", 1.0,
                    stats=TokenStats(n_input_tokens=80, n_output_tokens=12), turns=2),
    ])
    r = EvalResult(model="org/m", backend="vllm")
    r.add(BenchmarkResult.from_agent_result(agent_result, name="swebench"))
    m = build_manifest(_cfg(), r, run_id="r")
    eff = m.to_dict()["results"]["efficiency"]
    assert eff["tokens"]["n_input_tokens"] == 80
    assert eff["total_turns"] == 2
    assert eff["num_benchmarks"] == 1


def test_manifest_self_heal_block_recorded():
    cfg = _cfg()
    result = _result({"piqa": 0.8})
    self_heal = {
        "agent_attempts": 3,
        "deterministic_heals": 1,
        "per_phase": {
            "quantize": {"agent_attempts": 2, "deterministic_heals": 0,
                         "tiers": 1, "tokens": {"n_input_tokens": 100,
                                                "n_output_tokens": 30,
                                                "total_tokens": 130}},
        },
        "tokens": {"n_input_tokens": 100, "n_output_tokens": 30, "total_tokens": 130},
    }
    m = build_manifest(cfg, result, run_id="rid-heal", self_heal=self_heal)
    sh = m.to_dict()["results"]["self_heal"]
    assert sh["agent_attempts"] == 3
    assert sh["deterministic_heals"] == 1
    assert sh["per_phase"]["quantize"]["tiers"] == 1
    assert sh["tokens"]["total_tokens"] == 130


def test_manifest_no_self_heal_when_empty():
    cfg = _cfg()
    result = _result({"piqa": 0.8})
    m = build_manifest(cfg, result, run_id="rid-noheal", self_heal=None)
    assert "self_heal" not in m.to_dict()["results"]

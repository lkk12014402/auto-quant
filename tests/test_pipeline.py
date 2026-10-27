import json
from pathlib import Path

from aqa.config import GateConfig, QuantConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.quantize import build_autoround_command
from aqa.report import evaluate_gate


def _result(scores: dict[str, float]) -> EvalResult:
    r = EvalResult(model="org/m", backend="vllm")
    for name, s in scores.items():
        r.add(BenchmarkResult(name=name, score=s, metric="acc"))
    return r


def test_gate_go_when_drop_small():
    base = _result({"piqa": 0.80, "gsm8k": 0.50})
    quant = _result({"piqa": 0.79, "gsm8k": 0.49})
    d = evaluate_gate(quant, base, GateConfig(max_accuracy_drop=0.02))
    assert d.verdict == "GO"


def test_gate_nogo_when_drop_large():
    base = _result({"piqa": 0.80})
    quant = _result({"piqa": 0.70})
    d = evaluate_gate(quant, base, GateConfig(max_accuracy_drop=0.02))
    assert d.verdict == "NO-GO"


def test_gate_inconclusive_without_baseline():
    quant = _result({"piqa": 0.70})
    d = evaluate_gate(quant, None, GateConfig(max_accuracy_drop=0.02))
    assert d.verdict == "INCONCLUSIVE"


def test_gate_min_accuracy():
    quant = _result({"piqa": 0.40})
    d = evaluate_gate(quant, None, GateConfig(min_accuracy=0.5))
    assert d.verdict == "NO-GO"


def test_build_autoround_command():
    cfg = QuantConfig(scheme="MXFP4", method="model_free", iters=200)
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert cmd[:3] == ["auto-round", "--model_name", "org/m"]
    assert "--model_free" in cmd
    assert "--scheme" in cmd and "MXFP4" in cmd
    assert "--iters" in cmd and "200" in cmd
    assert "--format" in cmd and "llm_compressor" in cmd


def _adjacent(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_build_autoround_command_algorithm_awq():
    cfg = QuantConfig(scheme="W4A16", method="autoround", algorithm="awq,auto_round")
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert "--model_free" not in cmd
    assert _adjacent(cmd, "--algorithm") == "awq,auto_round"


def test_build_autoround_command_alg_ext():
    cfg = QuantConfig(scheme="W4A16", method="autoround", algorithm="auto_round", enable_alg_ext=True)
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert "--enable_alg_ext" in cmd


def test_build_autoround_command_hadamard():
    cfg = QuantConfig(scheme="MXFP4", method="autoround", algorithm="hadamard", hadamard_type="quarot_hadamard")
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert _adjacent(cmd, "--algorithm") == "hadamard"
    assert _adjacent(cmd, "--rotation-hadamard-type") == "quarot_hadamard"


def test_build_autoround_command_opt_rtn():
    on = build_autoround_command("org/m", QuantConfig(algorithm="rtn", opt_rtn=True), Path("/tmp/o"))
    off = build_autoround_command("org/m", QuantConfig(algorithm="rtn", opt_rtn=False), Path("/tmp/o"))
    assert "--enable_opt_rtn" in on and "--disable_opt_rtn" not in on
    assert "--disable_opt_rtn" in off and "--enable_opt_rtn" not in off


def test_build_autoround_command_layer_config_emitted_as_json():
    # Per-layer mixed-precision overrides must reach the CLI as compact JSON,
    # not be silently dropped (regression: field existed but was never emitted).
    cfg = QuantConfig(
        scheme="MXFP8", method="autoround", iters=0,
        ignore_layers="lm_head,embed_tokens,mlp.router",
        layer_config={"mlp.experts": {"bits": 4, "group_size": 32, "data_type": "mx_fp"}},
    )
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert "--model_free" not in cmd            # model-load RTN path
    assert _adjacent(cmd, "--iters") == "0"
    lc = _adjacent(cmd, "--layer_config")
    assert json.loads(lc) == {"mlp.experts": {"bits": 4, "group_size": 32, "data_type": "mx_fp"}}
    assert " " not in lc                        # compact (no spaces)


def test_build_autoround_command_no_layer_config_flag_when_absent():
    cfg = QuantConfig(scheme="MXFP8", method="autoround", iters=0)
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert "--layer_config" not in cmd



    cfg = QuantConfig(scheme="MXFP8", method="model_free")
    single = build_autoround_command("org/m", cfg, Path("/tmp/o"), num_gpus=1)
    multi = build_autoround_command("org/m", cfg, Path("/tmp/o"), num_gpus=4)
    assert "--device_map" not in single          # single GPU => auto-round default
    assert _adjacent(multi, "--device_map") == "auto"


def test_build_autoround_command_device_map_explicit_wins():
    cfg = QuantConfig(scheme="MXFP8", method="model_free", device_map="0,1")
    # explicit config is honored even on a single GPU, and not doubled on multi.
    single = build_autoround_command("org/m", cfg, Path("/tmp/o"), num_gpus=1)
    multi = build_autoround_command("org/m", cfg, Path("/tmp/o"), num_gpus=4)
    assert _adjacent(single, "--device_map") == "0,1"
    assert multi.count("--device_map") == 1 and _adjacent(multi, "--device_map") == "0,1"


def test_ensure_device_map_injects_for_agent_command():
    from aqa.quantize import ensure_device_map

    cfg = QuantConfig(scheme="MXFP8", method="model_free")
    base = "auto-round --model_name org/m --scheme MXFP8 --model_free"
    assert ensure_device_map(base, cfg, num_gpus=1) == base            # single GPU: untouched
    injected = ensure_device_map(base, cfg, num_gpus=8)
    assert injected == base + " --device_map auto"
    # already-present device_map is never doubled
    with_dm = base + " --device_map cpu"
    assert ensure_device_map(with_dm, cfg, num_gpus=8) == with_dm


def test_quantconfig_rejects_unknown_algorithm():
    import pytest

    with pytest.raises(ValueError, match="unknown quant algorithm"):
        QuantConfig(algorithm="not_a_real_algo")


def test_quantconfig_alg_ext_requires_auto_round():
    import pytest

    with pytest.raises(ValueError, match="enable_alg_ext"):
        QuantConfig(algorithm="rtn", enable_alg_ext=True)


def test_build_autoround_command_stacked_transforms():
    cfg = QuantConfig(scheme="MXFP4", method="autoround", algorithm="awq,hadamard,auto_round")
    cmd = build_autoround_command("org/m", cfg, Path("/tmp/out"))
    assert _adjacent(cmd, "--algorithm") == "awq,hadamard,auto_round"


def test_quantconfig_rejects_two_block_quantizers():
    import pytest

    with pytest.raises(ValueError, match="only one block-quantizer"):
        QuantConfig(algorithm="rtn,auto_round")


def test_quantconfig_model_free_rejects_calibrated_algorithm():
    import pytest

    with pytest.raises(ValueError, match="calibration-free"):
        QuantConfig(method="model_free", algorithm="awq,auto_round")


def test_pipeline_writes_back_status_on_eval_failure(tmp_path, monkeypatch):
    """A crashed eval still uploads artifacts + a failure status (not silence)."""
    import pytest

    import aqa.pipeline as P
    from aqa.config import RunConfig
    from aqa.report.sinks import RunMeta

    captured: list[RunMeta] = []

    def _fake_dispatch(run_cfg, run_dir, meta):
        captured.append(meta)

    def _boom(*a, **k):
        from aqa.logging_util import AqaError
        raise AqaError("evaluate failed after fix-loop: drift: unknown")

    monkeypatch.setattr(P, "_dispatch_sinks", _fake_dispatch)
    monkeypatch.setattr(P, "_run_evaluate", _boom)

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": False},
        "output": {"dir": str(tmp_path), "sinks": []},
    })
    with pytest.raises(P.AqaError, match="evaluate failed"):
        P.run_pipeline(cfg, tmp_path)

    assert captured, "failure path must still dispatch sinks"
    assert captured[-1].status == "Eval Failed"
    assert captured[-1].run_root == tmp_path


def test_pipeline_failure_status_is_quant_failed_when_quantize_raises(tmp_path, monkeypatch):
    import pytest

    import aqa.pipeline as P
    from aqa.config import RunConfig
    from aqa.report.sinks import RunMeta

    captured: list[RunMeta] = []
    monkeypatch.setattr(P, "_dispatch_sinks", lambda rc, rd, m: captured.append(m))

    def _boom(*a, **k):
        raise P.AqaError("quantize failed after fix-loop: drift: oom")

    monkeypatch.setattr(P, "_resolve_model", _boom)

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": False},
        "output": {"dir": str(tmp_path), "sinks": []},
    })
    with pytest.raises(P.AqaError, match="quantize failed"):
        P.run_pipeline(cfg, tmp_path)

    assert captured and captured[-1].status == "Quant Failed"



# ── multi-GPU resolution + baseline agent-fix routing ─────────────────────────
def test_num_gpus_prefers_explicit_executor_field():
    import aqa.pipeline as P
    from aqa.config import RunConfig

    # In-container run: executor.type=local with an explicit device count.
    cfg = RunConfig.model_validate({
        "model": "org/m",
        "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
        "executor": {"type": "local", "num_gpus": 4},
    })
    assert P._num_gpus(cfg) == 4


def test_num_gpus_derived_from_ssh_gpu_list():
    import aqa.pipeline as P
    from aqa.config import RunConfig

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
        "executor": {"type": "ssh",
                     "ssh": {"host": "u@h", "gpus": "5,6",
                             "auth": {"method": "password", "password_env": "GPU_SSH_PASS"}}},
    })
    assert P._num_gpus(cfg) == 2


def test_baseline_goes_through_agent_fix_wrapper(tmp_path, monkeypatch):
    import aqa.pipeline as P
    from aqa.config import RunConfig

    monkeypatch.setattr(P, "_dispatch_sinks", lambda *a, **k: None)
    monkeypatch.setattr(P, "_make_knowledge", lambda rc: None)
    monkeypatch.setattr(P, "_finalize_knowledge", lambda k, **kw: None)
    monkeypatch.setattr(P, "_maybe_upload_model", lambda *a, **k: None)

    class _Resolved:
        model = "org/m"
        upload = None

    monkeypatch.setattr(P, "_resolve_model", lambda *a, **k: _Resolved())

    calls: list[str] = []

    def _fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge=None,
                   run_id="", phase_name="evaluate", heal=None, session=None):
        calls.append(phase_name)
        return _result({"piqa": 0.8})

    monkeypatch.setattr(P, "_run_evaluate", _fake_eval)

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": True},
        "output": {"dir": str(tmp_path), "sinks": []},
    })
    P.run_pipeline(cfg, tmp_path)
    # baseline AND the quantized eval both route through the fix-loop wrapper.
    assert calls == ["baseline", "evaluate"]


def test_pipeline_records_self_heal_in_manifest(tmp_path, monkeypatch):
    import json

    import aqa.pipeline as P
    from aqa.agent.fix_loop import FixResult
    from aqa.agent_eval.stats import TokenStats
    from aqa.config import RunConfig

    monkeypatch.setattr(P, "_dispatch_sinks", lambda *a, **k: None)
    monkeypatch.setattr(P, "_make_knowledge", lambda rc: None)
    monkeypatch.setattr(P, "_finalize_knowledge", lambda k, **kw: None)
    monkeypatch.setattr(P, "_maybe_upload_model", lambda *a, **k: None)

    class _Resolved:
        model = "org/m"
        upload = None

    # Quantize provider path records a two-attempt agent fix into the tally.
    def _fake_resolve(run_cfg, out_dir, num_gpus, knowledge=None, run_id="", heal=None,
                      hot_start="", session=None):
        if heal is not None:
            heal.record_fix("quantize", FixResult(
                ok=True, attempts=2, tier_index=0,
                tokens=TokenStats(n_input_tokens=100, n_output_tokens=30)))
        return _Resolved()

    monkeypatch.setattr(P, "_resolve_model", _fake_resolve)

    def _fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge=None,
                   run_id="", phase_name="evaluate", heal=None, session=None):
        if heal is not None:
            heal.record_deterministic(phase_name, ["retry_oom"])
        return _result({"piqa": 0.8})

    monkeypatch.setattr(P, "_run_evaluate", _fake_eval)

    cfg = RunConfig.model_validate({
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
        "output": {"dir": str(tmp_path), "sinks": []},
    })
    P.run_pipeline(cfg, tmp_path)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    sh = manifest["results"]["self_heal"]
    assert sh["agent_attempts"] == 2
    assert sh["deterministic_heals"] == 1
    assert sh["per_phase"]["quantize"]["agent_attempts"] == 2
    assert sh["per_phase"]["evaluate"]["deterministic_heals"] == 1
    assert sh["tokens"]["n_input_tokens"] == 100

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

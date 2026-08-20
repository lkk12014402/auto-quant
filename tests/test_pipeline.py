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

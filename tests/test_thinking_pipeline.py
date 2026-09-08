"""Pipeline-level thinking-dual-mode integration: dir routing + gate pairing."""

from __future__ import annotations

import json
from pathlib import Path

import aqa.pipeline as pl
from aqa.config import EvalConfig, RunConfig
from aqa.eval.registry import EvalResult


def _stub_result(model: str, score: float) -> EvalResult:
    from aqa.eval.registry import BenchmarkResult
    r = EvalResult(model=model, backend="hf")
    r.add(BenchmarkResult(name="gsm8k", score=score, metric="exact_match", num_samples=100))
    return r


def _run_cfg(tmp_path: Path, **eval_kw) -> RunConfig:
    return RunConfig(model="org/m", eval=EvalConfig(**eval_kw),
                     output={"dir": str(tmp_path), "sinks": []})


def test_run_evaluate_modes_dual_dirs(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge=None,
                  run_id="", phase_name="evaluate", heal=None):
        calls.append(str(eval_dir))
        assert model == "org/m"
        return _stub_result(model, 0.9)

    monkeypatch.setattr(pl, "_run_evaluate", fake_eval)
    cfg = _run_cfg(tmp_path, thinking="both")
    modes = pl._run_evaluate_modes(cfg, "org/m", tmp_path / "eval", tmp_path, 1)
    assert list(modes) == ["on", "off"]
    assert calls[0].endswith("/eval") and calls[1].endswith("/eval_thinking_off")
    for d in calls:
        assert (Path(d) / "accuracy.json").is_file()


def test_run_evaluate_modes_single_legacy_dir(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_eval(run_cfg, model, eval_dir, out_dir, num_gpus, knowledge=None,
                  run_id="", phase_name="evaluate", heal=None):
        calls.append(str(eval_dir))
        return _stub_result(model, 0.9)

    monkeypatch.setattr(pl, "_run_evaluate", fake_eval)
    cfg = _run_cfg(tmp_path, thinking="on")
    modes = pl._run_evaluate_modes(cfg, "org/m", tmp_path / "eval", tmp_path, 1)
    assert list(modes) == ["on"] and calls == [str(tmp_path / "eval")]

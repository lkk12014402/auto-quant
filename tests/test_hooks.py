"""Phase-hook layer tests: protocol/registration, config, seed hooks, wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import aqa.pipeline as P
from aqa.agent.fix_loop import FixResult
from aqa.config import parse_run_config
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.hooks import (
    DEFAULT_HOOKS,
    PhaseHook,
    resolve_hooks,
    run_hook_chain,
    run_hooks,
)
from aqa.hooks.sanity import EvalSanityHook, QuantSanityHook
from aqa.logging_util import AqaError, ConfigError
from aqa.quantize import run_quantization
from aqa.registry import _BUILTINS, HOOKS

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── registration: builtins + entry-point dual track ──────────────────────────
def test_hooks_registered_in_builtins_and_pyproject():
    assert _BUILTINS["aqa.hooks"] == {
        "quant_sanity": "aqa.hooks.sanity:QuantSanityHook",
        "eval_sanity": "aqa.hooks.sanity:EvalSanityHook",
    }
    assert set(HOOKS.names()) >= {"quant_sanity", "eval_sanity"}
    assert HOOKS.create("quant_sanity").name == "quant_sanity"
    assert HOOKS.create("eval_sanity").name == "eval_sanity"
    # entry-point group mirrors the builtins (installed installs expose them)
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert '[project.entry-points."aqa.hooks"]' in pyproject
    assert 'quant_sanity = "aqa.hooks.sanity:QuantSanityHook"' in pyproject
    assert 'eval_sanity = "aqa.hooks.sanity:EvalSanityHook"' in pyproject


def test_runtime_registration_dual_track():
    HOOKS.register("tmp_hook", "aqa.hooks.sanity:EvalSanityHook")
    try:
        assert "tmp_hook" in HOOKS.names()
        assert isinstance(HOOKS.create("tmp_hook"), EvalSanityHook)
    finally:
        HOOKS._extra.pop("tmp_hook", None)


def test_protocol_conformance():
    class _Full:
        name = "full"

        def after_quantize(self, run_cfg, quant_cfg, output_dir):
            ...

        def after_benchmark(self, run_cfg, recipe, result_or_exception, out_dir):
            ...

    assert isinstance(_Full(), PhaseHook)
    # hooks may implement only the phases they care about
    assert not isinstance(EvalSanityHook(), PhaseHook)


# ── config ───────────────────────────────────────────────────────────────────
def test_hooks_config_default_explicit_empty():
    assert parse_run_config({"model": "org/m"}).hooks is None
    assert [h.name for h in resolve_hooks(parse_run_config({"model": "org/m"}))] \
        == list(DEFAULT_HOOKS)
    cfg = parse_run_config({"model": "org/m", "hooks": ["eval_sanity"]})
    assert [h.name for h in resolve_hooks(cfg)] == ["eval_sanity"]
    assert resolve_hooks(parse_run_config({"model": "org/m", "hooks": []})) == ()


def test_hooks_config_unknown_name_rejected():
    with pytest.raises(ConfigError, match="unknown hook"):
        parse_run_config({"model": "org/m", "hooks": ["nope"]})


# ── quant_sanity ─────────────────────────────────────────────────────────────
def _write_st(dirpath: Path, keys: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    tensors = {k: np.zeros((2, 2), dtype=np.float32) for k in keys}
    save_file(tensors, str(dirpath / "model.safetensors"))


def _claim_autoround(dirpath: Path, quant_method="auto-round") -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((dirpath / "config.json").read_text()) \
        if (dirpath / "config.json").exists() else {}
    cfg["quantization_config"] = {"quant_method": quant_method, "bits": 4}
    (dirpath / "config.json").write_text(json.dumps(cfg))


def test_quant_sanity_fails_on_claimed_but_unpacked_export(tmp_path):
    out = tmp_path / "quant"
    _claim_autoround(out)
    _write_st(out, ["model.layers.0.self_attn.q_proj.weight"])  # plain weights only
    with pytest.raises(AqaError, match="0 packed layers"):
        QuantSanityHook().after_quantize(None, None, out)


def test_quant_sanity_passes_with_packed_weights(tmp_path):
    out = tmp_path / "quant"
    _claim_autoround(out)
    _write_st(out, ["model.layers.0.self_attn.q_proj.qweight"])
    QuantSanityHook().after_quantize(None, None, out)  # no raise


def test_quant_sanity_passes_without_quantization_config(tmp_path):
    out = tmp_path / "quant"
    _write_st(out, ["model.layers.0.weight"])
    (out / "config.json").write_text("{}")  # config without quantization_config
    QuantSanityHook().after_quantize(None, None, out)
    (out / "config.json").unlink()
    QuantSanityHook().after_quantize(None, None, out)  # no config at all


def test_quant_sanity_diffusion_component_layout(tmp_path):
    out = tmp_path / "quant"
    _claim_autoround(out / "transformer")
    _write_st(out / "transformer", ["blocks.0.attn.to_q.weight"])  # unpacked!
    _write_st(out / "text_encoder", ["model.layers.0.mlp.weight"])
    with pytest.raises(AqaError, match="0 packed layers"):
        QuantSanityHook().after_quantize(None, None, out)
    _write_st(out / "transformer", ["blocks.0.attn.to_q.qweight"])
    QuantSanityHook().after_quantize(None, None, out)


def test_quant_sanity_auto_round_format_claims_via_quant_cfg(tmp_path):
    from aqa.config import QuantConfig

    out = tmp_path / "quant"
    out.mkdir()
    # quantization_config without an explicit auto-round quant_method
    (out / "config.json").write_text(json.dumps({"quantization_config": {"bits": 4}}))
    _write_st(out, ["model.layers.0.weight"])
    cfg = QuantConfig(export_format="auto_round")
    with pytest.raises(AqaError, match="0 packed layers"):
        QuantSanityHook().after_quantize(None, cfg, out)
    # a different format does not claim auto-round -> passes
    QuantSanityHook().after_quantize(None, QuantConfig(export_format="llm_compressor"), out)


# ── eval_sanity ──────────────────────────────────────────────────────────────
def test_eval_sanity_zero_score_fails_normal_passes(tmp_path):
    hook = EvalSanityHook()
    zero = BenchmarkResult(name="piqa", score=0.0, metric="acc")
    with pytest.raises(AqaError, match="piqa.*acc=0"):
        hook.after_benchmark(None, None, zero, tmp_path)
    ok = BenchmarkResult(name="piqa", score=0.5, metric="acc")
    hook.after_benchmark(None, None, ok, tmp_path)
    # harness failures are surfaced by the runner, not re-failed by the hook
    hook.after_benchmark(None, None, RuntimeError("boom"), tmp_path)


# ── runner semantics ─────────────────────────────────────────────────────────
def test_hook_crash_is_logged_not_fatal():
    calls = []

    class _Crash:
        name = "crash"

        def after_benchmark(self, run_cfg, **ctx):
            raise RuntimeError("kaput")

    class _Good:
        name = "good"

        def after_benchmark(self, run_cfg, **ctx):
            calls.append("good")

    run_hook_chain("after_benchmark", [_Crash(), _Good()], None)
    assert calls == ["good"]  # the crash did not stop the chain


def test_hook_ordering_and_validator_propagation():
    calls = []

    class _Rec:
        def __init__(self, name):
            self.name = name

        def after_quantize(self, run_cfg, **ctx):
            calls.append(self.name)

    class _Veto:
        name = "veto"

        def after_quantize(self, run_cfg, **ctx):
            raise AqaError("vetoed")

    run_hook_chain("after_quantize", [_Rec("a"), _Rec("b")], None)
    assert calls == ["a", "b"]
    with pytest.raises(AqaError, match="vetoed"):
        run_hook_chain("after_quantize", [_Veto(), _Rec("c")], None)
    assert calls == ["a", "b"]  # the validator aborted the chain


def test_run_hooks_respects_config(tmp_path):
    cfg = parse_run_config({"model": "org/m", "hooks": []})
    out = tmp_path / "quant"
    _claim_autoround(out)
    _write_st(out, ["model.layers.0.weight"])
    run_hooks("after_quantize", cfg, quant_cfg=None, output_dir=out)  # all off: no raise
    cfg = parse_run_config({"model": "org/m"})  # defaults on
    with pytest.raises(AqaError, match="0 packed layers"):
        run_hooks("after_quantize", cfg, quant_cfg=None, output_dir=out)


# ── quantize / pipeline integration ──────────────────────────────────────────
def test_run_quantization_invokes_after_quantize_hooks(tmp_path, monkeypatch):
    out = tmp_path / "quant"

    def _fake_autoround(cmd, output_dir):
        _claim_autoround(output_dir)
        _write_st(output_dir, ["model.layers.0.weight"])  # claimed but unpacked
        return output_dir

    monkeypatch.setattr("aqa.quantize._run_autoround", _fake_autoround)
    from aqa.config import QuantConfig

    cfg = parse_run_config({"model": "org/m"})
    with pytest.raises(AqaError, match="0 packed layers"):
        run_quantization("org/m", QuantConfig(), out, hooks=resolve_hooks(cfg))
    # hooks off -> no validation (unchanged legacy behavior)
    off = tmp_path / "quant_off"
    monkeypatch.setattr("aqa.quantize._run_autoround",
                        lambda cmd, output_dir: output_dir)
    assert run_quantization("org/m", QuantConfig(), off, hooks=()) == off


def test_pipeline_quantize_phase_fails_on_quant_sanity(tmp_path, monkeypatch):
    def _fake_autoround(cmd, output_dir):
        _claim_autoround(output_dir)
        _write_st(output_dir, ["model.layers.0.weight"])
        return output_dir

    monkeypatch.setattr("aqa.quantize._run_autoround", _fake_autoround)
    cfg = parse_run_config(
        {"model": "org/m", "quant": {"scheme": "MXFP4"}, "agent": {"fix_loop": False}})
    with pytest.raises(AqaError, match="0 packed layers"):
        P._run_quantize(cfg, tmp_path / "quant", tmp_path / "out")
    # the same export passes when hooks are disabled
    cfg_off = parse_run_config(
        {"model": "org/m", "quant": {"scheme": "MXFP4"},
         "agent": {"fix_loop": False}, "hooks": []})
    P._run_quantize(cfg_off, tmp_path / "quant2", tmp_path / "out2")


def test_pipeline_evaluate_phase_runs_eval_sanity(tmp_path, monkeypatch):
    def _result(scores):
        res = EvalResult(model="m", backend="hf")
        for name, score in scores.items():
            res.add(BenchmarkResult(name=name, score=score, metric="acc"))
        return res

    captured = {}

    def _fake_fix_loop(phase_name, phase, **kw):
        pr = phase()
        captured["phase_result"] = pr
        return FixResult(ok=pr.ok, attempts=0, tier_index=0, reason=pr.log_text)

    monkeypatch.setattr(P, "run_fix_loop", _fake_fix_loop)
    cfg = parse_run_config({"model": "org/m"})

    # a zero benchmark score -> the eval_sanity verdict fails the phase
    monkeypatch.setattr(P, "run_evaluation",
                        lambda *a, **k: _result({"piqa": 0.0, "mmlu": 0.6}))
    res = P._run_evaluate(cfg, "m", tmp_path / "eval", tmp_path / "out", 1)
    pr = captured["phase_result"]
    assert not pr.ok
    assert "piqa" in pr.log_text and "acc=0" in pr.log_text
    # an unresolved evaluate phase soft-fails to the last result (existing behavior)
    assert res.benchmarks["piqa"].score == 0.0

    # normal scores -> phase ok, result returned
    monkeypatch.setattr(P, "run_evaluation", lambda *a, **k: _result({"piqa": 0.5}))
    res = P._run_evaluate(cfg, "m", tmp_path / "eval2", tmp_path / "out2", 1)
    assert res.benchmarks["piqa"].score == 0.5
    assert captured["phase_result"].ok


def test_pipeline_evaluate_hooks_disabled_keeps_inline_check_only(tmp_path, monkeypatch):
    """hooks=[] -> only the aggregate all-zero heuristic remains (legacy behavior)."""

    def _result(scores):
        res = EvalResult(model="m", backend="hf")
        for name, score in scores.items():
            res.add(BenchmarkResult(name=name, score=score, metric="acc"))
        return res

    captured = {}

    def _fake_fix_loop(phase_name, phase, **kw):
        pr = phase()
        captured["phase_result"] = pr
        return FixResult(ok=pr.ok, attempts=0, tier_index=0, reason=pr.log_text)

    monkeypatch.setattr(P, "run_fix_loop", _fake_fix_loop)
    monkeypatch.setattr(P, "run_evaluation",
                        lambda *a, **k: _result({"piqa": 0.0, "mmlu": 0.6}))
    cfg = parse_run_config({"model": "org/m", "hooks": []})
    res = P._run_evaluate(cfg, "m", tmp_path / "eval", tmp_path / "out", 1)
    # partial zero passes with hooks off (inline check only fires on ALL zero)
    assert res.benchmarks["piqa"].score == 0.0
    assert captured["phase_result"].ok

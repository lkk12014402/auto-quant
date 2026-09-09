"""Tests for sweep candidate generation + CLI plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqa.agent.optimize import ModelAnalysis
from aqa.config import QuantConfig
from aqa.quantize.candidates import (
    apply_candidate,
    builtin_candidates,
    custom_candidates,
    detect_module_families,
    resolve_candidates,
)

MOE_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.shared_expert.gate_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def _analysis(names: list[str], *, moe: bool = True) -> ModelAnalysis:
    return ModelAnalysis(
        model_id="org/m", is_moe=moe, num_experts=8 if moe else None,
        has_router=moe, has_shared_experts=moe, has_self_attn=True,
        sample_modules=names,
    )


def test_detect_module_families():
    fams = detect_module_families(MOE_NAMES)
    assert "self_attn" in fams and "shared_expert" in fams
    assert "linear_attn" not in fams and "visual" not in fams


def test_builtin_moe_candidate_set():
    cands = builtin_candidates(_analysis(MOE_NAMES))
    by_name = {c.name: c for c in cands}
    assert set(by_name) == {"conservative", "aggressive", "mixed", "tuning"}
    con = by_name["conservative"].overrides
    assert con["method"] == "model_free" and "self_attn" in con["ignore_layers"]
    agg = by_name["aggressive"].overrides
    # aggressive drops shared_expert from the ignore list (it gets quantized)
    assert "shared_expert" not in agg["ignore_layers"]
    assert "lm_head" in agg["ignore_layers"]
    mix = by_name["mixed"].overrides
    assert mix["scheme"] == "MXFP8"
    assert mix["layer_config"] == {"mlp.experts": {"bits": 4, "data_type": "mx_fp"}}
    tun = by_name["tuning"].overrides
    assert tun["method"] == "autoround" and tun["iters"] == 200


def test_builtin_dense_mixed_degrades_to_mxfp8():
    names = [n for n in MOE_NAMES if "experts" not in n and "shared_expert" not in n]
    cands = builtin_candidates(_analysis(names, moe=False))
    mix = next(c for c in cands if c.name == "mixed")
    assert mix.overrides["scheme"] == "MXFP8" and "layer_config" not in mix.overrides


def test_custom_candidates_and_unknown_fields():
    cands = custom_candidates([
        {"name": "mine", "scheme": "W4A16", "method": "autoround", "iters": 0,
         "description": "我的自定义档"}])
    assert cands[0].name == "mine" and cands[0].overrides["scheme"] == "W4A16"
    assert cands[0].rationale == "我的自定义档"
    with pytest.raises(ValueError, match="unknown QuantConfig fields"):
        custom_candidates([{"name": "bad", "bogus_field": 1}])


def test_resolve_prefers_custom():
    a = _analysis(MOE_NAMES)
    qc = QuantConfig(recipe_sweep=True, recipe_sweep_candidates=[
        {"name": "only", "scheme": "MXFP4"}])
    cands = resolve_candidates(a, qc)
    assert [c.name for c in cands] == ["only"]


def test_apply_candidate_patches_config():
    base = QuantConfig(scheme="MXFP4", recipe_sweep=True)
    cand = builtin_candidates(_analysis(MOE_NAMES))[0]
    eff = apply_candidate(base, cand)
    assert eff.method == "model_free"
    assert eff.ignore_layers == cand.overrides["ignore_layers"]


def test_sweep_schema_validation():
    with pytest.raises(ValueError, match="recipe_sweep"):
        QuantConfig(recipe_sweep_candidates=[{"name": "x"}])
    ok = QuantConfig(recipe_sweep=True, recipe_sweep_candidates=[{"name": "x"}])
    assert ok.recipe_sweep


def test_sweep_dry_run_cli(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from aqa.cli import main

    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 8}')
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.layers.0.self_attn.q_proj.weight": "x",'
        ' "model.layers.0.mlp.experts.0.gate_proj.weight": "x"}}')
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        f"model: {tmp_path}\nquant:\n  scheme: MXFP4\n  recipe_sweep: true\n"
        f"eval:\n  benchmarks: [piqa]\noutput:\n  dir: {tmp_path}/out\n")
    result = CliRunner().invoke(main, ["sweep", str(yaml_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    for name in ("conservative", "aggressive", "mixed", "tuning"):
        assert f"== {name}:" in result.output
    assert "auto-round" in result.output


def test_custom_candidate_eval_model_args():
    cands = custom_candidates([
        {"name": "fp8kv", "scheme": "MXFP8", "method": "autoround", "iters": 0,
         "static_kv_fp8": True, "eval_model_args": {"kv_cache_dtype": "fp8"}}])
    assert cands[0].eval_model_args == {"kv_cache_dtype": "fp8"}
    # eval_model_args is not a QuantConfig field and must not leak into overrides
    assert "eval_model_args" not in cands[0].overrides


def test_sweep_level_sink_dispatch(tmp_path, monkeypatch):
    """sweep 结束时把 base 目录（含 sweep_report.json）单独 dispatch 一次。"""
    from click.testing import CliRunner

    import aqa.cli as cli_mod
    from aqa.cli import main

    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 8}')
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.layers.0.self_attn.q_proj.weight": "x"}}')
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        f"model: {tmp_path}\nquant:\n  scheme: MXFP4\n  recipe_sweep: true\n"
        f"  recipe_sweep_candidates: [{{name: only, scheme: MXFP4}}]\n"
        f"eval:\n  benchmarks: [piqa]\noutput:\n  dir: {tmp_path}/out\n")

    dispatched = []
    monkeypatch.setattr("aqa.pipeline._dispatch_sinks",
                        lambda cfg, d, meta: dispatched.append((str(d), meta.artifact_name)))

    class _FakeResult:
        class state:
            value = "done"
        output_dir = tmp_path / "out" / "only"

    monkeypatch.setattr("aqa.cli.EXECUTORS", type("E", (), {
        "create": staticmethod(lambda *a, **k: type("X", (), {
            "run": lambda self, c, **kw: _FakeResult()})())}))

    r = CliRunner().invoke(main, ["sweep", str(yaml_path)])
    assert r.exit_code == 0, r.output
    assert dispatched and dispatched[0][0] == str(tmp_path / "out")
    assert dispatched[0][1].endswith("AutoRound-sweep")

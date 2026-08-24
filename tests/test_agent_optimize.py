"""Tests for agent-optimize command extraction and safety validation."""

from __future__ import annotations

from aqa.agent.optimize import (
    ModelAnalysis,
    ensure_model_name,
    extract_command,
    run_agent_optimize,
    validate_command,
)


def _dense() -> ModelAnalysis:
    return ModelAnalysis(model_id="org/dense", is_moe=False)


def _moe() -> ModelAnalysis:
    return ModelAnalysis(model_id="org/moe", is_moe=True, has_router=True,
                         router_names=["mlp.gate"])


def test_extract_command_from_code_block():
    log = (
        "here is my recommendation\n"
        "```bash\n"
        "auto-round --model_name org/dense --scheme W4A16 --output_dir /out\n"
        "```\n"
        "done"
    )
    assert extract_command(log) == "auto-round --model_name org/dense --scheme W4A16 --output_dir /out"


def test_extract_command_none():
    assert extract_command("no command here") is None


def test_ensure_model_name_from_positional():
    cmd = "auto-round org/dense --scheme W4A16"
    out = ensure_model_name(cmd, "org/dense")
    assert "--model_name org/dense" in out
    assert out.count("org/dense") == 1


def test_ensure_model_name_drops_valueless_flag():
    cmd = "auto-round --model_name --model_free org/dense --scheme W4A16"
    out = ensure_model_name(cmd, "org/dense")
    assert "--model_name org/dense" in out
    assert "--model_free" in out


def test_validate_ok_dense():
    cmd = "auto-round --model_name org/dense --scheme W4A16 --format auto_round --output_dir /out"
    ok, reason = validate_command(cmd, _dense(), "/out")
    assert ok, reason


def test_validate_wrong_output_dir():
    cmd = "auto-round --model_name org/dense --output_dir /wrong"
    ok, reason = validate_command(cmd, _dense(), "/out")
    assert not ok and "output_dir" in reason


def test_validate_rejects_expert_protection_on_moe():
    cmd = ("auto-round --model_name org/moe --output_dir /out "
           "--fp_layers experts")
    ok, reason = validate_command(cmd, _moe(), "/out")
    assert not ok and "expert" in reason.lower()


def test_validate_moe_requires_router_protection():
    cmd = ("auto-round --model_name org/moe --output_dir /out "
           "--ignore_layers lm_head")
    ok, reason = validate_command(cmd, _moe(), "/out")
    assert not ok and "router" in reason.lower()


def test_validate_moe_router_protection_ok():
    cmd = ("auto-round --model_name org/moe --output_dir /out "
           "--ignore_layers mlp.gate,router --format auto_round")
    ok, reason = validate_command(cmd, _moe(), "/out")
    assert ok, reason


def test_validate_moe_experts_reenabled_via_layer_config():
    cmd = ("auto-round --model_name org/moe --output_dir /out "
           "--fp_layers experts --layer_config mlp.experts:bits=4")
    ok, reason = validate_command(cmd, _moe(), "/out")
    assert ok, reason


_GOOD_CMD = "auto-round --model_name org/dense --scheme W4A16 --format auto_round --output_dir /out"
_AGENT_LOG = f"Analysis complete.\n```bash\n{_GOOD_CMD}\n```\n"


def _patch_analysis(monkeypatch):
    monkeypatch.setattr(
        "aqa.agent.optimize.analyze_model",
        lambda *a, **k: ModelAnalysis(model_id="org/dense", is_moe=False),
    )


def test_run_agent_optimize_str_runner(monkeypatch, tmp_path):
    _patch_analysis(monkeypatch)
    cmd = run_agent_optimize(
        "org/dense", scheme="W4A16", method="auto_round", export_format="auto_round",
        output_dir="/out", tier="low", run_agent=lambda *a, **k: _AGENT_LOG,
        log_dir=tmp_path,
    )
    assert cmd == _GOOD_CMD


def test_run_agent_optimize_accepts_agent_outcome(monkeypatch, tmp_path):
    """Regression: a shared runner returning a structured AgentOutcome must not
    make command extraction blow up (it previously raised a TypeError and the
    recommendation was silently discarded)."""
    from aqa.agent.fix_loop import AgentOutcome

    _patch_analysis(monkeypatch)
    outcome = AgentOutcome(text=_AGENT_LOG, rc=0, timed_out=False)
    cmd = run_agent_optimize(
        "org/dense", scheme="W4A16", method="auto_round", export_format="auto_round",
        output_dir="/out", tier="low", run_agent=lambda *a, **k: outcome,
        log_dir=tmp_path,
    )
    assert cmd == _GOOD_CMD


def test_extract_command_from_autoround_cmd_block():
    log = (
        "## 6. Recommended command\n"
        "```autoround-cmd\n"
        "auto-round --model_name org/moe --scheme MXFP8 --output_dir /out\n"
        "```\n"
    )
    assert extract_command(log) == "auto-round --model_name org/moe --scheme MXFP8 --output_dir /out"


def test_load_recipe_knowledge_ships():
    from aqa.agent.optimize import load_recipe_knowledge

    text = load_recipe_knowledge()
    assert "MXFP4" in text and "MoE" in text


def test_build_optimize_prompt_embeds_recipe_knowledge_and_structure():
    from aqa.agent.optimize import build_optimize_prompt

    prompt = build_optimize_prompt(
        _dense(), scheme="MXFP8", method="MODEL_FREE", export_format="auto_round",
        output_dir="/out", recipe_knowledge="RULE: never quantize router",
    )
    assert "RULE: never quantize router" in prompt
    assert "autoround-cmd" in prompt
    assert "recipe document" in prompt


def test_run_agent_optimize_saves_recipe_doc(monkeypatch, tmp_path):
    _patch_analysis(monkeypatch)
    doc = f"# org/dense Recipe\n\n## 6.\n```autoround-cmd\n{_GOOD_CMD}\n```\n"
    cmd = run_agent_optimize(
        "org/dense", scheme="W4A16", method="auto_round", export_format="auto_round",
        output_dir="/out", tier="low", run_agent=lambda *a, **k: doc,
        recipe_knowledge="", log_dir=tmp_path,
    )
    assert cmd == _GOOD_CMD
    saved = (tmp_path / "quantization_recipe.md").read_text()
    assert "# org/dense Recipe" in saved

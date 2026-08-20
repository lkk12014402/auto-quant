"""Tests for agent-optimize command extraction and safety validation."""

from __future__ import annotations

from aqa.agent.optimize import (
    ModelAnalysis,
    ensure_model_name,
    extract_command,
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

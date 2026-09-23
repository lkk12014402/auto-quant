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


def test_load_recipe_knowledge_with_diffusion():
    from aqa.agent.optimize import load_recipe_knowledge

    base = load_recipe_knowledge()
    full = load_recipe_knowledge(include_diffusion=True)
    assert len(full) > len(base)
    assert "block_name_to_quantize" in full and "coco2014" in full
    assert "block_name_to_quantize" not in base


def test_analyze_model_diffusion_branch(monkeypatch):
    """Diffusion repos have no root config.json: analyze_model must detect
    model_index.json and fall back to the transformer/ subfolder."""
    import aqa.agent.optimize as opt

    def fake_config(model_id, revision, token, filename="config.json"):
        if filename == "config.json":
            raise RuntimeError("not found")
        assert filename == "transformer/config.json"
        return {"model_type": "flux", "num_layers": 19}

    monkeypatch.setattr(opt, "_hf_config", fake_config)
    monkeypatch.setattr(opt, "_has_model_index", lambda *a, **k: True)
    monkeypatch.setattr(opt, "_tensor_names", lambda *a, **k: [])
    res = opt.analyze_model("org/flux-like")
    assert res.ok and res.is_diffusion
    assert res.model_type == "flux" and res.num_layers == 19


def test_analyze_model_plain_failure_without_model_index(monkeypatch):
    import aqa.agent.optimize as opt

    def fake_config(model_id, revision, token, filename="config.json"):
        raise RuntimeError("not found")

    monkeypatch.setattr(opt, "_hf_config", fake_config)
    monkeypatch.setattr(opt, "_has_model_index", lambda *a, **k: False)
    res = opt.analyze_model("org/broken")
    assert not res.ok and not res.is_diffusion


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


# ── operator-fixed quant config precedence ────────────────────────────────────
def _quant(**kw):
    from aqa.config import QuantConfig
    return QuantConfig(**kw)


def test_build_quant_constraints_only_explicit_fields():
    from aqa.agent.optimize import build_quant_constraints

    # Defaults are NOT constrained; only fields the user actually set are.
    cons = build_quant_constraints(_quant(scheme="MXFP8", method="autoround", iters=0))
    flags = {c.flag for c in cons}
    assert "--scheme" in flags and "--iters" in flags
    assert "--model_free" in flags          # method explicitly set -> absent-constraint
    # nsamples / ignore_layers / layer_config were never set -> absent from constraints
    assert "--nsamples" not in flags and "--ignore_layers" not in flags


def test_build_quant_constraints_model_free_present_vs_absent():
    from aqa.agent.optimize import build_quant_constraints

    mf = build_quant_constraints(_quant(method="model_free"))
    al = build_quant_constraints(_quant(method="autoround"))
    assert any(c.flag == "--model_free" and c.kind == "present" for c in mf)
    assert any(c.flag == "--model_free" and c.kind == "absent" for c in al)


def test_validate_rejects_model_free_when_operator_forbids_it():
    # method=autoround => command must NOT pass --model_free (the gpt-oss trap).
    from aqa.agent.optimize import build_quant_constraints

    cons = build_quant_constraints(_quant(scheme="MXFP8", method="autoround"))
    bad = "auto-round --model_name org/dense --scheme MXFP8 --model_free --output_dir /out"
    ok, reason = validate_command(bad, _dense(), "/out", cons)
    assert not ok and "operator-fixed" in reason and "model_free" in reason.lower()

    good = "auto-round --model_name org/dense --scheme MXFP8 --iters 0 --output_dir /out"
    ok, reason = validate_command(good, _dense(), "/out", cons)
    assert ok, reason


def test_validate_rejects_scheme_and_iters_mismatch():
    from aqa.agent.optimize import build_quant_constraints

    cons = build_quant_constraints(_quant(scheme="MXFP8", method="autoround", iters=200))
    wrong_iters = "auto-round --model_name org/dense --scheme MXFP8 --iters 0 --output_dir /out"
    ok, reason = validate_command(wrong_iters, _dense(), "/out", cons)
    assert not ok and "--iters" in reason

    wrong_scheme = "auto-round --model_name org/dense --scheme MXFP4 --iters 200 --output_dir /out"
    ok, reason = validate_command(wrong_scheme, _dense(), "/out", cons)
    assert not ok and "scheme" in reason


def test_validate_ignore_layers_subset_and_layer_config_keys():
    from aqa.agent.optimize import build_quant_constraints

    cons = build_quant_constraints(_quant(
        method="autoround",
        ignore_layers="lm_head,embed_tokens,mlp.router",
        layer_config={"mlp.experts": {"bits": 4}},
    ))
    # Command missing one required ignore token AND the layer_config key -> reject.
    missing = ("auto-round --model_name org/dense --ignore_layers lm_head,embed_tokens "
               "--output_dir /out")
    ok, reason = validate_command(missing, _dense(), "/out", cons)
    assert not ok

    full = ('auto-round --model_name org/dense '
            '--ignore_layers lm_head,embed_tokens,mlp.router '
            '--layer_config {"mlp.experts":{"bits":4}} --output_dir /out')
    ok, reason = validate_command(full, _dense(), "/out", cons)
    assert ok, reason


def test_build_optimize_prompt_renders_hard_constraints():
    from aqa.agent.optimize import build_optimize_prompt, build_quant_constraints

    cons = build_quant_constraints(_quant(scheme="MXFP8", method="autoround", iters=0))
    prompt = build_optimize_prompt(
        _dense(), scheme="MXFP8", method="autoround", export_format="auto_round",
        output_dir="/out", constraints=cons,
    )
    assert "HARD CONSTRAINTS" in prompt
    assert "do NOT pass `--model_free`" in prompt
    assert "advisory only" in prompt


def test_run_agent_optimize_rejects_constraint_violating_command(monkeypatch, tmp_path):
    from aqa.agent.optimize import build_quant_constraints

    _patch_analysis(monkeypatch)
    # Agent ignores the operator and emits --model_free anyway -> rejected -> None
    # (caller falls back to the deterministic builder, which honors the config).
    bad = ("auto-round --model_name org/dense --scheme W4A16 --model_free "
           "--format auto_round --output_dir /out")
    cmd = run_agent_optimize(
        "org/dense", scheme="W4A16", method="autoround", export_format="auto_round",
        output_dir="/out", tier="low",
        run_agent=lambda *a, **k: f"```bash\n{bad}\n```",
        constraints=build_quant_constraints(_quant(method="autoround")),
        recipe_knowledge="", log_dir=tmp_path,
    )
    assert cmd is None



# ── diffusion (DiT) optimize branch ─────────────────────────────────────────
def test_build_optimize_prompt_diffusion_branch():
    from aqa.agent.optimize import build_optimize_prompt

    dit = ModelAnalysis(model_id="org/flux", is_diffusion=True)
    prompt = build_optimize_prompt(
        dit, scheme="W4A16", method="autoround", export_format="auto_round",
        output_dir="/out",
    )
    assert "diffusion (DiT) pipeline" in prompt
    assert "coco2014" in prompt          # DiT CLI examples
    assert "Example A -- MoE" not in prompt  # LLM examples not used
    assert "PR #5544" in prompt          # DiT hard requirements
    assert "is_diffusion: True" in prompt

    dense = ModelAnalysis(model_id="org/dense")
    prompt_llm = build_optimize_prompt(
        dense, scheme="W4A16", method="autoround", export_format="auto_round",
        output_dir="/out",
    )
    assert "Example A -- MoE" in prompt_llm
    assert "PR #5544" not in prompt_llm


def test_run_agent_optimize_loads_diffusion_knowledge_for_dit(monkeypatch, tmp_path):
    import aqa.agent.optimize as opt

    monkeypatch.setattr(opt, "analyze_model",
                        lambda *a, **k: ModelAnalysis(model_id="org/flux", is_diffusion=True))
    seen = {}

    def fake_load(*, include_diffusion=False):
        seen["include_diffusion"] = include_diffusion
        return ""

    monkeypatch.setattr(opt, "load_recipe_knowledge", fake_load)
    cmd = run_agent_optimize(
        "org/flux", scheme="W4A16", method="autoround", export_format="auto_round",
        output_dir="/out", tier="low", run_agent=lambda *a, **k: _AGENT_LOG,
        log_dir=tmp_path,
    )
    assert seen["include_diffusion"] is True
    assert cmd == _GOOD_CMD

"""Recipe KB (RFC §3.4): hot-start lookup relaxation + strict pin-back."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aqa.agent.recipe_kb import RecipeKB


def _write(root: Path, key: str, recipe: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{key}.json").write_text(json.dumps(recipe), encoding="utf-8")


@pytest.fixture()
def kb(tmp_path: Path) -> RecipeKB:
    curated = tmp_path / "quant"
    auto = curated / "auto"
    # Curated recipe keyed by HF id, free-form curated shape.
    _write(curated, "Qwen__Qwen3.8-27B", {
        "model": "Qwen/Qwen3.8-27B",
        "architecture": "qwen3_5 (Qwen3_5ForConditionalGeneration)",
        "ignore_layers": "visual.*",
        "ignore_layers_rationale": "keep the vision tower in bf16",
        "schemes": {
            "MXFP8": {"model_free_rtn": {"command": "auto-round --scheme MXFP8 ..."}},
            "MXFP4": {"command": "auto-round --scheme MXFP4 ...", "risk": "lm_head sensitive"},
        },
    })
    # A different model in the SAME family (qwen3_5) for family relaxation.
    _write(curated, "Qwen__Qwen3.8-Flash-Next", {
        "model": "Qwen/Qwen3.8-Flash-Next",
        "model_type": "qwen3_5",
        "schemes": {"MXFP8": {"command": "auto-round flash ..."}},
    })
    return RecipeKB(curated_root=curated, auto_root=auto)


def test_lookup_exact_model_and_scheme(kb: RecipeKB) -> None:
    hit = kb.lookup("/home/u/models/Qwen3.8-27B", scheme="MXFP8")
    assert hit is not None
    assert hit.match_level == 4
    assert hit.key == "Qwen/Qwen3.8-27B"
    assert hit.scheme == "MXFP8"
    assert hit.command == "auto-round --scheme MXFP8 ..."
    assert hit.source == "curated"


def test_lookup_exact_model_relaxed_scheme(kb: RecipeKB) -> None:
    hit = kb.lookup("Qwen/Qwen3.8-27B", scheme="NVFP4")
    assert hit is not None
    assert hit.match_level == 3  # model id matched, scheme relaxed
    assert hit.scheme in {"MXFP8", "MXFP4"}


def test_lookup_family_relaxation(kb: RecipeKB) -> None:
    # Unknown model id, but same family (qwen3_5) via model_type.
    hit = kb.lookup("Qwen/Qwen3.8-27B-Brand-New", scheme="MXFP8",
                    model_type="qwen3_5")
    assert hit is not None
    assert hit.match_level == 2  # family + scheme
    assert hit.match_reason.startswith("same family")


def test_lookup_miss_returns_none(kb: RecipeKB) -> None:
    assert kb.lookup("meta-llama/Llama-3-8B", scheme="MXFP8",
                     model_type="llama") is None


def test_curated_outranks_auto_same_level(tmp_path: Path) -> None:
    curated = tmp_path / "quant"
    auto = curated / "auto"
    _write(curated, "foo__Bar", {"model": "foo/Bar",
                                  "schemes": {"MXFP8": {"command": "CURATED"}}})
    _write(auto, "bar", {"model": "foo/Bar",
                         "schemes": {"MXFP8": {"command": "AUTO"}}})
    hit = RecipeKB(curated_root=curated, auto_root=auto).lookup("foo/Bar", scheme="MXFP8")
    assert hit is not None
    assert hit.source == "curated"
    assert hit.command == "CURATED"


def test_as_hot_start_context_is_advisory(kb: RecipeKB) -> None:
    hit = kb.lookup("Qwen/Qwen3.8-27B", scheme="MXFP8")
    text = RecipeKB.as_hot_start_context(hit)
    assert "ADVISORY" in text
    assert "RE-VALIDATE" in text
    assert "auto-round --scheme MXFP8" in text
    assert "visual.*" in text  # ignore_layers surfaced


def test_pin_refuses_non_go(kb: RecipeKB) -> None:
    path = kb.pin(model="Qwen/Qwen3.8-27B", scheme="MXFP8",
                  command="x", gate_verdict="NO-GO")
    assert path is None
    assert not (kb.auto_root).exists() or not list(kb.auto_root.glob("*.json"))


def test_pin_writes_to_auto_namespace_not_curated(kb: RecipeKB) -> None:
    curated_before = sorted(p.name for p in kb.curated_root.glob("*.json"))
    path = kb.pin(model="/home/u/models/Qwen3.8-27B", scheme="MXFP8",
                  command="auto-round validated", gate_verdict="GO",
                  result={"mean_score": 0.71}, source_run="run-1")
    assert path is not None
    assert Path(path).parent == kb.auto_root
    # Curated recipes untouched.
    assert sorted(p.name for p in kb.curated_root.glob("*.json")) == curated_before


def test_pin_merges_scheme_and_unions_lists(kb: RecipeKB) -> None:
    kb.pin(model="foo/New", scheme="MXFP8", command="cmd8", gate_verdict="GO",
           lessons=["a"], failures=["boom"])
    path = kb.pin(model="foo/New", scheme="MXFP4", command="cmd4", gate_verdict="GO",
                  lessons=["a", "b"], failures=["boom", "bang"])
    data = json.loads(Path(path).read_text())
    assert set(data["schemes"]) == {"MXFP8", "MXFP4"}
    assert data["schemes"]["MXFP8"]["command"] == "cmd8"
    assert data["schemes"]["MXFP4"]["command"] == "cmd4"
    assert data["lessons"] == ["a", "b"]
    assert data["failures"] == ["boom", "bang"]


def test_pinned_recipe_is_found_by_lookup(kb: RecipeKB) -> None:
    kb.pin(model="foo/New", scheme="MXFP8", command="cmd8", gate_verdict="GO")
    hit = kb.lookup("foo/New", scheme="MXFP8")
    assert hit is not None
    assert hit.source == "auto"
    assert hit.command == "cmd8"


# ── Hyperloom-borrowed: negative knowledge + staleness + session ledger ──────

def test_record_outcome_rejects_go(kb: RecipeKB) -> None:
    with pytest.raises(ValueError):
        kb.record_outcome(model="foo/New", scheme="MXFP4", outcome="GO")


def test_record_outcome_rejects_unknown_outcome(kb: RecipeKB) -> None:
    with pytest.raises(ValueError):
        kb.record_outcome(model="foo/New", scheme="MXFP4", outcome="MAYBE")


def test_record_outcome_writes_failure_without_validated_schemes(kb: RecipeKB) -> None:
    path = kb.record_outcome(
        model="foo/New", scheme="MXFP4", outcome="NO-GO",
        mean_before=0.80, mean_after=0.70,
        reasons=["gsm8k dropped 0.10 > gate 0.02"], source_run="run-9")
    assert path is not None
    data = json.loads(Path(path).read_text())
    # Negative knowledge: a failure is recorded, but NO validated scheme and no trust.
    assert "schemes" not in data or not data["schemes"]
    assert "confidence" not in data
    assert any("NO-GO" in f and "MXFP4" in f for f in data["failures"])
    assert data["sessions"][-1]["outcome"] == "NO-GO"
    assert data["sessions"][-1]["drop"] == pytest.approx(0.10, abs=1e-6)


def test_negative_knowledge_surfaces_in_hot_start(kb: RecipeKB) -> None:
    kb.record_outcome(model="foo/New", scheme="MXFP4", outcome="NO-GO",
                      mean_before=0.80, mean_after=0.70,
                      reasons=["gsm8k regressed"], source_run="run-9")
    hit = kb.lookup("foo/New", scheme="MXFP4")
    assert hit is not None
    # No validated command (negative-only row) but the failure is surfaced.
    assert hit.command is None
    assert hit.confidence == 0.0  # nothing validated yet
    text = RecipeKB.as_hot_start_context(hit)
    assert "past failures" in text
    assert "NO-GO" in text


def test_go_then_nogo_keeps_validated_scheme_and_adds_failure(kb: RecipeKB) -> None:
    kb.pin(model="foo/New", scheme="MXFP8", command="good", gate_verdict="GO",
           mean_before=0.80, mean_after=0.79)
    path = kb.record_outcome(model="foo/New", scheme="MXFP4", outcome="NO-GO",
                             mean_before=0.80, mean_after=0.60,
                             reasons=["MXFP4 too lossy"])
    data = json.loads(Path(path).read_text())
    # The GO scheme survives; the NO-GO is recorded as a failure + session row.
    assert data["schemes"]["MXFP8"]["command"] == "good"
    assert "MXFP4" not in data["schemes"]
    assert any("MXFP4" in f for f in data["failures"])
    outcomes = [s["outcome"] for s in data["sessions"]]
    assert outcomes == ["GO", "NO-GO"]


def test_pin_stores_fingerprint_and_confidence(kb: RecipeKB) -> None:
    fp = {"auto_round": "0.9", "vllm": "0.11", "accelerator": "cuda"}
    path = kb.pin(model="foo/New", scheme="MXFP8", command="cmd", gate_verdict="GO",
                  stack_fingerprint=fp, mean_before=0.80, mean_after=0.79)
    data = json.loads(Path(path).read_text())
    assert data["stack_fingerprint"] == fp
    assert data["authority"] == "experiential"
    assert data["confidence"] == 0.85  # first GO
    assert data["last_profiled"]


def test_confidence_grows_with_repeated_go(kb: RecipeKB) -> None:
    kb.pin(model="foo/New", scheme="MXFP8", command="c1", gate_verdict="GO")
    path = kb.pin(model="foo/New", scheme="MXFP8", command="c2", gate_verdict="GO")
    data = json.loads(Path(path).read_text())
    assert data["confidence"] == 0.90  # 0.85 + 0.05 for one prior GO


def test_lookup_flags_stale_recipe_and_lowers_confidence(kb: RecipeKB) -> None:
    kb.pin(model="foo/New", scheme="MXFP8", command="cmd", gate_verdict="GO",
           stack_fingerprint={"auto_round": "0.9", "vllm": "0.11"})
    # Same model, but the live stack has a newer vllm → stale.
    hit = kb.lookup("foo/New", scheme="MXFP8",
                    current_stack={"auto_round": "0.9", "vllm": "0.12"})
    assert hit is not None
    assert hit.stale is True
    assert hit.stale_tools == ["vllm"]
    assert hit.confidence == pytest.approx(0.85 * 0.4, abs=1e-6)
    text = RecipeKB.as_hot_start_context(hit)
    assert "STALE" in text and "vllm" in text


def test_lookup_not_stale_when_stack_matches(kb: RecipeKB) -> None:
    fp = {"auto_round": "0.9", "vllm": "0.11"}
    kb.pin(model="foo/New", scheme="MXFP8", command="cmd", gate_verdict="GO",
           stack_fingerprint=fp)
    hit = kb.lookup("foo/New", scheme="MXFP8", current_stack=dict(fp))
    assert hit is not None
    assert hit.stale is False
    assert hit.confidence == 0.85


def test_lookup_missing_probe_is_not_stale(kb: RecipeKB) -> None:
    # A tool absent from the current probe must NOT raise a false staleness alarm.
    kb.pin(model="foo/New", scheme="MXFP8", command="cmd", gate_verdict="GO",
           stack_fingerprint={"auto_round": "0.9", "vllm": "0.11"})
    hit = kb.lookup("foo/New", scheme="MXFP8", current_stack={"auto_round": "0.9"})
    assert hit is not None
    assert hit.stale is False


def test_session_ledger_is_bounded(kb: RecipeKB) -> None:
    from aqa.agent.recipe_kb import _MAX_SESSIONS
    path = None
    for i in range(_MAX_SESSIONS + 5):
        path = kb.record_outcome(model="foo/New", scheme="MXFP4", outcome="NO-GO",
                                 reasons=[f"attempt {i}"])
    data = json.loads(Path(path).read_text())
    assert len(data["sessions"]) == _MAX_SESSIONS


def test_from_config_honors_root_and_auto_subdir(tmp_path: Path) -> None:
    cfg = SimpleNamespace(root=str(tmp_path / "kb"), auto_subdir="machine")
    kb = RecipeKB.from_config(cfg)
    assert kb.curated_root == tmp_path / "kb"
    assert kb.auto_root == tmp_path / "kb" / "machine"


# ── pipeline wiring: hot-start lookup + pin-back helpers ─────────────────────

def _model_dir(tmp_path: Path, name: str = "M") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"model_type": "foo", "architectures": ["FooForCausalLM"]}))
    return d


def _quant_run_cfg(model: str, curated: Path, *, optimize: bool, pin: bool = False,
                   auto_sync: str = "off"):
    from aqa.config import (AgentConfig, ProviderConfig, QuantConfig,
                            RecipeKBConfig, RunConfig)
    return RunConfig(
        model=model,
        provider=ProviderConfig(type="quantize", quant=QuantConfig(
            scheme="MXFP8", method="autoround", ignore_layers="lm_head")),
        agent=AgentConfig(optimize=optimize, recipe_kb=RecipeKBConfig(
            root=str(curated), pin=pin, auto_sync=auto_sync)))


def test_lookup_helper_injects_recipe_and_writes_audit(tmp_path: Path) -> None:
    from aqa.pipeline import _recipe_kb_hot_start, _recipe_kb_lookup
    from aqa.session import Phase, open_session

    curated = tmp_path / "quant"
    _write(curated, "org__M", {"model": "org/M", "model_type": "foo",
                               "schemes": {"MXFP8": {"command": "auto-round CMD"}}})
    model_dir = _model_dir(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    run_cfg = _quant_run_cfg(str(model_dir), curated, optimize=True)
    session = open_session(out, run_cfg, "run-1", [Phase.QUANT])

    hit = _recipe_kb_lookup(run_cfg, out, session, "run-1")
    hot = _recipe_kb_hot_start(hit, run_cfg)

    assert hit is not None and hit.command == "auto-round CMD"
    assert "auto-round CMD" in hot and "ADVISORY" in hot
    rec = json.loads((out / "recipe_kb.json").read_text())
    assert rec["used_as_hot_start"] is True
    assert rec["hit"]["command"] == "auto-round CMD"
    # Decision is recorded on the session for provenance.
    assert any(d["phase"] == "ANALYZE" for d in session.state.decisions)


def test_lookup_helper_records_null_hit_on_miss(tmp_path: Path) -> None:
    from aqa.pipeline import _recipe_kb_lookup
    from aqa.session import Phase, open_session

    curated = tmp_path / "quant"
    curated.mkdir()
    model_dir = _model_dir(tmp_path, "Unseen")
    (model_dir / "config.json").write_text(json.dumps({"model_type": "other"}))
    out = tmp_path / "run"
    out.mkdir()
    run_cfg = _quant_run_cfg(str(model_dir), curated, optimize=True)
    session = open_session(out, run_cfg, "run-1", [Phase.QUANT])

    hit = _recipe_kb_lookup(run_cfg, out, session, "run-1")

    assert hit is None
    rec = json.loads((out / "recipe_kb.json").read_text())
    assert rec["hit"] is None and rec["used_as_hot_start"] is False


def test_lookup_runs_but_no_prompt_when_optimize_off(tmp_path: Path) -> None:
    """Read side is decoupled from optimize: lookup + audit still happen, but no
    hot-start prompt is rendered (nothing consumes it without agent-optimize)."""
    from aqa.pipeline import _recipe_kb_hot_start, _recipe_kb_lookup
    from aqa.session import Phase, open_session

    curated = tmp_path / "quant"
    _write(curated, "org__M", {"model": "org/M", "model_type": "foo",
                               "schemes": {"MXFP8": {"command": "c"}}})
    out = tmp_path / "run"
    out.mkdir()
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=False)
    session = open_session(out, run_cfg, "run-1", [Phase.QUANT])

    hit = _recipe_kb_lookup(run_cfg, out, session, "run-1")
    assert hit is not None                      # KB is read regardless of optimize
    assert (out / "recipe_kb.json").exists()     # audit written regardless of optimize
    rec = json.loads((out / "recipe_kb.json").read_text())
    assert rec["used_as_hot_start"] is False     # ... but not injected as a prompt
    assert _recipe_kb_hot_start(hit, run_cfg) == ""


def test_lookup_disabled_when_kb_disabled(tmp_path: Path) -> None:
    from aqa.config import (AgentConfig, ProviderConfig, QuantConfig,
                            RecipeKBConfig, RunConfig)
    from aqa.pipeline import _recipe_kb_lookup
    from aqa.session import Phase, open_session

    curated = tmp_path / "quant"
    _write(curated, "org__M", {"model": "org/M", "model_type": "foo",
                               "schemes": {"MXFP8": {"command": "c"}}})
    out = tmp_path / "run"
    out.mkdir()
    run_cfg = RunConfig(
        model=str(_model_dir(tmp_path)),
        provider=ProviderConfig(type="quantize", quant=QuantConfig(
            scheme="MXFP8", method="autoround", ignore_layers="lm_head")),
        agent=AgentConfig(recipe_kb=RecipeKBConfig(root=str(curated), enabled=False)))
    session = open_session(out, run_cfg, "run-1", [Phase.QUANT])

    assert _recipe_kb_lookup(run_cfg, out, session, "run-1") is None
    assert not (out / "recipe_kb.json").exists()


def test_pin_helper_writes_auto_recipe_on_go(tmp_path: Path) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    model_dir = _model_dir(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "auto-round VALIDATED", source="agent_optimized")
    run_cfg = _quant_run_cfg(str(model_dir), curated, optimize=True, pin=True)
    result = _eval_result("org/M", {"mmlu": 0.7})

    _recipe_kb_pin(run_cfg, out, result, "GO", "run-1")

    auto_file = curated / "auto" / "m.json"
    assert auto_file.is_file()
    data = json.loads(auto_file.read_text())
    assert data["schemes"]["MXFP8"]["command"] == "auto-round VALIDATED"
    assert data["schemes"]["MXFP8"]["gate"] == "GO"
    assert data["ignore_layers"] == "lm_head"


def test_pin_helper_noop_when_pin_disabled(tmp_path: Path) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "cmd", source="builder")
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=True, pin=False)

    _recipe_kb_pin(run_cfg, out, _eval_result("org/M", {"mmlu": 0.7}), "GO", "run-1")

    assert not (curated / "auto").exists()


def test_pin_helper_skips_record_without_gate(tmp_path: Path) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "cmd", source="builder")
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=True, pin=True)

    # No gate configured -> no GO/NO-GO verdict. Do not poison the KB with FAILED.
    _recipe_kb_pin(run_cfg, out, _eval_result("org/M", {"mmlu": 0.7}), None, "run-1")

    assert not (curated / "auto").exists()


def test_pin_helper_records_negative_knowledge_on_non_go(tmp_path: Path) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "cmd", source="agent_optimized")
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=True, pin=True)

    _recipe_kb_pin(run_cfg, out, _eval_result("org/M", {"mmlu": 0.7}), "NO-GO", "run-1",
                   baseline=_eval_result("org/M", {"mmlu": 0.8}),
                   gate_reasons=["mmlu dropped 0.10 > gate 0.02"])

    # Non-GO now records negative knowledge (what failed) WITHOUT validated schemes.
    row = curated / "auto" / "m.json"
    assert row.exists()
    data = json.loads(row.read_text())
    assert "schemes" not in data or not data["schemes"]
    assert "confidence" not in data
    assert any("NO-GO" in f for f in data["failures"])
    assert data["sessions"][-1]["outcome"] == "NO-GO"


def test_pin_helper_falls_back_to_run_dir_when_code_root_readonly(
        tmp_path: Path, monkeypatch) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "auto-round VALIDATED", source="builder")
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=True, pin=True)

    original_pin = RecipeKB.pin

    def _pin(self, **kwargs):
        if str(self.auto_root).startswith(str(curated)):
            raise OSError(30, "Read-only file system")
        return original_pin(self, **kwargs)

    monkeypatch.setattr(RecipeKB, "pin", _pin)
    _recipe_kb_pin(run_cfg, out, _eval_result("org/M", {"mmlu": 0.7}), "GO", "run-1")

    fallback = out / "recipe_kb" / "auto" / "m.json"
    assert fallback.is_file()
    record = json.loads((out / "recipe_kb.json").read_text())["record"]
    assert record["status"] == "fallback"
    assert record["path"] == str(fallback)


def _eval_result(model: str, scores: dict[str, float]):
    from aqa.eval.registry import BenchmarkResult, EvalResult
    r = EvalResult(model=model, backend="hf")
    for name, score in scores.items():
        r.add(BenchmarkResult(name=name, score=score, metric="acc", num_samples=10))
    return r



# ── read surface for external agents: entries_summary / show / CLI ───────────

def test_entries_summary(kb: RecipeKB) -> None:
    rows = kb.entries_summary()
    models = {r["model"] for r in rows}
    assert "Qwen/Qwen3.8-27B" in models
    row = next(r for r in rows if r["model"] == "Qwen/Qwen3.8-27B")
    assert row["source"] == "curated"
    assert set(row["schemes"]) >= {"MXFP8", "MXFP4"}
    assert row["path"].endswith("Qwen__Qwen3.8-27B.json")


def test_show_exact_model_by_path_slug(kb: RecipeKB) -> None:
    rec = kb.show("/home/u/models/Qwen3.8-27B")  # local path slugs to HF id
    assert rec is not None
    assert rec["model"] == "Qwen/Qwen3.8-27B"
    assert "schemes" in rec


def test_show_miss_returns_none(kb: RecipeKB) -> None:
    assert kb.show("meta-llama/Llama-3.1-8B") is None


def test_cli_recipe_surface(kb: RecipeKB, monkeypatch) -> None:
    import aqa.cli as cli
    from click.testing import CliRunner

    monkeypatch.setattr(cli, "_recipe_kb", lambda: kb)
    runner = CliRunner()

    r = runner.invoke(cli.main, ["recipe", "list"])
    assert r.exit_code == 0, r.output
    assert "Qwen/Qwen3.8-27B" in r.output

    r = runner.invoke(cli.main, ["recipe", "lookup", "Qwen/Qwen3.8-27B", "--scheme", "MXFP8"])
    assert r.exit_code == 0, r.output
    assert "Hot-start recipe" in r.output
    assert "MXFP8" in r.output

    r = runner.invoke(cli.main, ["recipe", "lookup", "Qwen/Qwen3.8-27B", "--scheme", "MXFP8", "--json"])
    assert r.exit_code == 0, r.output
    assert '"match_level": 4' in r.output

    r = runner.invoke(cli.main, ["recipe", "show", "Qwen/Qwen3.8-27B"])
    assert r.exit_code == 0, r.output
    assert '"model": "Qwen/Qwen3.8-27B"' in r.output

    r = runner.invoke(cli.main, ["recipe", "lookup", "nope/nope"])
    assert r.exit_code != 0  # cold start exits non-zero


# ── KB -> eval bridge: structured eval_parity auto-apply ─────────────────────

def _hit(eval_parity):
    from aqa.agent.recipe_kb import RecipeHit
    return RecipeHit(
        key="org/M", match_level=4, match_reason="exact", source="curated",
        source_path="/x.json", scheme="MXFP8", command="c", ignore_layers=None,
        eval_parity=eval_parity)


def _eval_run_cfg(**eval_kwargs):
    from aqa.config import EvalConfig, ProviderConfig, QuantConfig, RunConfig
    return RunConfig(
        model="org/M",
        provider=ProviderConfig(type="quantize", quant=QuantConfig(
            scheme="MXFP8", method="autoround", ignore_layers="lm_head")),
        eval=EvalConfig(**eval_kwargs))


def test_bridge_applies_eval_parity_when_user_default(tmp_path: Path) -> None:
    from aqa.pipeline import _apply_recipe_eval_parity
    from aqa.session import Phase, open_session

    run_cfg = _eval_run_cfg(benchmarks=["gsm8k"], generative_tasks=["gsm8k"])
    out = tmp_path / "run"; out.mkdir()
    session = open_session(out, run_cfg, "r", [Phase.QUANT])
    hit = _hit({"thinking": "off",
                "thinking_off_model_args": {"enable_thinking": False,
                                            "reasoning_parser": "qwen3"}})

    new_cfg = _apply_recipe_eval_parity(run_cfg, hit, session)

    assert new_cfg.eval.thinking == "off"
    assert new_cfg.eval.thinking_off_model_args["reasoning_parser"] == "qwen3"
    assert any(d["what"] == "applied recipe eval parity"
               for d in session.state.decisions)


def test_bridge_user_thinking_wins(tmp_path: Path) -> None:
    from aqa.pipeline import _apply_recipe_eval_parity
    from aqa.session import Phase, open_session

    # User explicitly set thinking=on -> recipe must NOT override it.
    run_cfg = _eval_run_cfg(thinking="on", benchmarks=["gsm8k"])
    out = tmp_path / "run"; out.mkdir()
    session = open_session(out, run_cfg, "r", [Phase.QUANT])
    hit = _hit({"thinking": "off",
                "thinking_off_model_args": {"reasoning_parser": "qwen3"}})

    new_cfg = _apply_recipe_eval_parity(run_cfg, hit, session)

    assert new_cfg.eval.thinking == "on"                      # user wins
    assert new_cfg.eval.thinking_off_model_args["reasoning_parser"] == "qwen3"


def test_bridge_noop_without_eval_parity(tmp_path: Path) -> None:
    from aqa.pipeline import _apply_recipe_eval_parity
    from aqa.session import Phase, open_session

    run_cfg = _eval_run_cfg(benchmarks=["gsm8k"])
    out = tmp_path / "run"; out.mkdir()
    session = open_session(out, run_cfg, "r", [Phase.QUANT])

    assert _apply_recipe_eval_parity(run_cfg, None, session) is run_cfg
    assert _apply_recipe_eval_parity(run_cfg, _hit(None), session) is run_cfg


def test_pin_and_lookup_roundtrip_eval_parity(tmp_path: Path) -> None:
    curated = tmp_path / "quant"
    auto = curated / "auto"
    _write(curated, ".keep", {})  # ensure curated root exists
    kb = RecipeKB(curated_root=curated, auto_root=auto)

    kb.pin(model="org/M", scheme="MXFP8", command="auto-round CMD",
           model_type="foo", gate_verdict="GO",
           eval_parity={"thinking": "off",
                        "thinking_off_model_args": {"reasoning_parser": "qwen3"}})

    hit = kb.lookup("org/M", scheme="MXFP8", model_type="foo")
    assert hit is not None
    assert hit.eval_parity == {"thinking": "off",
                               "thinking_off_model_args": {"reasoning_parser": "qwen3"}}


def test_lookup_scheme_family_suffix_matches_curated_plan(tmp_path: Path) -> None:
    curated = tmp_path / "quant"
    auto = curated / "auto"
    _write(curated, "zai-org__GLM-5.3-Flash", {
        "model": "zai-org/GLM-5.3-Flash",
        "schemes": {"MXFP4-NVFP4-Parity": {"command": "auto-round parity"}},
    })
    hit = RecipeKB(curated_root=curated, auto_root=auto).lookup(
        "zai-org/GLM-5.3-Flash", scheme="MXFP4")
    assert hit is not None
    assert hit.match_level == 4
    assert hit.scheme == "MXFP4-NVFP4-Parity"
    assert hit.command == "auto-round parity"


def test_pin_helper_records_auto_sync_branch(tmp_path: Path, monkeypatch) -> None:
    from aqa.report.manifest import record_quant_command
    from aqa.pipeline import _recipe_kb_pin

    curated = tmp_path / "quant"
    curated.mkdir()
    out = tmp_path / "run"
    out.mkdir()
    record_quant_command(out, "auto-round VALIDATED", source="builder")
    run_cfg = _quant_run_cfg(str(_model_dir(tmp_path)), curated, optimize=True,
                             pin=True, auto_sync="branch")
    calls: list[dict] = []

    def _sync(**kwargs):
        calls.append(kwargs)
        return {"status": "branch", "branch": "aqa/kb-run-1"}

    monkeypatch.setattr("aqa.agent.recipe_git.sync_auto_recipe", _sync)
    _recipe_kb_pin(run_cfg, out, _eval_result("org/M", {"mmlu": 0.7}), "GO", "run-1")
    record = json.loads((out / "recipe_kb.json").read_text())["record"]
    assert record["status"] == "ok"
    assert record["auto_sync"]["status"] == "branch"
    assert calls and calls[0]["mode"] == "branch"

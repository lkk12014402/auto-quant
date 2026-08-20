"""Tests for the agent fix-loop state machine and its parts."""

from __future__ import annotations

from pathlib import Path

from aqa.agent.drift import classify_error, extract_progress, logs_similarity
from aqa.agent.fix_loop import FixLoop, FixLoopConfig, PhaseResult
from aqa.agent.lessons import LessonStore, extract_signature
from aqa.agent.prompt import build_fix_prompt
from aqa.agent.tiers import resolve_tier, resolve_tiers
from aqa.agent.verdict import AgentAnalysis, Verdict, extract_field, parse_verdict

# ── verdict parsing ──────────────────────────────────────────────────────────


def test_parse_verdict_real_line():
    assert parse_verdict("VERDICT: FIXABLE") is Verdict.FIXABLE
    assert parse_verdict("verdict: unfixable") is Verdict.UNFIXABLE
    assert parse_verdict("- **VERDICT:** ESCALATE") is Verdict.ESCALATE


def test_parse_verdict_ignores_placeholder_and_prose():
    assert parse_verdict("VERDICT: <FIXABLE | UNFIXABLE | ESCALATE>") is None
    assert parse_verdict("Only declare VERDICT: UNFIXABLE for corrupt weights.") is None


def test_parse_verdict_last_wins():
    text = "VERDICT: ESCALATE\n...more...\nVERDICT: FIXABLE\n"
    assert parse_verdict(text) is Verdict.FIXABLE


def test_extract_field_last_and_placeholder():
    text = "ERROR_CLASS: <ONE token>\nERROR_CLASS: shape_mismatch\n"
    assert extract_field(text, "ERROR_CLASS") == "shape_mismatch"


def test_agent_analysis_parse():
    text = (
        "COMPONENT: auto_round\n"
        "ERROR_CLASS: Shape_Mismatch!!\n"
        "ROOT_CAUSE_HYPOTHESIS: layer 3 weight shape wrong\n"
        "VERDICT: FIXABLE\n"
        "FIX_PLAN: reshape the tensor\n"
    )
    a = AgentAnalysis.parse(text)
    assert a.verdict is Verdict.FIXABLE
    assert a.component == "auto_round"
    assert a.error_class == "shape_mismatch"  # normalized
    assert "reshape" in a.fix_plan
    assert a.to_dict()["error_class"] == "shape_mismatch"


# ── drift signals ────────────────────────────────────────────────────────────


def test_classify_error():
    assert classify_error("torch.cuda.OutOfMemoryError: CUDA out of memory") == "oom"
    assert classify_error("RuntimeError: size mismatch for x") == "shape_mismatch"
    assert classify_error("something totally unknown here") == "unknown"


def test_extract_progress():
    assert extract_progress("quantizing model.layers.3 ... model.layers.17 done") == 17
    assert extract_progress("no layers here") == -1


def test_logs_similarity_same_vs_different():
    a = "2024-01-01T00:00:00 RuntimeError: size mismatch for layer 3"
    b = "2024-06-01T12:00:00 RuntimeError: size mismatch for layer 9"
    assert logs_similarity(a, b) is not None
    assert logs_similarity(a, b) >= 0.90
    c = "ImportError: No module named vllm"
    assert logs_similarity(a, c) < 0.90


# ── tiers ────────────────────────────────────────────────────────────────────


def test_resolve_tiers_default():
    tiers = resolve_tiers([])
    assert [t.name for t in tiers] == ["openclaw", "minimax"]


def test_resolve_tier_kinds():
    assert resolve_tier("openclaw").backend == "openclaw"
    mm = resolve_tier("minimax")
    assert mm.backend == "copilot" and mm.byok and "COPILOT_PROVIDER_BASE_URL" in mm.env
    assert resolve_tier("opus").model == "claude-opus-4.8"
    assert resolve_tier("weird-model").model == "weird-model"


# ── prompt ───────────────────────────────────────────────────────────────────


def test_build_fix_prompt_sections():
    p = build_fix_prompt("quantize", "boom traceback", lessons="- did X",
                         attempt=2, prior_class="oom", user_context="use NGC image")
    assert "quantize" in p
    assert "VERDICT:" in p
    assert "boom traceback" in p
    assert "use NGC image" in p
    assert "oom" in p
    assert "did X" in p


# ── lessons ──────────────────────────────────────────────────────────────────


def test_extract_signature_prefers_exception():
    text = "some log\n15:51:56 [ERROR] wrapper\nRuntimeError: size mismatch for x\n"
    assert extract_signature(text) == "RuntimeError: size mismatch for x"


def test_lesson_store_roundtrip_and_dedup(tmp_path: Path):
    store = LessonStore(tmp_path)
    err = "RuntimeError: size mismatch for layer 3"
    store.save("quantize", err, "fixed", "reshaped it", model_id="m", scheme="W4A16")
    store.save("quantize", err, "fixed", "reshaped it better")  # same signature -> replace
    lessons = store.load("quantize")
    assert len(lessons) == 1
    assert lessons[0]["solution"] == "reshaped it better"
    assert lessons[0]["error_class"] == "shape_mismatch"
    assert "shape_mismatch" in store.format_for_prompt("quantize")


# ── fix-loop state machine ───────────────────────────────────────────────────


def _agent_runner(outputs):
    """Build an AgentRunner that returns queued outputs and records tiers used."""
    calls: list[str] = []

    def runner(tier, prompt, log_path, session_id):
        calls.append(tier.name)
        out = outputs.pop(0) if outputs else "VERDICT: FIXABLE"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(out)
        return out

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_fix_loop_success_first_try(tmp_path: Path):
    loop = FixLoop("quantize", lambda: PhaseResult(ok=True), out_dir=tmp_path)
    fr = loop.run()
    assert fr.ok and fr.attempts == 0


def test_fix_loop_fixes_after_one_attempt(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    runner = _agent_runner(["COMPONENT: our_code\nVERDICT: FIXABLE\nFIX_PLAN: patch"])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 1
    assert runner.calls == ["openclaw"]  # never escalated


def test_fix_loop_unfixable_escalates_then_aborts(tmp_path: Path):
    # Always fails; agent always says UNFIXABLE -> escalate once, then abort.
    runner = _agent_runner([
        "VERDICT: UNFIXABLE\nUNFIXABLE_REASON: multimodal-unsupported",
        "VERDICT: UNFIXABLE\nUNFIXABLE_REASON: multimodal-unsupported",
    ])
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="ValueError: no"),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert not fr.ok
    assert "unfixable" in fr.reason
    assert runner.calls == ["openclaw", "opus"]


def test_fix_loop_drift_escalates(tmp_path: Path):
    # Same error_class repeated -> drift threshold 2 -> escalate to opus.
    same = "ERROR_CLASS: shape_mismatch\nVERDICT: FIXABLE"
    runner = _agent_runner([same, same, same, same])
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="RuntimeError: size mismatch"),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=2,
                             tier_attempt_budget=5, max_attempts=6),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert not fr.ok
    assert "opus" in runner.calls  # escalated on drift


def test_fix_loop_budget_failsafe_escalation(tmp_path: Path):
    # Agent never emits a stable class -> drift can't fire; budget forces escalation.
    runner = _agent_runner(["VERDICT: FIXABLE"] * 10)
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="opaque failure"),
        config=FixLoopConfig(tiers=["openclaw", "opus"], tier_attempt_budget=2,
                             max_attempts=4),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert not fr.ok
    assert "opus" in runner.calls


def test_fix_loop_progress_resets_drift(tmp_path: Path):
    # Deeper layer each attempt -> not drift even though class repeats; eventually fixed.
    logs = [
        PhaseResult(ok=False, log_text="err layers.1 shape mismatch"),
        PhaseResult(ok=False, log_text="err layers.5 shape mismatch"),
        PhaseResult(ok=False, log_text="err layers.9 shape mismatch"),
        PhaseResult(ok=True),
    ]
    same = "ERROR_CLASS: shape_mismatch\nVERDICT: FIXABLE"
    runner = _agent_runner([same, same, same])
    loop = FixLoop(
        "quantize", lambda: logs.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=2),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok
    assert runner.calls == ["openclaw", "openclaw", "openclaw"]  # never escalated

"""Tests for the agent fix-loop state machine and its parts."""

from __future__ import annotations

from pathlib import Path

from aqa.agent.drift import (
    classify_error,
    exception_type,
    extract_progress,
    logs_similarity,
)
from aqa.agent.fix_loop import (
    AgentOutcome,
    FixLoop,
    FixLoopConfig,
    PhaseResult,
    _coerce_outcome,
)
from aqa.agent.lessons import LessonStore, extract_signature
from aqa.agent.prompt import build_fix_prompt, load_fix_playbook
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


def test_classify_error_granular():
    # Specific patterns win over the generic RuntimeError bucket.
    assert classify_error("RuntimeError: CUDA error: no kernel image is available") \
        == "cuda_arch_mismatch"
    assert classify_error("OSError: [Errno 28] No space left on device") == "disk_space"
    assert classify_error("requests.ConnectionError: Max retries exceeded") == "network_transient"
    assert classify_error("huggingface_hub GatedRepoError: 403 Client Error") == "auth_error"
    assert classify_error("Entry Not Found for url") == "download_missing"
    assert classify_error("ModuleNotFoundError: No module named 'auto_round'") == "import_error"


def test_exception_type_extracts_deepest_class():
    assert exception_type("RuntimeError: size mismatch") == "RuntimeError"
    assert exception_type("modeling_qwen.QwenError: bad rope") == "QwenError"
    # deepest (last) exception in a traceback wins
    text = "ValueError: outer\n  ...\nKeyError: 'inner'\n"
    assert exception_type(text) == "KeyError"
    assert exception_type("no exception here") == ""


def test_exception_type_unwraps_dynamo_wrapper():
    # torch-dynamo wraps the true error; the outer colon-form is a generic
    # RuntimeError. We should recover the inner AssertionError, not RuntimeError.
    text = (
        "RuntimeError: Worker failed with error 'Observed exception\n"
        "  Developer debug context: raised exception AssertionError("
        "[ConstantVariable(str: 'mm_mxfp8 requires N >= 128, got N=32.')])\n"
        "RuntimeError: Engine core initialization failed. See root cause above.\n"
    )
    assert exception_type(text) == "AssertionError"


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


def test_build_fix_prompt_prior_state_and_diff():
    p = build_fix_prompt(
        "evaluate", "boom traceback", attempt=2,
        prior_state="- agent self-report: FIXABLE\n- ACTUAL re-run ...: STILL FAILING",
        prior_diff="--- a/x\n+++ b/x\n@@\n-old\n+new",
    )
    assert "Previous fix attempt" in p
    assert "STILL FAILING" in p
    assert "```diff" in p
    assert "+new" in p


def test_load_fix_playbook_ships():
    text = load_fix_playbook()
    assert "ignore" in text.lower() and "MXFP8" in text


def test_build_fix_prompt_embeds_skill_playbook():
    p = build_fix_prompt("quantize", "boom", attempt=1,
                         skill_playbook="RULE: keep in_proj_a FP16")
    assert "Domain playbook" in p
    assert "keep in_proj_a FP16" in p


def test_build_fix_prompt_omits_skill_playbook_when_absent():
    p = build_fix_prompt("quantize", "boom", attempt=1)
    assert "Domain playbook" not in p


def test_build_fix_prompt_omits_prior_state_when_absent():
    p = build_fix_prompt("evaluate", "boom", attempt=1)
    assert "Previous fix attempt" not in p


# ── lessons ──────────────────────────────────────────────────────────────────


def test_extract_signature_prefers_exception():
    text = "some log\n15:51:56 [ERROR] wrapper\nRuntimeError: size mismatch for x\n"
    assert extract_signature(text) == "RuntimeError: size mismatch for x"


def test_extract_signature_unwraps_dynamo_over_generic_wrapper():
    # The generic "lm_eval failed for … (rc=1)" wrapper + a dynamo-wrapped
    # AssertionError: the signature must reflect the real kernel assertion, not
    # the task list or the outer RuntimeError, so knowledge keys on the true cause.
    text = (
        "lm_eval failed for piqa+hellaswag+mmlu (rc=1); see /workspace/x.log\n"
        "(EngineCore) ERROR   Developer debug context: raised exception "
        "AssertionError([ConstantVariable(str: 'mm_mxfp8 requires N >= 128, "
        "got N=32. out_features is too small for mm_mxfp8.')])\n"
        "RuntimeError: Engine core initialization failed. See root cause above.\n"
    )
    sig = extract_signature(text)
    assert sig.startswith("AssertionError: mm_mxfp8 requires N >= 128")
    assert "piqa" not in sig


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


def test_fix_loop_accrues_agent_tokens(tmp_path: Path):
    # Agent output carries a token-usage JSON blob -> FixResult.tokens accrues it.
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    usage = ('VERDICT: FIXABLE\nFIX_PLAN: patch\n'
             '{"token_usage": {"input_tokens": 120, "output_tokens": 40, '
             '"cached_input_tokens": 10}}')
    runner = _agent_runner([usage])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 1
    assert fr.tokens is not None and not fr.tokens.is_empty()
    assert fr.tokens.n_input_tokens == 120
    assert fr.tokens.n_output_tokens == 40
    assert fr.tokens.n_cache_tokens == 10


def test_fix_loop_no_tokens_when_agent_silent(tmp_path: Path):
    # First-try success -> no agent invoked -> tokens empty/None.
    loop = FixLoop("quantize", lambda: PhaseResult(ok=True), out_dir=tmp_path)
    fr = loop.run()
    assert fr.ok and fr.attempts == 0
    assert fr.tokens is None or fr.tokens.is_empty()


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


# ── prior-state block + timeout signal ───────────────────────────────────────


def _recording_runner(outputs):
    """AgentRunner that records the prompts it received; supports str or AgentOutcome."""
    calls: list[str] = []
    prompts: list[str] = []

    def runner(tier, prompt, log_path, session_id):
        calls.append(tier.name)
        prompts.append(prompt)
        out = outputs.pop(0) if outputs else "VERDICT: FIXABLE"
        text = out.text if isinstance(out, AgentOutcome) else out
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(text)
        return out

    runner.calls = calls          # type: ignore[attr-defined]
    runner.prompts = prompts      # type: ignore[attr-defined]
    return runner


def test_coerce_outcome_accepts_str_and_outcome():
    assert _coerce_outcome("hello").text == "hello"
    assert not _coerce_outcome("hello").timed_out
    oc = AgentOutcome(text="x", rc=124, timed_out=True)
    assert _coerce_outcome(oc) is oc


def test_prior_state_injected_into_next_attempt(tmp_path: Path):
    # Attempt 1 claims FIXED but the re-run still fails; attempt 2 sees the honest
    # 'STILL FAILING' prior-state and then succeeds.
    results = [
        PhaseResult(ok=False, log_text="RuntimeError: size mismatch for x"),
        PhaseResult(ok=False, log_text="RuntimeError: size mismatch for x"),  # attempt-1 re-run
        PhaseResult(ok=True),                                                 # attempt-2 re-run
    ]
    runner = _recording_runner([
        "ERROR_CLASS: shape_mismatch\nVERDICT: FIXABLE\nFIX_PLAN: reshape it",
        "ERROR_CLASS: shape_mismatch\nVERDICT: FIXABLE\nFIX_PLAN: really reshape it",
    ])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=5,
                             tier_attempt_budget=5),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 2
    p2 = runner.prompts[1]
    assert "Previous fix attempt" in p2
    assert "agent self-report: FIXABLE" in p2
    assert "STILL FAILING" in p2
    assert "error_class=shape_mismatch" in p2
    # And it points the next attempt at the prior attempt's on-disk artifacts.
    assert "attempt_1.log" in p2
    assert "prompt_1.txt" in p2
    # First attempt has no prior-state.
    assert "Previous fix attempt" not in runner.prompts[0]


def test_timeout_surfaced_as_truncated_in_prior_state(tmp_path: Path):
    # Attempt 1 is wall-clock-killed (timed_out) -> next attempt's prompt warns the
    # edits may be half-applied, rather than treating it as a clean finish.
    results = [
        PhaseResult(ok=False, log_text="RuntimeError: boom"),
        PhaseResult(ok=False, log_text="RuntimeError: boom"),
        PhaseResult(ok=True),
    ]
    runner = _recording_runner([
        AgentOutcome(text="ERROR_CLASS: boom_err\n(no verdict -- killed)",
                     rc=124, timed_out=True),
        "VERDICT: FIXABLE\nFIX_PLAN: finish it",
    ])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=5,
                             tier_attempt_budget=5),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 2
    p2 = runner.prompts[1]
    assert "timed out / truncated: YES" in p2
    assert "half" in p2.lower()  # "edits may be HALF-APPLIED"


# ── inter-attempt information flow (history / evidence / evolution) ─────────


def test_agent_analysis_parses_evidence_fields():
    text = (
        "COMPONENT: vllm\n"
        "ERROR_CLASS: import_error\n"
        "EVIDENCE_CMD: python -c 'import vllm'\n"
        "EVIDENCE_RESULT: ModuleNotFoundError: No module named 'vllm'\n"
        "VERDICT: FIXABLE\n"
    )
    a = AgentAnalysis.parse(text)
    assert a.evidence_cmd == "python -c 'import vllm'"
    assert "ModuleNotFoundError" in a.evidence_result
    assert a.to_dict()["evidence_cmd"] == "python -c 'import vllm'"
    # absent evidence -> empty fields, absent from to_dict
    b = AgentAnalysis.parse("VERDICT: FIXABLE\n")
    assert b.evidence_cmd == "" and "evidence_cmd" not in b.to_dict()


def test_build_fix_prompt_history_block_placement():
    p = build_fix_prompt("quantize", "boom", attempt=2,
                         history="TABLE", prior_state="PRIOR")
    assert "Attempt history THIS run" in p and "TABLE" in p
    assert p.index("Attempt history THIS run") < p.index("Previous fix attempt")
    # omitted by default (attempt-1 prompts unchanged)
    assert "## Attempt history" not in build_fix_prompt("quantize", "boom")
    # the protocol tells the agent to consult the history block
    assert "consult it FIRST" in build_fix_prompt("quantize", "boom")


def test_prior_state_carries_probe_evidence(tmp_path: Path):
    results = [
        PhaseResult(ok=False, log_text="ModuleNotFoundError: No module named 'vllm'"),
        PhaseResult(ok=False, log_text="ModuleNotFoundError: No module named 'vllm'"),
        PhaseResult(ok=True),
    ]
    runner = _recording_runner([
        ("ERROR_CLASS: import_error\nROOT_CAUSE_HYPOTHESIS: vllm missing\n"
         "EVIDENCE_CMD: python -c 'import vllm'\n"
         "EVIDENCE_RESULT: ModuleNotFoundError: No module named 'vllm'\n"
         "VERDICT: FIXABLE\nFIX_PLAN: install vllm"),
        "VERDICT: FIXABLE\nFIX_PLAN: done",
    ])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=5,
                             tier_attempt_budget=5),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok
    p2 = runner.prompts[1]
    assert "what the probe showed" in p2
    assert "$ python -c 'import vllm'" in p2
    assert "ModuleNotFoundError: No module named 'vllm'" in p2
    # attempt 1 had no probe to show
    assert "what the probe showed" not in runner.prompts[0]


def test_prior_state_evidence_truncated(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom")] * 2 + [
        PhaseResult(ok=True)]
    runner = _recording_runner([
        "EVIDENCE_CMD: cat log\nEVIDENCE_RESULT: " + "x" * 800 + "\nVERDICT: FIXABLE",
        "VERDICT: FIXABLE\nFIX_PLAN: done",
    ])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=5,
                             tier_attempt_budget=5),
        out_dir=tmp_path, run_agent=runner,
    )
    assert loop.run().ok
    p2 = runner.prompts[1]
    probe = p2.split("what the probe showed", 1)[1]
    assert "x" * 400 in probe  # kept a substantial prefix
    assert "x" * 600 not in probe  # capped (~500 chars total)


def test_history_absent_on_first_attempt(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    runner = _recording_runner(["VERDICT: FIXABLE\nFIX_PLAN: patch"])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=runner,
    )
    assert loop.run().ok
    assert "## Attempt history THIS run" not in runner.prompts[0]


def test_history_table_dead_ends_and_evolution(tmp_path: Path):
    results = [
        PhaseResult(ok=False, log_text="ModuleNotFoundError: No module named 'vllm'"),
        PhaseResult(ok=False, log_text="torch.cuda.OutOfMemoryError: CUDA out of memory"),
        PhaseResult(ok=False, log_text="RuntimeError: size mismatch for x"),
        PhaseResult(ok=True),
    ]
    runner = _recording_runner([
        ("ERROR_CLASS: import_error\nROOT_CAUSE_HYPOTHESIS: vllm missing\n"
         "VERDICT: FIXABLE\nFIX_PLAN: install vllm"),
        ("ERROR_CLASS: oom\nROOT_CAUSE_HYPOTHESIS: kv cache too big\n"
         "VERDICT: FIXABLE\nFIX_PLAN: lower mem util"),
        "VERDICT: FIXABLE\nFIX_PLAN: final fix",
    ])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=5,
                             tier_attempt_budget=5),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 3
    p3 = runner.prompts[2]
    # the table lists both prior attempts with tier / class-before / verdict / summary
    assert "| 1 | openclaw | import_error | -1 | FIXABLE | vllm missing |" in p3
    assert "| 2 | openclaw | oom | -1 | FIXABLE | kv cache too big |" in p3
    # error evolution: class sequence across attempts + the current failure
    assert "error evolution: import_error → oom → shape_mismatch" in p3
    # dead ends from the rejected trail are shown (not just persisted on success)
    assert "hypotheses already ruled out this run" in p3
    assert "- import_error: vllm missing" in p3
    assert "- oom: kv cache too big" in p3
    # no progress segment for a non-quantize log family (no layers.N markers)
    assert "progress depth" not in p3


def test_history_progress_advancement_line(tmp_path: Path):
    logs = [
        PhaseResult(ok=False, log_text="quant err layers.1\nRuntimeError: size mismatch"),
        PhaseResult(ok=False, log_text="quant err layers.3\nRuntimeError: size mismatch"),
        PhaseResult(ok=False, log_text="quant err layers.5\nRuntimeError: size mismatch"),
        PhaseResult(ok=True),
    ]
    same = "ERROR_CLASS: shape_mismatch\nVERDICT: FIXABLE\nFIX_PLAN: tweak"
    runner = _recording_runner([same, same, same])
    loop = FixLoop(
        "quantize", lambda: logs.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"], drift_threshold=9,
                             tier_attempt_budget=9),
        out_dir=tmp_path, run_agent=runner,
    )
    assert loop.run().ok
    p3 = runner.prompts[2]
    assert "quantize progress depth 1 → 3 (advanced)" in p3


def test_history_capped_and_summaries_truncated(tmp_path: Path):
    long_cause = "hypothesis " + "r" * 200
    outputs = [f"ROOT_CAUSE_HYPOTHESIS: {long_cause}\nVERDICT: FIXABLE"] * 8
    runner = _recording_runner(outputs)
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="opaque failure"),
        config=FixLoopConfig(tiers=["openclaw"], tier_attempt_budget=8,
                             drift_threshold=99, max_attempts=8),
        out_dir=tmp_path, run_agent=runner,
    )
    fr = loop.run()
    assert not fr.ok and fr.attempts == 8
    p8 = runner.prompts[7]
    # only the last ~6 attempts are tabulated
    assert "| 1 |" not in p8 and "| 7 |" in p8
    # history-table summaries are capped at 120 chars (the prior_state block has
    # its own, larger cap -- restrict the check to the table region)
    table = p8.split("## Attempt history THIS run", 1)[1].split("error evolution", 1)[0]
    assert "r" * 121 not in table
    assert "hypothesis " + "r" * 100 in table


# ── upgraded deterministic classification prior ──────────────────────────────

from aqa.agent.drift import classify_error_detail
from aqa.agent.prompt import playbook_section_for


def test_classify_error_real_run_regressions():
    """Tonight's seven real GPU-run failures, pinned to their expected classes."""
    cases = [
        ("auto-round is not installed; `pip install auto-round`", "missing_tool"),
        ("Error: Access denied. This repository requires approval.", "auth_error"),
        (("RuntimeError: model parallel state must be empty before initialization; "
          "found dp, cfg, sp"), "engine_reinit"),
        ("KeyError: 'layers.29.adaLN_modulation.0.qweight'", "quant_artifact"),
        (("ValueError: There is no module or parameter named "
          "'text_encoder.embed_tokens' in ZImagePipeline..."), "quant_artifact"),
        (("ValueError: Sequence length must be less than max_position_embeddings "
          "(got 127 and 77)"), "context_length"),
        ("PermissionError: [Errno 13] Permission denied: '/root/x'", "permission_error"),
    ]
    for text, expected in cases:
        assert classify_error(text) == expected, text


def test_classify_error_rule_ordering_guards():
    # missing_tool fires before import_error even inside an ImportError message
    assert classify_error("ImportError: auto-round is not installed") == "missing_tool"
    # quant_artifact fires before the coarse exception buckets
    assert classify_error("KeyError: 'model.layers.0.qweight'") == "quant_artifact"
    # engine_reinit fires before vllm_engine_error / runtime_error buckets
    assert classify_error("RuntimeError: model parallel state must be empty") \
        == "engine_reinit"
    # context_length fires before the value_error bucket
    assert classify_error("ValueError: Sequence length must be less than "
                          "max_position_embeddings") == "context_length"
    # checkpoint_key_mismatch keeps its own patterns even when a packed-weight
    # name appears in the same line
    assert classify_error("RuntimeError: Missing key(s) in state_dict: 'layers.0.qweight'") \
        == "checkpoint_key_mismatch"


def test_classify_error_wrapper_compatibility():
    # the wrapper stays a thin alias returning exactly the detail's class name
    for text in ("torch.cuda.OutOfMemoryError: CUDA out of memory",
                 "ModuleNotFoundError: No module named 'auto_round'",
                 "something totally unknown here"):
        assert classify_error(text) == classify_error_detail(text).class_name


def test_classify_error_detail_fields():
    d = classify_error_detail("line one\ntorch.cuda.OutOfMemoryError: CUDA out of memory")
    assert d.class_name == "oom"
    assert d.rule_id.startswith("oom#")
    assert d.evidence == "torch.cuda.OutOfMemoryError: CUDA out of memory"
    assert len(d.evidence) <= 200
    long_line = "CUDA out of memory " + "y" * 500
    assert len(classify_error_detail(long_line).evidence) <= 200
    unknown = classify_error_detail("totally opaque")
    assert unknown.class_name == "unknown"
    assert unknown.rule_id == "" and unknown.evidence == ""


def test_classification_prompt_block_with_evidence():
    p = build_fix_prompt("quantize", "boom", prior_class="oom",
                         prior_evidence="torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert "error_class guess: oom (matched: torch.cuda.OutOfMemoryError" in p
    # no evidence -> the block keeps its exact previous format
    p2 = build_fix_prompt("quantize", "boom", prior_class="oom")
    assert "error_class guess: oom\n" in p2


def test_playbook_section_pointer():
    assert playbook_section_for("oom") == "OOM / memory"
    assert playbook_section_for("quant_artifact") == "Diffusion (vllm-omni text-to-image)"
    assert playbook_section_for("engine_reinit") == "Diffusion (vllm-omni text-to-image)"
    assert playbook_section_for("unmapped_class") == ""
    p = build_fix_prompt("quantize", "boom", skill_playbook="BODY",
                         playbook_pointer="OOM / memory")
    assert "The section most relevant to this class: OOM / memory" in p
    assert p.index("most relevant to this class") < p.index("BODY")  # pointer before content
    p2 = build_fix_prompt("quantize", "boom", skill_playbook="BODY")
    assert "most relevant to this class" not in p2


def test_fix_loop_prompt_shows_evidence_and_pointer(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="torch.cuda.OutOfMemoryError: CUDA OOM"),
               PhaseResult(ok=True)]
    runner = _recording_runner(["VERDICT: FIXABLE\nFIX_PLAN: lower mem"])
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=runner,
    )
    assert loop.run().ok
    p1 = runner.prompts[0]
    assert "error_class guess: oom (matched: torch.cuda.OutOfMemoryError" in p1
    assert "The section most relevant to this class: OOM / memory" in p1

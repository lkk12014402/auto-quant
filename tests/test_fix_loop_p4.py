"""P4 wiring: Critic calibration events, PolicyGate review/enforce, state re-seed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aqa.agent.critic import Critic
from aqa.agent.fix_loop import FixLoop, FixLoopConfig, PhaseResult
from aqa.agent.policy_gate import PolicyGate
from aqa.session.state import Phase, SessionState


def _runner(outputs):
    def runner(tier, prompt, log_path, session_id):
        out = outputs.pop(0) if outputs else "VERDICT: FIXABLE"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(out)
        return out
    return runner


@dataclass
class _FakePatchCapture:
    """Minimal PatchCapture double: a fixed diff + recorded restore() calls."""

    diff: str = ""
    restored: int = 0
    snapshots: int = 0
    written: int = 0

    def write_patches(self, out_dir) -> dict:
        self.written += 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "auto_round.patch"
        path.write_text(self.diff)
        return {"auto_round": path}

    def snapshot(self) -> int:
        self.snapshots += 1
        return 1

    def combined_diff(self) -> str:
        return self.diff

    def restore(self) -> int:
        self.restored += 1
        return 1


def _events_sink():
    events: list[tuple[str, dict]] = []
    return events, (lambda t, f: events.append((t, f)))


_TEST_DIFF = "--- a/auto_round/tests/test_q.py\n+++ b/auto_round/tests/test_q.py\n+assert True\n"


def test_critic_emits_calibration_and_state_seed_threaded(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    events, sink = _events_sink()
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw", "opus"]),
        out_dir=tmp_path, run_agent=_runner(["VERDICT: FIXABLE\nFIX_PLAN: patch"]),
        critic=Critic(), on_event=sink,
        state_seed="- run: r1  |  current phase: QUANT",
    )
    fr = loop.run()
    assert fr.ok and fr.attempts == 1
    # Critic recorded the (confirmed) calibration as an event.
    cal = [f for t, f in events if t == "critic_calibration"]
    assert cal and cal[0]["calibration"] == "confirmed" and cal[0]["phase"] == "quantize"
    # The state seed was threaded into the fix prompt.
    prompt = (loop.fix_log_dir / "prompt_1.txt").read_text()
    assert "current phase: QUANT" in prompt
    assert "re-seeded from session state" in prompt


def test_critic_flags_overclaim(tmp_path: Path):
    # Agent claims FIXABLE but the re-run keeps failing -> overclaimed (dishonest).
    events, sink = _events_sink()
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="ValueError: still"),
        config=FixLoopConfig(tiers=["openclaw"], tier_attempt_budget=1, max_attempts=1),
        out_dir=tmp_path, run_agent=_runner(["VERDICT: FIXABLE"]),
        critic=Critic(), on_event=sink,
    )
    loop.run()
    cals = [f for t, f in events if t == "critic_calibration"]
    assert cals and cals[-1]["calibration"] == "overclaimed"
    assert cals[-1]["honest"] is False


def test_policy_gate_advisory_records_but_does_not_revert(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    events, sink = _events_sink()
    pc = _FakePatchCapture(diff=_TEST_DIFF)
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path, run_agent=_runner(["VERDICT: FIXABLE"]),
        patch_capture=pc, policy_gate=PolicyGate(enforce=False), on_event=sink,
    )
    loop.run()
    viol = [f for t, f in events if t == "policy_violation"]
    assert viol and "edit_tests" in viol[0]["violations"][0]["rule"]
    assert viol[0]["enforced"] is False
    assert pc.restored == 0  # advisory: never reverted
    assert not any(t == "policy_revert" for t, _ in events)


def test_policy_gate_enforce_reverts_violation(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="RuntimeError: boom"), PhaseResult(ok=True)]
    events, sink = _events_sink()
    pc = _FakePatchCapture(diff=_TEST_DIFF)
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path, run_agent=_runner(["VERDICT: FIXABLE"]),
        patch_capture=pc, policy_gate=PolicyGate(enforce=True), on_event=sink,
    )
    loop.run()
    assert pc.restored >= 1  # enforce: the violating edit was rolled back
    assert any(t == "policy_revert" for t, _ in events)


def test_clean_diff_passes_policy(tmp_path: Path):
    results = [PhaseResult(ok=False, log_text="boom"), PhaseResult(ok=True)]
    events, sink = _events_sink()
    pc = _FakePatchCapture(diff="--- a/auto_round/q.py\n+++ b/auto_round/q.py\n+ok = 1\n")
    loop = FixLoop(
        "quantize", lambda: results.pop(0),
        config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path, run_agent=_runner(["VERDICT: FIXABLE"]),
        patch_capture=pc, policy_gate=PolicyGate(enforce=True), on_event=sink,
    )
    loop.run()
    assert not any(t in ("policy_violation", "policy_revert") for t, _ in events)
    assert pc.restored == 0


def test_failure_path_writes_patches_without_knowledge(tmp_path: Path):
    pc = _FakePatchCapture(diff=_TEST_DIFF)
    loop = FixLoop(
        "quantize", lambda: PhaseResult(ok=False, log_text="RuntimeError: boom"),
        config=FixLoopConfig(tiers=["openclaw"]), out_dir=tmp_path,
        run_agent=_runner(["VERDICT: UNFIXABLE\nUNFIXABLE_REASON: nope"]),
        patch_capture=pc,
    )
    result = loop.run()
    assert not result.ok
    assert pc.written >= 1
    assert (tmp_path / "patches" / "quantize" / "auto_round.patch").is_file()


# ── SessionState.prompt_seed ────────────────────────────────────────────────

def test_prompt_seed_renders_mission_and_decisions():
    state = SessionState.new("run-xyz", [Phase.QUANT, Phase.EVAL, Phase.CLOSE])
    state.enter(Phase.QUANT)
    state.baseline = {"mean_score": 0.80}
    state.current_best = {"scheme": "MXFP4", "mean_score": 0.78}
    state.record_decision(Phase.ANALYZE, "hot-start from recipe KB: Qwen/X",
                          "exact model + scheme")
    seed = state.prompt_seed()
    assert "run-xyz" in seed and "QUANT" in seed
    assert "mean_score=0.8" in seed and "do NOT regress" in seed
    assert "MXFP4" in seed
    assert "hot-start from recipe KB" in seed


def test_prompt_seed_minimal_when_empty():
    state = SessionState.new("r", [Phase.EVAL])
    seed = state.prompt_seed()
    assert "r" in seed and "EVAL" in seed
    assert "decisions already made" not in seed  # none recorded

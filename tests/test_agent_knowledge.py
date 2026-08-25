"""Tests for the shared knowledge store + deterministic patch replay."""

from __future__ import annotations

import json
from pathlib import Path

from aqa.agent.fix_loop import FixLoop, FixLoopConfig, PhaseResult
from aqa.agent.knowledge import KnowledgeStore, signature_hash
from aqa.agent.patch import PatchArea, PatchCapture


# ── fixtures ──────────────────────────────────────────────────────────────────
def _area(tmp_path: Path) -> tuple[PatchArea, Path]:
    src = tmp_path / "pkg"
    src.mkdir(parents=True)
    (src / "mod.py").write_text("def f():\n    return 1\n")
    return PatchArea("auto_round", "auto_round", src), src


def _make_patch(tmp_path: Path) -> dict[str, Path]:
    """Produce a real per-area .patch (return-1 -> return-2) via PatchCapture."""
    area, src = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    cap.snapshot()
    (src / "mod.py").write_text("def f():\n    return 2\n")  # simulated fix
    written = cap.write_patches(tmp_path / "patches")
    # reset the source so a later replay re-applies the change
    (src / "mod.py").write_text("def f():\n    return 1\n")
    return {"area": area, "src": src, "snap": tmp_path / "snap", "written": written}


# ── PatchCapture.apply_recorded / restore ─────────────────────────────────────
def test_apply_recorded_applies_and_restore_reverts(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    cap = PatchCapture(ctx["snap"], areas=[area])  # snapshot already on disk
    cap._snapped.add("auto_round")  # snapshot dir exists from _make_patch

    ok = cap.apply_recorded({"auto_round": ctx["written"]["auto_round"]})
    assert ok is True
    assert "return 2" in (src / "mod.py").read_text()

    assert cap.restore() == 1
    assert "return 1" in (src / "mod.py").read_text()


def test_apply_recorded_miss_on_incompatible_source(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    # Diverge the live source so the recorded hunk no longer applies cleanly.
    (src / "mod.py").write_text("totally different content\n")
    cap = PatchCapture(ctx["snap"], areas=[area])
    assert cap.apply_recorded({"auto_round": ctx["written"]["auto_round"]}) is False
    assert (src / "mod.py").read_text() == "totally different content\n"  # untouched


# ── KnowledgeStore record + find ──────────────────────────────────────────────
def test_record_and_find_by_signature(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    rec = kn.record("quantize", err, ctx["written"], model_id="m1", scheme="MXFP4")
    assert rec is not None
    assert (kn.local_dir / "patches" / "quantize" / rec.sig_hash / "auto_round.patch").is_file()
    assert (kn.local_dir / "patches" / "quantize" / rec.sig_hash / "meta.json").is_file()

    # Same signature on a *different* model still matches (cross-model share).
    found = kn.find_patches("quantize", "RuntimeError: size mismatch for layer 3")
    assert len(found) == 1
    assert found[0].sig_hash == signature_hash(rec.signature)
    assert "auto_round" in found[0].patch_files


def test_find_ranks_signature_over_class(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    # Two shape_mismatch records: only one shares the exact signature.
    kn.record("quantize", "RuntimeError: size mismatch for layer 9", ctx["written"])
    kn.record("quantize", "RuntimeError: size mismatch for layer 3", ctx["written"])
    found = kn.find_patches("quantize", "RuntimeError: size mismatch for layer 3")
    assert len(found) == 2
    assert found[0].signature == "RuntimeError: size mismatch for layer 3"  # exact first


def test_find_ranks_exc_type_over_coarse_class(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    # Custom model exception: coarse category is 'unknown', but exc_type is precise.
    kn.record("quantize", "modeling_x.WeirdError: boom in layer 5", ctx["written"])
    rec = kn.find_patches("quantize", "modeling_x.WeirdError: totally different message")
    assert len(rec) == 1
    assert rec[0].exc_type == "WeirdError"  # matched on exact exception class, not text


def test_merge_lessons_dedups_by_signature(tmp_path: Path):
    local = tmp_path / "lessons.jsonl"
    remote = tmp_path / "remote.jsonl"
    local.write_text(json.dumps({"signature": "sig-a", "solution": "old"}) + "\n")
    remote.write_text(
        json.dumps({"signature": "sig-a", "solution": "new"}) + "\n"
        + json.dumps({"signature": "sig-b", "solution": "other"}) + "\n"
    )
    KnowledgeStore._merge_lessons(local, remote)
    rows = [json.loads(x) for x in local.read_text().splitlines() if x.strip()]
    by_sig = {r["signature"]: r["solution"] for r in rows}
    assert by_sig == {"sig-a": "new", "sig-b": "other"}  # remote wins, no dup


# ── FixLoop replay-as-reference (Option A: no auto-apply) ─────────────────────
def test_fixloop_injects_recorded_patch_as_reference(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    kn.record("quantize", err, ctx["written"])

    cap = PatchCapture(tmp_path / "snap2", areas=[area])

    results = [PhaseResult(ok=False, log_text=err), PhaseResult(ok=True)]
    prompts: list[str] = []

    # The agent consults the injected reference and applies the fix ITSELF; the
    # recorded patch is never auto-applied by the loop.
    def agent(_t, prompt, log_path, _s):
        prompts.append(prompt)
        (src / "mod.py").write_text("def f():\n    return 2\n")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("VERDICT: FIXABLE")
        return "VERDICT: FIXABLE"

    loop = FixLoop(
        "quantize", lambda: results.pop(0), config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, patch_capture=cap,
    )
    result = loop.run()
    assert result.ok is True
    assert result.attempts == 1          # an agent tier WAS spent (no free deterministic win)
    assert not result.reason.startswith("replayed:")
    # The recorded diff was handed to the agent as reference (not auto-applied).
    assert "REFERENCE ONLY" in prompts[0]
    assert "return 2" in prompts[0]      # the recorded diff body
    assert "sig=" in prompts[0]          # provenance line


def test_fixloop_no_reference_when_replay_disabled(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    kn.record("quantize", err, ctx["written"])
    cap = PatchCapture(tmp_path / "snap2", areas=[area])

    results = [PhaseResult(ok=False, log_text=err), PhaseResult(ok=True)]
    prompts: list[str] = []

    def agent(_t, prompt, log_path, _s):
        prompts.append(prompt)
        (src / "mod.py").write_text("def f():\n    return 2\n")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("VERDICT: FIXABLE")
        return "VERDICT: FIXABLE"

    loop = FixLoop(
        "quantize", lambda: results.pop(0), config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, patch_capture=cap,
        replay=False,
    )
    result = loop.run()
    assert result.ok is True
    assert "REFERENCE ONLY" not in prompts[0]


def test_fixloop_stat_reference_used_when_recorded_match(tmp_path: Path):
    # A recorded fix matches the signature and is injected as reference; the agent
    # resolves it -> credited as `reference_used`, and hit_rate reflects it.
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    kn.record("quantize", err, ctx["written"])
    cap = PatchCapture(tmp_path / "snap2", areas=[area])

    results = [PhaseResult(ok=False, log_text=err), PhaseResult(ok=True)]

    def agent(_t, _prompt, log_path, _s):
        (src / "mod.py").write_text("def f():\n    return 2\n")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("VERDICT: FIXABLE")
        return "VERDICT: FIXABLE"

    loop = FixLoop(
        "quantize", lambda: results.pop(0), config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, patch_capture=cap,
    )
    assert loop.run().ok is True
    s = kn.stats()
    assert s["reference_used"] == 1
    assert s["agent_fixed"] == 0
    assert s["replay_hit"] == 0
    assert s["hit_rate"] == 1.0


def test_fixloop_stat_agent_fixed_when_no_recorded_match(tmp_path: Path):
    # No recorded fix for this signature -> nothing injected as reference; a
    # successful agent fix is credited as plain `agent_fixed`, hit_rate stays 0.
    ctx = _make_patch(tmp_path)
    src = ctx["src"]
    kn = KnowledgeStore(local_dir=tmp_path / "kn")  # empty store
    err = "RuntimeError: size mismatch for layer 3"

    results = [PhaseResult(ok=False, log_text=err), PhaseResult(ok=True)]

    def agent(_t, _prompt, log_path, _s):
        (src / "mod.py").write_text("def f():\n    return 2\n")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("VERDICT: FIXABLE")
        return "VERDICT: FIXABLE"

    loop = FixLoop(
        "quantize", lambda: results.pop(0), config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn,
    )
    assert loop.run().ok is True
    s = kn.stats()
    assert s["reference_used"] == 0
    assert s["agent_fixed"] == 1
    assert s["hit_rate"] == 0.0


def test_fixloop_reference_summarises_large_diff(tmp_path: Path):
    # A big recorded patch is NOT inlined verbatim: the prompt gets a summary +
    # on-disk pointer instead (so a multi-file source patch can't bloat the prompt).
    area, src = _area(tmp_path)
    cap0 = PatchCapture(tmp_path / "snap", areas=[area])
    cap0.snapshot()
    big = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(400)) + "    return 2\n"
    (src / "mod.py").write_text(big)
    written = cap0.write_patches(tmp_path / "patches")
    (src / "mod.py").write_text("def f():\n    return 1\n")

    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    kn.record("quantize", err, written)
    cap = PatchCapture(tmp_path / "snap2", areas=[area])

    results = [PhaseResult(ok=False, log_text=err), PhaseResult(ok=True)]
    prompts: list[str] = []

    def agent(_t, prompt, log_path, _s):
        prompts.append(prompt)
        (src / "mod.py").write_text(big)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("VERDICT: FIXABLE")
        return "VERDICT: FIXABLE"

    loop = FixLoop(
        "quantize", lambda: results.pop(0), config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, patch_capture=cap,
    )
    result = loop.run()
    assert result.ok is True
    p0 = prompts[0]
    assert "REFERENCE ONLY" in p0
    assert "full patch on disk" in p0      # summarised, not inlined
    assert "touches" in p0 and "lines" in p0
    assert "x399 = 399" not in p0          # the big body is NOT pasted verbatim


def test_fixloop_reverts_and_escalates_when_replay_fails(tmp_path: Path):
    ctx = _make_patch(tmp_path)
    area, src = ctx["area"], ctx["src"]
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: size mismatch for layer 3"
    kn.record("quantize", err, ctx["written"])
    cap = PatchCapture(tmp_path / "snap2", areas=[area])

    # Phase never passes -> replay applies patch, phase still fails, must revert.
    def run_phase() -> PhaseResult:
        return PhaseResult(ok=False, log_text=err)

    def agent(_t, _p, log_path, _s):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("VERDICT: UNFIXABLE\nUNFIXABLE_REASON: test\n")
        return log_path.read_text()

    loop = FixLoop(
        "quantize", run_phase, config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, patch_capture=cap,
    )
    result = loop.run()
    assert result.ok is False
    # After a failed replay the source was reverted to pristine (return 1).
    assert "return 1" in (src / "mod.py").read_text()

"""Tests for distilled agent-fix trajectory summaries as replayable knowledge."""

from __future__ import annotations

import json
from pathlib import Path

from aqa.agent.fix_loop import FixLoop, FixLoopConfig, PhaseResult
from aqa.agent.knowledge import KnowledgeStore
from aqa.agent.trajectories import (
    TrajectoryStore,
    build_summary,
)
from aqa.agent.verdict import AgentAnalysis


# ── build_summary distillation ────────────────────────────────────────────────
def test_build_summary_distils_from_analysis():
    err = "RuntimeError: CUDA out of memory while packing layer 7"
    analysis = AgentAnalysis.parse(
        "VERDICT: FIXABLE\n"
        "COMPONENT: auto_round\n"
        "ERROR_CLASS: oom\n"
        "ROOT_CAUSE_HYPOTHESIS: device_map=auto sharded embeddings across GPUs\n"
        "FIX_PLAN: pin to a single GPU and stream packing to disk\n"
    )
    summary = build_summary(
        "quantize", err, analysis,
        tried_and_rejected=["oom: lowered batch size", "oom: raised max_memory"],
        tier="opus", steps=3, model_id="Qwen/Qwen3.5-9B", run_id="r1",
    )
    assert summary.error_class == "oom"
    assert "device_map" in summary.root_cause
    assert "single GPU" in summary.fix_rationale
    assert summary.tried_and_rejected == [
        "oom: lowered batch size", "oom: raised max_memory"]
    assert summary.tier == "opus" and summary.steps == 3
    d = summary.to_dict()
    assert d["signature"] and d["exc_type"] == "RuntimeError"


# ── TrajectoryStore dedup + retrieval ──────────────────────────────────────────
def test_store_dedups_by_signature(tmp_path: Path):
    store = TrajectoryStore(tmp_path / "traj")
    err = "ValueError: bad config"
    s1 = build_summary("quantize", err, None, tier="a", steps=1)
    s1.root_cause = "first"
    s2 = build_summary("quantize", err, None, tier="b", steps=2)
    s2.root_cause = "second"
    store.save("quantize", s1)
    store.save("quantize", s2)
    rows = store.load("quantize")
    assert len(rows) == 1
    assert rows[0]["root_cause"] == "second"  # updated in place


def test_format_ranks_matching_error_class(tmp_path: Path):
    store = TrajectoryStore(tmp_path / "traj")
    other = build_summary("quantize", "TypeError: x", None)
    other.error_class = "type_error"
    other.root_cause = "unrelated"
    match = build_summary("quantize", "RuntimeError: oom", None)
    match.error_class = "oom"
    match.root_cause = "sharded across gpus"
    match.tried_and_rejected = ["oom: lowered batch"]
    store.save("quantize", other)
    store.save("quantize", match)
    text = store.format_for_prompt("quantize", error_class="oom", limit=1)
    assert "sharded across gpus" in text
    assert "dead ends" in text.lower()
    assert "unrelated" not in text


def test_format_empty_when_no_trajectories(tmp_path: Path):
    store = TrajectoryStore(tmp_path / "traj")
    assert store.format_for_prompt("quantize") == ""


# ── KnowledgeStore integration ─────────────────────────────────────────────────
def test_knowledge_records_and_counts_trajectory(tmp_path: Path):
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    summary = build_summary("quantize", "RuntimeError: oom", None, tier="opus")
    kn.record_trajectory(summary)
    assert kn.stats()["trajectories"] == 1
    assert (tmp_path / "kn" / "trajectories" / "quantize.jsonl").is_file()


def test_merge_tree_dedups_trajectories_by_signature(tmp_path: Path):
    src = tmp_path / "remote" / "trajectories"
    src.mkdir(parents=True)
    row = {"signature": "RuntimeError: oom", "root_cause": "remote wins"}
    (src / "quantize.jsonl").write_text(json.dumps(row) + "\n")

    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    local_row = {"signature": "RuntimeError: oom", "root_cause": "local"}
    (kn.local_dir / "trajectories" / "quantize.jsonl").write_text(
        json.dumps(local_row) + "\n")

    kn._merge_tree(tmp_path / "remote", kn.local_dir)
    rows = kn.trajectories.load("quantize")
    assert len(rows) == 1
    assert rows[0]["root_cause"] == "remote wins"


# ── end-to-end: fix-loop records a trajectory on a successful agent fix ─────────
def test_fixloop_records_trajectory_on_agent_fix(tmp_path: Path):
    kn = KnowledgeStore(local_dir=tmp_path / "kn")
    err = "RuntimeError: CUDA out of memory"
    calls = {"n": 0}

    def run_phase() -> PhaseResult:
        calls["n"] += 1
        # first two runs fail, the run after the 2nd agent attempt passes
        return PhaseResult(ok=calls["n"] > 2, log_text=err)

    def agent(_t, _p, log_path, _s):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rc = "first guess" if calls["n"] == 1 else "pin single gpu"
        log_path.write_text(
            f"VERDICT: FIXABLE\nERROR_CLASS: oom\n"
            f"ROOT_CAUSE_HYPOTHESIS: {rc}\nFIX_PLAN: {rc}\n")
        return log_path.read_text()

    loop = FixLoop(
        "quantize", run_phase, config=FixLoopConfig(tiers=["openclaw"]),
        out_dir=tmp_path / "out", run_agent=agent, knowledge=kn, run_id="run42",
    )
    result = loop.run()
    assert result.ok is True
    rows = kn.trajectories.load("quantize")
    assert len(rows) == 1
    traj = rows[0]
    assert traj["error_class"] == "oom"
    assert traj["run_id"] == "run42"
    # earlier hypothesis becomes a dead end; winning one is excluded
    assert traj["tried_and_rejected"] == ["oom: first guess"]
    assert "pin single gpu" not in traj["tried_and_rejected"]

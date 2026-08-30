"""Knowledge-extraction tests: ATIF trajectory -> distilled knowledge entries."""

import json
from pathlib import Path

from click.testing import CliRunner

from aqa.agent import knowledge_extract as ke
from aqa.agent.trajectories import TrajectoryStore
from aqa.cli import main


def _atif(*, solved=True, with_error=True):
    steps = [
        {"step_id": 1, "source": "user", "message": "fix the bug"},
    ]
    if with_error:
        steps.append({
            "step_id": 2, "source": "agent", "message": "reading the file",
            "tool_calls": [{"tool_call_id": "c1", "function_name": "bash",
                            "arguments": {"command": "cat app.py"}}],
            "observation": {"results": [{"source_call_id": "c1",
                                         "content": "Traceback (most recent call last):\nRuntimeError: boom",
                                         "extra": {"exit_code": 1}}]},
        })
    steps.append({
        "step_id": 3, "source": "agent", "message": "the import path was wrong; fixed it",
        "tool_calls": [{"tool_call_id": "c2", "function_name": "bash",
                        "arguments": {"command": "sed -i s/old/new/ app.py"}}],
    })
    steps.append({"step_id": 4, "source": "agent", "message": "done"})
    return {
        "schema_version": "ATIF-v1.7", "session_id": "s1",
        "agent": {"name": "aqa-react", "version": "0.1", "model_name": "m"},
        "steps": steps,
        "extra": {"solved": solved, "reward": 1.0 if solved else 0.0},
    }


def test_extract_error_to_progress_entry():
    entries = ke.extract_from_atif(_atif(), run_id="r1")
    assert len(entries) == 1
    e = entries[0]
    assert e["error_class"]  # classified
    assert "RuntimeError" in e["exc_type"] or "RuntimeError" in e["signature"] or e["signature"]
    assert e["tried_and_rejected"] == ["cat app.py"]
    assert "import path" in e["root_cause"]  # the next agent message is the root-cause line
    assert "sed -i" in e["fix_rationale"]
    assert e["outcome"] == "fixed"
    assert e["run_id"] == "r1"


def test_extract_skips_clean_trajectory():
    entries = ke.extract_from_atif(_atif(with_error=False))
    assert entries == []


def test_extract_skips_error_without_progress():
    atif = _atif()
    atif["steps"] = atif["steps"][:2]  # error is the last thing that happened
    assert ke.extract_from_atif(atif) == []


def test_summarizer_can_refine_fields():
    def fake_summary(atif):
        return {"root_cause": "agent says: the import path", "fix_rationale": "agent fix"}

    entries = ke.extract_from_atif(_atif(), summarizer=fake_summary)
    assert entries[0]["root_cause"] == "agent says: the import path"
    assert entries[0]["fix_rationale"] == "agent fix"


def test_summarizer_failure_falls_back():
    def boom(atif):
        raise RuntimeError("nope")

    entries = ke.extract_from_atif(_atif(), summarizer=boom)
    assert len(entries) == 1  # rule-based entry kept


def test_extract_from_file_and_write_back(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "trajectories.jsonl").write_text(json.dumps(_atif()) + "\n")
    entries = ke.extract_from_file(run)
    assert len(entries) == 1

    root = tmp_path / "knowledge"
    written = ke.write_back(entries, root)
    assert written == 1
    store = TrajectoryStore(root)
    loaded = store.load("trajectory")
    assert len(loaded) == 1
    assert loaded[0]["root_cause"]
    assert loaded[0]["tier"] == "distill"


def test_cli_distill(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "trajectories.jsonl").write_text(json.dumps(_atif()) + "\n")
    root = tmp_path / "kb"
    result = CliRunner().invoke(
        main, ["traces", "distill", "--path", str(run), "--knowledge-root", str(root)])
    assert result.exit_code == 0
    assert "distilled 1" in result.output
    assert (root / "trajectory.jsonl").exists() or (root / "trajectories").exists()


def test_cli_distill_no_patterns(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "trajectories.jsonl").write_text(json.dumps(_atif(with_error=False)) + "\n")
    result = CliRunner().invoke(main, ["traces", "distill", "--path", str(run),
                                       "--knowledge-root", str(tmp_path / "kb")])
    assert result.exit_code == 0
    assert "no knowledge entries" in result.output

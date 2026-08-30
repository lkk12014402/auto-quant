"""Layered trajectory publishing tests (staging local-first; upload mocked)."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aqa.agent import traces_publish as tp
from aqa.cli import main
from aqa.logging_util import AqaError


def _atif(session_id, *, with_tool=True):
    steps = [
        {"step_id": 1, "source": "user", "message": "fix the bug"},
        {"step_id": 2, "source": "agent", "message": "reading",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "bash",
                         "arguments": {"command": "cat app.py"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": "ok"}]}},
        {"step_id": 3, "source": "agent", "message": "done"},
    ]
    if not with_tool:
        steps = [steps[0], steps[2]]
    return {"schema_version": "ATIF-v1.7", "session_id": session_id,
            "agent": {"name": "aqa-react", "version": "0.1"}, "steps": steps,
            "extra": {"solved": True, "reward": 1.0}}


def _write_run(run_dir, atifs):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories.jsonl").write_text(
        "\n".join(json.dumps(a) for a in atifs) + "\n")
    return run_dir


def test_stage_archive_layout(tmp_path):
    run = _write_run(tmp_path / "run1", [_atif("sess/a b", with_tool=True)])
    dest = tmp_path / "staged"
    manifest = tp.stage_trajectory_archive([run], dest)
    # session id sanitized into a safe path segment
    sess = "sess_a_b"
    root = dest / "trajectories" / sess
    assert (root / "trajectory.json").exists()
    assert (root / "dataset_sft.jsonl").exists()
    assert (root / "dataset_rl.jsonl").exists()
    # ATIF round-trips intact
    atif = json.loads((root / "trajectory.json").read_text())
    assert atif["session_id"] == "sess/a b"
    # both views present and shaped right
    sft = json.loads((root / "dataset_sft.jsonl").read_text().strip())
    assert "messages" in sft and "tools" in sft
    rl = json.loads((root / "dataset_rl.jsonl").read_text().strip())
    assert "responses_create_params" in rl and "verifier" in rl
    assert rl["verifier"]["solved"] is True
    assert manifest["sessions"] == 1
    assert (dest / "manifest.json").exists()


def test_stage_skips_dataset_row_when_no_signal(tmp_path):
    # user + single agent text step only -> no RL row needs input msgs, SFT has content
    run = _write_run(tmp_path / "run1", [_atif("s1", with_tool=False)])
    dest = tmp_path / "staged"
    tp.stage_trajectory_archive([run], dest)
    root = dest / "trajectories" / "s1"
    assert (root / "trajectory.json").exists()  # raw session always archived
    assert (root / "dataset_sft.jsonl").exists()
    assert (root / "dataset_rl.jsonl").exists()


def test_publish_requires_token(tmp_path, monkeypatch):
    dest = tmp_path / "staged"
    tp.stage_trajectory_archive([_write_run(tmp_path / "r", [_atif("s1")])], dest)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(AqaError, match="token"):
        tp.publish_archive(dest, "org/ds")


def test_publish_uploads_staged_tree(tmp_path, monkeypatch):
    dest = tmp_path / "staged"
    tp.stage_trajectory_archive([_write_run(tmp_path / "r", [_atif("s1")])], dest)
    monkeypatch.setenv("HF_TOKEN", "fake")
    calls = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, repo_type, exist_ok):
            calls["repo"] = (repo_id, repo_type, exist_ok)

        def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type):
            calls["upload"] = (folder_path, path_in_repo, repo_id, repo_type)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    url = tp.publish_archive(dest, "org/ds")
    assert calls["repo"] == ("org/ds", "dataset", True)
    assert calls["upload"][2:] == ("org/ds", "dataset")
    assert "/datasets/org/ds/" in url


def test_cli_publish_stage_only(tmp_path):
    run = _write_run(tmp_path / "run1", [_atif("s1")])
    stage = tmp_path / "stage"
    result = CliRunner().invoke(main, ["traces", "publish", "--path", str(run),
                                       "--stage-only", "--stage-dir", str(stage)])
    assert result.exit_code == 0, result.output
    assert "staged 1 session(s)" in result.output
    assert (stage / "trajectories" / "s1" / "trajectory.json").exists()


def test_cli_publish_needs_repo_without_stage_only(tmp_path):
    run = _write_run(tmp_path / "run1", [_atif("s1")])
    result = CliRunner().invoke(main, ["traces", "publish", "--path", str(run)])
    assert result.exit_code != 0
    assert "needs --repo" in result.output

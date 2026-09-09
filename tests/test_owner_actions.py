"""Tests for owner_actions manifest section, log-action CLI, model card details."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aqa.cli import main


def test_log_action_appends_and_manifest_embeds(tmp_path):
    from aqa.config import RunConfig
    from aqa.eval.registry import BenchmarkResult, EvalResult
    from aqa.report.manifest import build_manifest

    out = tmp_path / "run"
    out.mkdir()
    r = CliRunner().invoke(main, ["log-action", str(out), "--kind", "env",
                                  "--text", "换了 transformers 补丁 wheel",
                                  "--command", "uv pip install /opt/wheels/transformers-5.16.1*.whl"])
    assert r.exit_code == 0, r.output
    lines = (out / "owner_actions.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["kind"] == "env"

    res = EvalResult(model="org/m", backend="vllm")
    res.add(BenchmarkResult(name="piqa", score=0.8, metric="acc", num_samples=10))
    cfg = RunConfig(model="org/m", output={"dir": str(out), "sinks": []})
    m = build_manifest(cfg, res, run_id="t", out_dir=out)
    d = m.to_dict()
    assert d["owner_actions"] and d["owner_actions"][0]["kind"] == "env"
    assert "transformers-5.16.1" in d["owner_actions"][0]["command"]


def test_manifest_without_owner_actions_is_none(tmp_path):
    from aqa.config import RunConfig
    from aqa.eval.registry import EvalResult
    from aqa.report.manifest import build_manifest

    cfg = RunConfig(model="org/m", output={"dir": str(tmp_path), "sinks": []})
    m = build_manifest(cfg, EvalResult(model="org/m", backend="hf"),
                       run_id="t", out_dir=tmp_path)
    assert m.to_dict()["owner_actions"] is None


def test_model_card_details_section():
    from aqa.report.sinks import RunMeta, build_model_card

    card = build_model_card(
        RunMeta(model="org/base", scheme="MXFP8", method="model_free"),
        repo_name="base-MXFP8",
        results={"gsm8k": 0.9},
        commands={"quantize": "auto-round --model_name org/base --scheme MXFP8 --model_free ...",
                  "quantize_source": "builder",
                  "evaluate": ["lm_eval --model vllm ..."]},
        versions={"auto-round": "0.15.0", "lm-eval": "0.4.12",
                  "transformers": "5.16.1", "vllm": "0.11.0", "torch": "2.13.0"},
    )
    assert "## Reproduce & Environment" in card
    assert "auto-round --model_name org/base" in card
    assert "lm_eval --model vllm" in card
    assert "| auto-round | 0.15.0 |" in card and "| vllm | 0.11.0 |" in card
    # 无 commands/versions 时保持旧结构（向后兼容）
    plain = build_model_card(RunMeta(model="org/base", scheme="MXFP8"), repo_name="x")
    assert "Reproduce & Environment" not in plain


def test_capture_patch_cli(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "base", "--allow-empty"], cwd=repo, check=True)
    (repo / "a.txt").write_text("v1")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "add a"], cwd=repo, check=True)
    (repo / "a.txt").write_text("v2")  # working-tree fix

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    r = CliRunner().invoke(main, ["capture-patch", str(run_dir), "--repo", str(repo)])
    assert r.exit_code == 0, r.output
    patch = run_dir / "patches" / "owner" / "repo.patch"
    assert patch.is_file() and "-v1" in patch.read_text() and "+v2" in patch.read_text()
    actions = (run_dir / "owner_actions.jsonl").read_text()
    assert '"kind": "patch"' in actions

    # clean tree -> skipped, no action logged
    (repo / "a.txt").write_text("v1")
    run2 = tmp_path / "run2"
    run2.mkdir()
    r2 = CliRunner().invoke(main, ["capture-patch", str(run2), "--repo", str(repo)])
    assert r2.exit_code == 0 and "clean tree" in r2.output
    assert not (run2 / "owner_actions.jsonl").exists()


def test_publish_patch_dry_run(tmp_path):
    import subprocess as sp
    # 造：upstream 裸仓 + fork 克隆（origin=fork 裸仓，upstream 已 fetch）
    up_src = tmp_path / "up_src"
    up_src.mkdir()
    sp.run(["git", "init", "-q"], cwd=up_src, check=True)
    (up_src / "f.py").write_text("x = 1\n")
    sp.run(["git", "add", "-A"], cwd=up_src, check=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
           cwd=up_src, check=True)
    fork = tmp_path / "fork"
    sp.run(["git", "clone", "-q", str(up_src), str(fork)], check=True)
    # patch: x = 1 -> x = 2
    patch = tmp_path / "fix.patch"
    patch.write_text(
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    sp.run(["git", "remote", "add", "upstream", str(up_src)], cwd=fork, check=True)
    sp.run(["git", "fetch", "-q", "upstream"], cwd=fork, check=True)
    base = sp.run(["git", "-C", str(fork), "branch", "--show-current"],
                  capture_output=True, text=True).stdout.strip() or "master"
    r = CliRunner().invoke(main, ["publish-patch", str(patch), "--repo-dir", str(fork),
                                  "--base", base, "--title", "fix f", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert f"upstream/{base} tree): OK" in r.output, r.output
    assert "dry-run" in r.output

    # 一个打不上的补丁要如实报 FAILS
    bad = tmp_path / "bad.patch"
    bad.write_text(
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-nope = 0\n+x = 2\n")
    r2 = CliRunner().invoke(main, ["publish-patch", str(bad), "--repo-dir", str(fork),
                                   "--base", base, "--title", "t", "--dry-run"])
    assert r2.exit_code == 0 and "FAILS" in r2.output


def test_session_sediment_end_to_end(tmp_path, monkeypatch):
    """run 目录里的 fix-loop events + owner sessions/ 都被转成 ATIF。"""
    import json as _json
    from aqa.agent import session_sediment as ss

    run = tmp_path / "run"
    fixdir = run / "logs" / "agent_fixes" / "quantize"
    fixdir.mkdir(parents=True)
    (fixdir / "attempt1.events.jsonl").write_text(
        _json.dumps({"type": "session.start", "data": {"sessionId": "fix-sid-1"}}) + "\n"
        + _json.dumps({"type": "user.message", "data": {"content": "修这个错"}}) + "\n"
        + _json.dumps({"type": "assistant.message", "data": {"content": "看看",
                                                             "toolRequests": []}}) + "\n")
    owner = run / "sessions"
    owner.mkdir()
    (owner / "copilot_owner.jsonl").write_text(
        _json.dumps({"type": "session.start", "data": {"sessionId": "owner-sid-1"}}) + "\n"
        + _json.dumps({"type": "user.message", "data": {"content": "任务"}}) + "\n"
        + _json.dumps({"type": "assistant.message", "data": {"content": "做", "toolRequests": []}}) + "\n")

    stage = tmp_path / "stage"
    staged = ss.sediment_sessions(run, "run-1", stage_root=stage)
    names = sorted(p.name for p in staged)
    assert names == ["fix-sid-1.jsonl", "owner-sid-1.jsonl"], names
    # ATIF 结构 + 脱敏在适配器收口处已执行
    rec = _json.loads((stage / "run-1" / "fix-sid-1.jsonl").read_text())
    assert rec["session_id"] == "fix-sid-1" and len(rec["steps"]) >= 2

    # push_sessions：路径打到 dataset 的 trajectories/（与 knowledge/ 并列）
    from aqa.agent.knowledge import KnowledgeStore
    calls = {}

    class _FakeApi:
        def __init__(self, **kw): pass

        def upload_folder(self, **kw):
            calls.update(kw)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    store = KnowledgeStore(local_dir=tmp_path / "kn", repo="org/ds")
    monkeypatch.setattr("aqa.agent.knowledge.resolve_hf_token",
                        lambda *a, **k: "tok")
    assert store.push_sessions(stage / "run-1")
    assert calls["path_in_repo"] == "trajectories"

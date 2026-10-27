"""Local-first knowledge sync: outbox, sha256 state, pull/push, sessions."""

from __future__ import annotations

import json
from pathlib import Path

from aqa.agent.knowledge import KnowledgeStore
from aqa.agent.knowledge_sync import KnowledgeSync


def _store(tmp_path: Path, *, repo: str | None = None, push: bool = True) -> KnowledgeStore:
    return KnowledgeStore(local_dir=tmp_path / "knowledge", repo=repo,
                          path_in_repo="knowledge", push=push)


def _write_lesson(store: KnowledgeStore, name: str = "quantize") -> Path:
    path = Path(store.local_dir) / "lessons" / f"{name}.jsonl"
    path.write_text(json.dumps({"signature": "sig-1", "outcome": "fixed"}) + "\n")
    return path


def test_local_only_is_not_an_error(tmp_path: Path):
    store = _store(tmp_path)
    sync = KnowledgeSync(store)
    assert sync.pull().status == "local_only"
    assert sync.push().status == "local_only"
    assert sync.status()["repo"] is None


def test_push_only_changed_files_and_marks_state(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    lesson = _write_lesson(store)
    captured: list[list[str] | None] = []

    def _push_up(allow_patterns=None):
        captured.append(allow_patterns)
        return True

    store.push_up = _push_up  # type: ignore[method-assign]
    sync = KnowledgeSync(store)
    assert sync.pending() == ["lessons/quantize.jsonl"]
    report = sync.push()
    assert report.status == "pushed"
    assert report.pushed == 1
    assert captured == [["lessons/quantize.jsonl"]]
    assert sync.pending() == []

    # unchanged -> no upload
    report2 = sync.push()
    assert report2.status == "up_to_date"
    assert captured == [["lessons/quantize.jsonl"]]
    assert lesson.is_file()


def test_push_failure_stays_pending(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    _write_lesson(store)
    store.push_up = lambda allow_patterns=None: False  # type: ignore[method-assign]
    sync = KnowledgeSync(store)
    report = sync.push()
    assert report.status == "pending"
    assert report.pending == ["lessons/quantize.jsonl"]


def test_pull_records_sync_state(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    store.sync_down = lambda: True  # type: ignore[method-assign]
    sync = KnowledgeSync(store)
    report = sync.pull()
    assert report.status == "pulled"
    assert report.last_pull
    state = json.loads((Path(store.local_dir) / ".sync_state.json").read_text())
    assert state["repo"] == "org/knowledge"
    assert state["last_pull"]


def test_push_sessions_uploads_staged_atif(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    stage = tmp_path / "sessions" / "run-1"
    stage.mkdir(parents=True)
    (stage / "s1.jsonl").write_text("{}\n")
    called: list[Path] = []

    def _push_sessions(path):
        called.append(Path(path))
        return True

    store.push_sessions = _push_sessions  # type: ignore[method-assign]
    sync = KnowledgeSync(store)
    report = sync.push_sessions(stage)
    assert report.sessions == 1
    assert report.status == "pushed"
    assert called == [stage]


def test_sync_state_is_not_a_pushed_file(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    _write_lesson(store)
    sync = KnowledgeSync(store)
    sync._save_state()
    assert ".sync_state.json" not in sync.pending()


def test_finalize_knowledge_writes_sync_artifact(tmp_path: Path):
    from aqa.config import parse_run_config
    from aqa.pipeline import _finalize_knowledge

    class _FakeSync:
        def push(self):
            return self

        def push_sessions(self, stage):
            return self

        def status(self):
            return {"status": "pushed", "pushed": 1, "sessions": 0,
                    "pending": [], "errors": []}

    class _FakeKnowledge:
        local_dir = tmp_path / "knowledge"
        sync = _FakeSync()

        def stats(self):
            return {"replay_hit": 0, "reference_used": 0, "agent_fixed": 0,
                    "unresolved": 0, "hit_rate": 0.0}

    cfg = parse_run_config({
        "model": "org/m",
        "agent": {"knowledge": {"sediment_sessions": False}},
    })
    _finalize_knowledge(_FakeKnowledge(), run_cfg=cfg, out_dir=tmp_path, run_id="r")
    data = json.loads((tmp_path / "knowledge_sync.json").read_text())
    assert data["status"] == "pushed"
    assert data["pushed"] == 1


def test_make_knowledge_is_independent_of_fix_loop(tmp_path: Path):
    from aqa.config import parse_run_config
    from aqa.pipeline import _make_knowledge

    cfg = parse_run_config({
        "model": "org/m",
        "quant": {"scheme": "MXFP4"},
        "agent": {"fix_loop": False,
                  "knowledge": {"local_dir": str(tmp_path / "kn")}},
    })
    store = _make_knowledge(cfg)
    assert store is not None
    assert store.sync.mode == "auto"
    assert store.sync.status()["status"] == "local_only"


def test_finalize_knowledge_wires_session_sediment(tmp_path: Path, monkeypatch):
    from aqa.config import parse_run_config
    from aqa.pipeline import _finalize_knowledge

    stage = tmp_path / "sessions" / "run-1"
    stage.mkdir(parents=True)
    (stage / "s1.jsonl").write_text("{}\n")
    monkeypatch.setattr(
        "aqa.agent.session_sediment.sediment_sessions",
        lambda out_dir, run_id: [stage / "s1.jsonl"])

    class _FakeSync:
        def __init__(self):
            self.session_dirs = []

        def push(self):
            return self

        def push_sessions(self, path):
            self.session_dirs.append(Path(path))
            return self

        def status(self):
            return {"status": "pushed", "pushed": 1, "sessions": 1,
                    "pending": [], "errors": []}

    class _FakeKnowledge:
        local_dir = tmp_path / "knowledge"
        sync = _FakeSync()

        def stats(self):
            return {"replay_hit": 0, "reference_used": 0, "agent_fixed": 0,
                    "unresolved": 0, "hit_rate": 0.0}

    cfg = parse_run_config({
        "model": "org/m",
        "agent": {"knowledge": {"sediment_sessions": True}},
    })
    _finalize_knowledge(_FakeKnowledge(), run_cfg=cfg, out_dir=tmp_path, run_id="run-1")
    assert _FakeKnowledge.sync.session_dirs == [stage]


def test_status_does_not_hide_post_push_errors(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    sync = KnowledgeSync(store)
    sync.report.status = "pushed"
    sync.report.errors.append("sediment: adapter failed")
    assert sync.status()["status"] == "pending"


def test_finalize_builds_run_scoped_snapshot_not_shared_cache(tmp_path: Path):
    from aqa.config import parse_run_config
    from aqa.pipeline import _finalize_knowledge

    store = _store(tmp_path)  # no repo -> results sink is the only destination
    sync = KnowledgeSync(store)
    store.sync = sync
    old = Path(store.local_dir) / "lessons" / "old.jsonl"
    old.write_text('{"signature": "old"}\n')  # pre-existing, must not be staged
    sync.mark_baseline()
    new = Path(store.local_dir) / "lessons" / "new.jsonl"
    new.write_text('{"signature": "new"}\n')

    cfg = parse_run_config({
        "model": "org/m",
        "quant": {"scheme": "MXFP4"},
        "agent": {"knowledge": {"local_dir": str(store.local_dir),
                                "sediment_sessions": False}},
    })
    out = tmp_path / "run"
    out.mkdir()
    snapshot = _finalize_knowledge(store, run_cfg=cfg, out_dir=out, run_id="run-1")
    assert snapshot == out / "knowledge"
    assert (snapshot / "lessons" / "new.jsonl").is_file()
    assert not (snapshot / "lessons" / "old.jsonl").exists()
    assert json.loads((out / "knowledge_sync.json").read_text())["snapshot_files"] == 1


def test_background_mode_defers_push(tmp_path: Path):
    store = _store(tmp_path, repo="org/knowledge")
    _write_lesson(store)
    called = []
    store.push_up = lambda allow_patterns=None: called.append(allow_patterns) or True  # type: ignore[method-assign]
    sync = KnowledgeSync(store, mode="background")
    report = sync.push()
    assert report.status == "pending"
    assert report.pending == ["lessons/quantize.jsonl"]
    assert called == []

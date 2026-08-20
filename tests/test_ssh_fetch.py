"""Tests for the SFTP fetch-back exclusions (no more multi-GB weight downloads /
hangs) and the agent-session archival that mirrors local_dispatch's report inputs.
"""

from __future__ import annotations

import stat as _stat

from aqa.agent.backends import _archive_agent_session
from aqa.executors.ssh import _sftp_fetch_tree


class _Attr:
    def __init__(self, filename: str, *, isdir: bool = False, size: int = 10, mode: int | None = None):
        self.filename = filename
        self.st_size = size
        if mode is not None:
            self.st_mode = mode
        else:
            self.st_mode = (_stat.S_IFDIR | 0o755) if isdir else (_stat.S_IFREG | 0o644)


class _FakeSftp:
    """Minimal SFTP: a dict tree of {remote_dir: [attrs]} + file contents."""

    def __init__(self, tree: dict[str, list[_Attr]]):
        self.tree = tree
        self.got: list[str] = []

    def listdir_attr(self, path: str) -> list[_Attr]:
        if path not in self.tree:
            raise OSError("no such dir")
        return self.tree[path]

    def get(self, rpath: str, lpath: str) -> None:
        self.got.append(rpath)
        with open(lpath, "w") as fh:
            fh.write("x")


def test_fetch_skips_weights_caches_and_oversized(tmp_path):
    tree = {
        "/run": [
            _Attr("eval", isdir=True),
            _Attr("quantized", isdir=True),          # excluded dir
            _Attr("__pycache__", isdir=True),        # excluded dir
            _Attr("results.json"),                   # kept
            _Attr("model.safetensors", size=5_000),  # excluded suffix
            _Attr("huge.log", size=10**12),          # oversized backstop
            _Attr("pipe", mode=_stat.S_IFIFO | 0o644),  # non-regular
        ],
        "/run/eval": [_Attr("piqa.jsonl")],          # kept
        "/run/quantized": [_Attr("w.safetensors", size=9_000)],
        "/run/__pycache__": [_Attr("x.pyc")],
    }
    sftp = _FakeSftp(tree)
    count = _sftp_fetch_tree(sftp, "/run", tmp_path)
    assert count == 2  # results.json + eval/piqa.jsonl
    assert set(sftp.got) == {"/run/results.json", "/run/eval/piqa.jsonl"}
    assert not (tmp_path / "quantized").exists() or not list((tmp_path / "quantized").iterdir())
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "eval" / "piqa.jsonl").is_file()


def test_fetch_missing_dir_returns_zero(tmp_path):
    assert _sftp_fetch_tree(_FakeSftp({}), "/nope", tmp_path) == 0


def test_archive_copilot_events(tmp_path):
    state = tmp_path / ".copilot" / "session-state" / "sess123"
    state.mkdir(parents=True)
    (state / "events.jsonl").write_text('{"role":"assistant","content":"fix"}\n')
    log_path = tmp_path / "logs" / "agent_fixes" / "evaluate" / "attempt_1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stream")
    _archive_agent_session("copilot", "sess123", log_path, {"HOME": str(tmp_path)})
    archived = log_path.with_name("attempt_1.events.jsonl")
    assert archived.is_file()
    assert "assistant" in archived.read_text()


def test_archive_openclaw_session(tmp_path):
    sessions = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "sessABC.jsonl").write_text('{"type":"message"}\n')
    log_path = tmp_path / "attempt_2.log"
    log_path.write_text("stream")
    _archive_agent_session("openclaw", "sessABC", log_path, {"HOME": str(tmp_path)})
    assert log_path.with_name("attempt_2.session.jsonl").is_file()


def test_archive_missing_session_is_noop(tmp_path):
    log_path = tmp_path / "attempt_3.log"
    log_path.write_text("stream")
    _archive_agent_session("copilot", "absent", log_path, {"HOME": str(tmp_path)})
    assert not log_path.with_name("attempt_3.events.jsonl").exists()

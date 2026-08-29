"""Tests for the tee-ing subprocess runner (live stream + persisted log)."""

from __future__ import annotations

import sys

import pytest

from aqa.proc import stream_subprocess


def test_stream_captures_output_and_rc(tmp_path):
    log = tmp_path / "run.log"
    rc = stream_subprocess([sys.executable, "-c", "print('hello world')"], log)
    assert rc == 0
    assert "hello world" in log.read_text()


def test_stream_propagates_nonzero_exit(tmp_path):
    log = tmp_path / "run.log"
    rc = stream_subprocess(["sh", "-c", "echo boom >&2; exit 7"], log)
    assert rc == 7
    assert "boom" in log.read_text()


def test_stream_merges_stderr(tmp_path):
    log = tmp_path / "run.log"
    rc = stream_subprocess(["sh", "-c", "echo out; echo err 1>&2"], log)
    assert rc == 0
    text = log.read_text()
    assert "out" in text
    assert "err" in text


def test_stream_carriage_return_progress_persisted(tmp_path):
    log = tmp_path / "run.log"
    script = "import sys\nfor i in range(3): sys.stdout.write('\\rP%d' % i)\nprint('\\nfin')"
    rc = stream_subprocess([sys.executable, "-c", script], log)
    assert rc == 0
    data = log.read_text()
    assert "P0" in data and "P2" in data and "fin" in data


def test_stream_pipe_fallback(tmp_path):
    log = tmp_path / "run.log"
    rc = stream_subprocess([sys.executable, "-c", "print('no-pty')"], log, use_pty=False)
    assert rc == 0
    assert "no-pty" in log.read_text()


def test_stream_creates_parent_dir(tmp_path):
    log = tmp_path / "nested" / "deep" / "run.log"
    rc = stream_subprocess([sys.executable, "-c", "print('ok')"], log)
    assert rc == 0
    assert log.exists()


def test_stream_timeout_kills_and_returns_124(tmp_path):
    from aqa.proc import TIMEOUT_RC

    log = tmp_path / "run.log"
    rc = stream_subprocess(
        [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
        log, timeout=1,
    )
    assert rc == TIMEOUT_RC
    text = log.read_text()
    assert "start" in text
    assert "TIMEOUT" in text


def test_stream_timeout_pipe_fallback(tmp_path):
    from aqa.proc import TIMEOUT_RC

    log = tmp_path / "run.log"
    rc = stream_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        log, timeout=1, use_pty=False,
    )
    assert rc == TIMEOUT_RC


def test_stream_no_timeout_completes_normally(tmp_path):
    log = tmp_path / "run.log"
    rc = stream_subprocess([sys.executable, "-c", "print('done')"], log, timeout=30)
    assert rc == 0
    assert "done" in log.read_text()


def test_pty_gets_real_winsize_so_tqdm_renders(tmp_path):
    """A fresh pty defaults to 0x0 and tqdm renders nothing against it; the
    runner must set a real winsize so progress bars land in the log."""
    pytest.importorskip("tqdm")
    log = tmp_path / "run.log"
    child = (
        "import tqdm, time\n"
        "for _ in tqdm.tqdm(range(3), desc='barprobe'):\n"
        "    time.sleep(0.05)\n"
    )
    rc = stream_subprocess([sys.executable, "-c", child], log)
    assert rc == 0
    assert "barprobe" in log.read_text()


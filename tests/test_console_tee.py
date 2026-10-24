"""Tests for the whole-run console tee (fd-level stdout/stderr transcript)."""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

from aqa.console_tee import start_console_tee, stop_console_tee


def test_console_tee_captures_fd_level_output(tmp_path):
    """fd-level writes, python stdout/stderr and child stdout all land in the log,
    the live echo still reaches the real terminal, and nothing after stop leaks in."""
    out_dir = tmp_path / "run"
    script = textwrap.dedent(
        """
        import os, sys, subprocess
        from aqa.console_tee import start_console_tee, stop_console_tee
        h = start_console_tee(sys.argv[1])
        assert h is not None, "expected a real fd under a pipe stdout"
        os.write(1, b"RAW_FD_LINE\\n")                        # like proc._emit (bypasses sys.stdout)
        print("PRINT_LINE", flush=True)                       # python stdout buffer
        print("STDERR_LINE", file=sys.stderr, flush=True)     # logger writes here
        subprocess.run([sys.executable, "-c", "print('CHILD_LINE')"])
        stop_console_tee(h)
        sys.stdout.write("AFTER_STOP\\n"); sys.stdout.flush()
        """
    )
    proc = subprocess.run([sys.executable, "-c", script, str(out_dir)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    log = (out_dir / "logs" / "run.console.log").read_text()
    assert "RAW_FD_LINE" in log      # os.write to fd 1 captured (the key case)
    assert "PRINT_LINE" in log
    assert "STDERR_LINE" in log
    assert "CHILD_LINE" in log       # child inherits the redirected fd 1
    assert "AFTER_STOP" not in log   # tee stopped before this write

    # Live echo still reached the terminal (the subprocess's real stdout).
    assert "RAW_FD_LINE" in proc.stdout
    assert "AFTER_STOP" in proc.stdout


def test_console_tee_noops_without_real_fd(tmp_path, monkeypatch):
    """Under captured stdio (no backing fd) the tee degrades to a no-op."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    handle = start_console_tee(tmp_path / "run")
    assert handle is None
    assert not (tmp_path / "run" / "logs" / "run.console.log").exists()
    stop_console_tee(handle)  # None handle must be safe

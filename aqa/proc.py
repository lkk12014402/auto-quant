"""Run a subprocess while *teeing* its output: live to this process's stdout
(so it streams back over the SSH channel in real time) and to a persisted log
file (for debugging / upload).

The previous behaviour redirected a long tool's (lm_eval / auto-round) output
entirely into a log file, so a remote ``aqa run`` looked hung for the many
minutes of model load + eval even though the GPU was busy.  Streaming fixes that.

A pseudo-terminal is allocated for the child by default so terminal-aware tools
(tqdm progress bars, vllm) render live and stay line-buffered; a plain pipe is
used as a fallback where a pty is unavailable.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

TIMEOUT_RC = 124  # conventional exit code for a wall-clock timeout (matches `timeout(1)`)


def _emit(data: bytes) -> None:
    """Best-effort write of raw bytes to this process's stdout."""
    try:
        os.write(sys.stdout.fileno(), data)
    except (OSError, ValueError):  # stdout not backed by a real fd (captured)
        try:
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
        except Exception:  # noqa: BLE001, S110 - never let tee-logging kill the run
            pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate the child's whole process group (agents spawn sub-processes)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


def _stream_pipe(
    cmd: Sequence[str], log_file, env: Mapping[str, str] | None,
    timeout: float | None,
) -> int:
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(env) if env is not None else None,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdout is not None
    deadline = (time.monotonic() + timeout) if timeout else None
    fd = proc.stdout.fileno()
    while True:
        if deadline is not None and time.monotonic() > deadline:
            _emit_timeout(log_file, timeout)
            _kill_tree(proc)
            return TIMEOUT_RC
        rlist, _, _ = select.select([fd], [], [], 0.2)
        if fd in rlist:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            log_file.write(chunk)
            log_file.flush()
            _emit(chunk)
        elif proc.poll() is not None:
            break
    return proc.wait()


def _stream_pty(
    cmd: Sequence[str], log_file, env: Mapping[str, str] | None,
    timeout: float | None,
) -> int:
    import pty

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        env=dict(env) if env is not None else None,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave)
    deadline = (time.monotonic() + timeout) if timeout else None
    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                _emit_timeout(log_file, timeout)
                _kill_tree(proc)
                return TIMEOUT_RC
            try:
                rlist, _, _ = select.select([master], [], [], 0.2)
            except InterruptedError:  # pragma: no cover - signal during select
                continue
            if master in rlist:
                try:
                    data = os.read(master, 65536)
                except OSError:  # EIO on Linux once the child's slave closes
                    break
                if not data:
                    break
                log_file.write(data)
                log_file.flush()
                _emit(data)
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
    return proc.wait()


def _emit_timeout(log_file, timeout: float | None) -> None:
    msg = f"\n[TIMEOUT] process exceeded {timeout}s\n".encode()
    log_file.write(msg)
    log_file.flush()
    _emit(msg)


def stream_subprocess(
    cmd: Sequence[str],
    log_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    use_pty: bool = True,
    timeout: float | None = None,
) -> int:
    """Run ``cmd``, teeing combined stdout/stderr to stdout and ``log_path``.

    Returns the child's exit code (or :data:`TIMEOUT_RC` when ``timeout`` elapses,
    after killing the child's process group).  ``use_pty`` allocates a
    pseudo-terminal so progress bars render live; it falls back to a pipe
    automatically where a pty cannot be created (e.g. non-POSIX platforms).
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as log_file:
        if use_pty and hasattr(os, "openpty"):
            try:
                return _stream_pty(cmd, log_file, env, timeout)
            except OSError:  # pragma: no cover - pty exhaustion / unsupported
                pass
        return _stream_pipe(cmd, log_file, env, timeout)

 
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
    """Terminate the child's whole process group (agents spawn sub-processes).

    Pass ``pgid`` explicitly when the parent process may already be dead so the
    ``os.getpgid(proc.pid)`` lookup doesn't fail before SIGTERM is delivered
    to the orphaned descendants. When ``pgid`` is omitted we fall back to the
    proc's own pgrp at call time (works only while the proc is alive).
    """
    sigterm_args: tuple = ()
    if proc.pid is not None:
        try:
            sigterm_args = (os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not sigterm_args:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        os.killpg(*sigterm_args)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(sigterm_args[0], signal.SIGKILL)
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
    # Capture the pgrp IMMEDIATELY while the child is alive so we can SIGTERM
    # orphaned descendants even after the parent process has exited and
    # ``os.getpgid(proc.pid)`` would raise ProcessLookupError.
    pgid = os.getpgid(proc.pid)
    deadline = (time.monotonic() + timeout) if timeout else None
    fd = proc.stdout.fileno()
    rc: int | None = None
    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                _emit_timeout(log_file, timeout)
                rc = TIMEOUT_RC
                break
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
        if rc is None:
            rc = proc.wait()
    finally:
        # vLLM spawns EngineCore + workers via `multiprocessing.spawn`; when the
        # tool exits via Python exception (rc != 0, normal exit), those
        # children are orphaned and keep holding GPU memory (they are not in
        # the tool's wait queue and were never sent SIGTERM by Python).
        # ``start_new_session=True`` makes the tool's pgrp == its sid, so
        # ``killpg(pgid)`` reaches exactly its descendants and nothing else on
        # the box. Use the captured ``pgid`` (not ``os.getpgid(proc.pid)``) so
        # the SIGTERM lands even after the parent has exited.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                proc.kill()
    assert rc is not None
    return rc


def _stream_pty(
    cmd: Sequence[str], log_file, env: Mapping[str, str] | None,
    timeout: float | None,
) -> int:
    import pty

    master, slave = pty.openpty()
    try:
        # A fresh pty has a 0x0 winsize; tqdm (and any curses-style UI) renders
        # an empty/collapsed bar against it. Inherit the parent's terminal size
        # (COLUMNS/LINES aware, 80x24 fallback) so progress bars actually render.
        import fcntl
        import shutil
        import struct
        import termios

        size = shutil.get_terminal_size()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", size.lines, size.columns, 0, 0))
    except OSError:  # pragma: no cover - exotic platform; bar cosmetic only
        pass
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
    # Capture the pgrp IMMEDIATELY while the child is alive (see _stream_pipe).
    pgid = os.getpgid(proc.pid)
    deadline = (time.monotonic() + timeout) if timeout else None
    rc: int | None = None
    try:
        while True:
            if deadline is not None and time.monotonic() > deadline:
                _emit_timeout(log_file, timeout)
                rc = TIMEOUT_RC
                break
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
        if rc is None:
            rc = proc.wait()
    finally:
        # See _stream_pipe: SIGTERM the captured pgrp on every exit so
        # orphaned vLLM EngineCore / WorkerProc children don't pin GPU
        # memory across fix-loop retries.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                proc.kill()
        os.close(master)
    assert rc is not None
    return rc


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

"""Whole-run console tee: mirror this process's stdout+stderr to
``<out_dir>/logs/run.console.log`` while still showing live on the terminal.

``attach_run_log`` only tees the ``aqa.*`` logger (phase banners + ``=> score``
summaries) to ``pipeline.log``.  The *full* console — lm_eval result tables,
auto-round progress bars, any child stdout — is written straight to OS file
descriptors 1/2 by :func:`aqa.proc.stream_subprocess` (``os.write(fileno, ...)``)
and bypasses the Python ``sys.stdout`` object entirely.  A Python-level stdout
wrapper would therefore miss it.  We redirect the *OS-level* fds 1/2 into a pipe
and a daemon pump thread fans every chunk out to the original terminal fd and
the log file, so the persisted ``run.console.log`` is a faithful transcript of
what a human saw.

Best-effort by construction: if stdout is not backed by a real fd (pytest /
click capture replace it with an in-memory buffer) we no-op and return ``None``
so a run is never broken by logging plumbing.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConsoleTee:
    """Live handle for an active tee; opaque to callers (pass to :func:`stop_console_tee`)."""

    log_file: object
    reader_fd: int
    out_fd: int
    err_fd: int | None
    saved_out_fd: int
    saved_err_fd: int | None
    pump: threading.Thread


def _real_fd(stream: object) -> int | None:
    """Return the OS fd backing ``stream``, or ``None`` when it has no real fd."""
    try:
        fd = stream.fileno()  # type: ignore[attr-defined]
    except (OSError, ValueError, AttributeError):
        return None
    try:
        os.fstat(fd)
    except OSError:
        return None
    return fd


def start_console_tee(out_dir: "str | os.PathLike[str]") -> ConsoleTee | None:
    """Begin teeing fds 1/2 to ``<out_dir>/logs/run.console.log``.

    Returns a :class:`ConsoleTee` handle, or ``None`` when stdout is not a real
    fd (capture context) or the log file cannot be opened.  Never raises.
    """
    out_fd = _real_fd(sys.stdout)
    if out_fd is None:
        return None
    err_fd = _real_fd(sys.stderr)

    log_dir = Path(out_dir) / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / "run.console.log").open("ab")
    except OSError:
        return None

    # Flush Python buffers so nothing already written straddles the redirect.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - flushing is best-effort
            pass

    saved_out = os.dup(out_fd)
    saved_err = os.dup(err_fd) if err_fd is not None else None
    reader_fd, writer_fd = os.pipe()

    # Point the process stdout (and stderr, when real) at the pipe's write end.
    os.dup2(writer_fd, out_fd)
    if err_fd is not None:
        os.dup2(writer_fd, err_fd)
    os.close(writer_fd)  # fds 1/2 now hold the only write-ends of the pipe

    def _pump() -> None:
        while True:
            try:
                chunk = os.read(reader_fd, 65536)
            except OSError:
                break
            if not chunk:  # all write-ends closed -> EOF -> run ended
                break
            try:
                os.write(saved_out, chunk)  # live echo to the real terminal
            except OSError:
                pass
            try:
                log_file.write(chunk)
                log_file.flush()
            except Exception:  # noqa: BLE001 - never let tee-logging kill the run
                pass

    pump = threading.Thread(target=_pump, name="aqa-console-tee", daemon=True)
    pump.start()
    return ConsoleTee(log_file=log_file, reader_fd=reader_fd, out_fd=out_fd,
                      err_fd=err_fd, saved_out_fd=saved_out, saved_err_fd=saved_err,
                      pump=pump)


def stop_console_tee(handle: ConsoleTee | None) -> None:
    """Restore fds 1/2, drain the pump and close the log file.  Never raises."""
    if handle is None:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - best-effort
            pass
    # Restoring the saved fds drops the pipe's write-ends, so the pump reads EOF.
    try:
        os.dup2(handle.saved_out_fd, handle.out_fd)
        if handle.err_fd is not None and handle.saved_err_fd is not None:
            os.dup2(handle.saved_err_fd, handle.err_fd)
    except OSError:
        pass
    handle.pump.join(timeout=5)
    for fd in (handle.reader_fd, handle.saved_out_fd, handle.saved_err_fd):
        if fd is None:
            continue
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        handle.log_file.close()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best-effort
        pass

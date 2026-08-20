"""LocalSandbox -- a scratch directory on the host.

Fast and dependency-free, but *not* isolated: it runs commands directly on the
host, so it enables the unsafe-command guard by default.  Use it for trusted
tasks (your own harness smoke tests) or when a container runtime is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging_util import get_logger
from .base import BaseSandbox, ExecResult, SandboxError, is_unsafe_command

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

_log = get_logger("sandbox.local")

# A runner turns (argv, timeout, cwd, env) into (returncode, stdout, stderr,
# timed_out).  Injected so tests need no real subprocess.
CommandRunner = Callable[
    [list[str], "float | None", str, "dict[str, str]"],
    "tuple[int, str, str, bool]",
]


def _subprocess_runner(
    argv: list[str], timeout: float | None, cwd: str, env: dict[str, str]
) -> tuple[int, str, str, bool]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err, True
    return proc.returncode, proc.stdout, proc.stderr, False


@dataclass
class LocalSandbox(BaseSandbox):
    """Run commands in a temporary directory on the host via ``/bin/sh -c``."""

    name: str = "local"
    workdir: str | None = None
    shell: str = "/bin/sh"
    allow_unsafe: bool = False
    default_timeout: float | None = 600.0
    env: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = True
    runner: CommandRunner = _subprocess_runner
    _root: Path | None = field(default=None, init=False, repr=False)
    _ephemeral: bool = field(default=False, init=False, repr=False)

    # -- lifecycle ----------------------------------------------------------
    def _provision(self) -> None:
        if self.workdir:
            root = Path(self.workdir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._ephemeral = False
        else:
            root = Path(tempfile.mkdtemp(prefix="aqa-sbx-"))
            self._ephemeral = True
        self._root = root
        _log.debug("local sandbox at %s (ephemeral=%s)", root, self._ephemeral)

    def _teardown(self) -> None:
        if self._ephemeral and self._root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    @property
    def root(self) -> Path:
        if self._root is None:
            self._ensure_started()
        assert self._root is not None
        return self._root

    # -- operations ---------------------------------------------------------
    def _build_env(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        env: dict[str, str] = dict(os.environ) if self.inherit_env else {}
        env.update(self.env)
        if extra:
            env.update(extra)
        return env

    def _resolve(self, path: str) -> Path:
        """Resolve a task-relative path, refusing to escape the workdir."""
        root = self.root
        target = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if root not in target.parents and target != root:
            raise SandboxError(f"path {path!r} escapes the sandbox workdir")
        return target

    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        self._ensure_started()
        if not self.allow_unsafe and is_unsafe_command(command):
            raise SandboxError(
                f"refusing unsafe command on host sandbox: {command!r} "
                "(use an isolated backend or allow_unsafe=True)"
            )
        run_cwd = str(self._resolve(cwd)) if cwd else str(self.root)
        run_env = self._build_env(env)
        argv = [self.shell, "-c", command]
        limit = self.default_timeout if timeout is None else timeout
        start = time.monotonic()
        code, out, err, timed_out = self.runner(argv, limit, run_cwd, run_env)
        return ExecResult(
            command=command,
            returncode=code,
            stdout=out,
            stderr=err,
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
        )

    def write_file(self, path: str, content: str) -> None:
        self._ensure_started()
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        self._ensure_started()
        target = self._resolve(path)
        if not target.exists():
            raise SandboxError(f"no such file in sandbox: {path!r}")
        return target.read_text(encoding="utf-8")

"""DockerSandbox -- a throwaway container providing true isolation.

Because the container is isolated from the host, the unsafe-command guard is off
by default: the model may run anything inside it.  All Docker interaction goes
through an injectable :data:`DockerRunner`, so the unit tests exercise the full
argv/lifecycle logic without a real Docker daemon.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..logging_util import get_logger
from .base import BaseSandbox, ExecResult, SandboxError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

_log = get_logger("sandbox.docker")

# A docker runner turns (argv, timeout, stdin_text) into
# (returncode, stdout, stderr, timed_out).  Injected for testability.
DockerRunner = Callable[
    [list[str], "float | None", "str | None"],
    "tuple[int, str, str, bool]",
]


def _subprocess_docker_runner(
    argv: list[str], timeout: float | None, stdin_text: str | None
) -> tuple[int, str, str, bool]:
    try:
        proc = subprocess.run(
            argv,
            input=stdin_text,
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
class DockerSandbox(BaseSandbox):
    """Provision a detached container; ``exec`` via ``docker exec``."""

    name: str = "docker"
    image: str = "python:3.12-slim"
    workdir: str = "/workspace"
    docker_bin: str = "docker"
    shell: str = "/bin/sh"
    default_timeout: float | None = 600.0
    env: dict[str, str] = field(default_factory=dict)
    gpus: str | None = None
    network: str | None = None
    shm_size: str | None = None
    keepalive: str = "sleep infinity"
    container_name: str | None = None
    extra_run_args: list[str] = field(default_factory=list)
    runner: DockerRunner = _subprocess_docker_runner
    _container: str | None = field(default=None, init=False, repr=False)

    # -- lifecycle ----------------------------------------------------------
    def _provision(self) -> None:
        cname = self.container_name or f"aqa-sbx-{uuid.uuid4().hex[:12]}"
        argv = [
            self.docker_bin, "run", "-d", "--rm",
            "--name", cname,
            "-w", self.workdir,
        ]
        if self.gpus:
            argv += ["--gpus", self.gpus]
        if self.network:
            argv += ["--network", self.network]
        if self.shm_size:
            argv += ["--shm-size", self.shm_size]
        for key, val in self.env.items():
            argv += ["-e", f"{key}={val}"]
        argv += self.extra_run_args
        argv += [self.image, self.shell, "-c", self.keepalive]
        code, out, err, _ = self.runner(argv, 120.0, None)
        if code != 0:
            raise SandboxError(
                f"failed to start docker sandbox (image={self.image!r}): "
                f"{(err or out).strip()}"
            )
        self._container = cname
        # Ensure the workdir exists (some images lack it).
        self.runner(
            [self.docker_bin, "exec", cname, self.shell, "-c",
             f"mkdir -p {shlex.quote(self.workdir)}"],
            60.0, None,
        )
        _log.debug("docker sandbox %s up (image=%s)", cname, self.image)

    def _teardown(self) -> None:
        if self._container:
            self.runner(
                [self.docker_bin, "rm", "-f", self._container], 60.0, None
            )
            self._container = None

    @property
    def container(self) -> str:
        if self._container is None:
            self._ensure_started()
        assert self._container is not None
        return self._container

    # -- operations ---------------------------------------------------------
    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        self._ensure_started()
        argv = [self.docker_bin, "exec", "-w", cwd or self.workdir]
        if env:
            for key, val in env.items():
                argv += ["-e", f"{key}={val}"]
        argv += [self.container, self.shell, "-c", command]
        limit = self.default_timeout if timeout is None else timeout
        start = time.monotonic()
        code, out, err, timed_out = self.runner(argv, limit, None)
        return ExecResult(
            command=command,
            returncode=code,
            stdout=out,
            stderr=err,
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
        )

    def _abs(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.workdir.rstrip('/')}/{path}"

    def write_file(self, path: str, content: str) -> None:
        self._ensure_started()
        target = self._abs(path)
        parent = target.rsplit("/", 1)[0] or "/"
        argv = [
            self.docker_bin, "exec", "-i", self.container, self.shell, "-c",
            f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(target)}",
        ]
        code, out, err, _ = self.runner(argv, 120.0, content)
        if code != 0:
            raise SandboxError(f"write_file {path!r} failed: {(err or out).strip()}")

    def read_file(self, path: str) -> str:
        self._ensure_started()
        target = self._abs(path)
        argv = [
            self.docker_bin, "exec", self.container, "cat", target,
        ]
        code, out, _err, _ = self.runner(argv, 120.0, None)
        if code != 0:
            raise SandboxError(f"no such file in sandbox: {path!r}")
        return out

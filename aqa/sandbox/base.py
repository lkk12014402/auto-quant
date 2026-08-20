"""Sandbox protocol, result type, base mixin, and the unsafe-command guard."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..logging_util import AqaError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from types import TracebackType
    from typing import Self


class SandboxError(AqaError):
    """A sandbox could not be started, or a required operation failed."""


@dataclass
class ExecResult:
    """Outcome of a single :meth:`Sandbox.exec` call."""

    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True when the command exited 0 and did not time out."""
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """stdout plus stderr (stderr appended), for feeding an agent."""
        if self.stderr:
            return f"{self.stdout}\n{self.stderr}".strip("\n")
        return self.stdout


@runtime_checkable
class Sandbox(Protocol):
    """A disposable workspace in which task commands run."""

    name: str

    def start(self) -> None:
        """Provision the workspace (idempotent)."""
        ...

    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Run ``command`` (a shell string) inside the sandbox."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Create/overwrite ``path`` (relative to workdir) with ``content``."""
        ...

    def read_file(self, path: str) -> str:
        """Return the text content of ``path`` (relative to workdir)."""
        ...

    def close(self) -> None:
        """Tear the workspace down (idempotent)."""
        ...


# ---------------------------------------------------------------------------
# Unsafe-command guard (used by non-isolated backends, i.e. LocalSandbox).
# ---------------------------------------------------------------------------

# Patterns that could damage the host if run outside a container.  This is a
# best-effort guard, NOT a security boundary -- untrusted models must use an
# isolated backend (docker/apptainer).
_UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\b.*\s-[a-z]*[rf][a-z]*\b.*(\s|^)(/|~|\$HOME|\*)"),
    re.compile(r":\s*\(\s*\)\s*\{.*\|\s*:.*&\s*\}\s*;\s*:"),  # fork bomb
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|hd|mapper)"),
    re.compile(r">\s*/dev/(sd|nvme|hd)[a-z0-9]*\b"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff|init\s+0)\b"),
    re.compile(r"\bchmod\b\s+-R\s+0*\s+/(?:\s|$)"),
    re.compile(r"\b(sudo|doas)\b"),
    re.compile(r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b"),
    re.compile(r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh\b"),
)


def is_unsafe_command(command: str) -> bool:
    """Heuristically flag host-dangerous commands (fork bombs, ``rm -rf /`` ...)."""
    text = " ".join(command.split())
    return any(p.search(text) for p in _UNSAFE_PATTERNS)


def _quote_env_prefix(env: Mapping[str, str] | None) -> str:
    """Build a leading ``K=V K2=V2 `` prefix for a shell command."""
    if not env:
        return ""
    return "".join(f"{k}={shlex.quote(v)} " for k, v in env.items())


@dataclass
class BaseSandbox:
    """Context-manager + bookkeeping shared by concrete sandboxes.

    Subclasses implement :meth:`exec`, :meth:`write_file`, :meth:`read_file`,
    and the private :meth:`_provision` / :meth:`_teardown` hooks.
    """

    name: str = "sandbox"
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    # -- lifecycle ----------------------------------------------------------
    def _provision(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _teardown(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def start(self) -> None:
        if self._started:
            return
        self._provision()
        self._started = True

    def close(self) -> None:
        if self._closed or not self._started:
            self._closed = True
            return
        try:
            self._teardown()
        finally:
            self._closed = True

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

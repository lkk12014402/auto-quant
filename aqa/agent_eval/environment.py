"""Agentic environment protocol + a sandbox-backed implementation.

An :class:`Environment` is the world the model *acts in* during a tool-use /
terminal task: ``reset`` prepares it, ``step`` applies one action (a shell
command) and returns an :class:`Observation`, and an optional ``is_solved`` check
computes the terminal reward.  This is the substrate E4 (terminal-bench) builds
tasks on.

:class:`SandboxEnvironment` binds that protocol to an E2 :class:`~aqa.sandbox`
sandbox: actions run as shell commands inside the (isolated) sandbox, files/setup
commands seed the task, and a verify command (or callable) decides success.  All
blocking sandbox calls are dispatched with :func:`asyncio.to_thread`, so many
environments can run concurrently under the async eval loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ..sandbox.base import Sandbox


@dataclass
class Observation:
    """What the agent sees after an action (or after ``reset``)."""

    text: str = ""
    done: bool = False
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    async def reset(self) -> Observation:
        """Prepare the world and return the initial observation."""
        ...

    async def step(self, action: str) -> Observation:
        """Apply one action; return the resulting observation."""
        ...

    async def close(self) -> None:
        """Release resources (idempotent)."""
        ...


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n... [{omitted} chars truncated] ...\n{text[-tail:]}"


@dataclass
class SandboxEnvironment:
    """Environment whose actions are shell commands run in an E2 sandbox."""

    sandbox: Sandbox
    setup: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    verify_command: str | None = None
    verify: Callable[[Sandbox], bool] | None = None
    exec_timeout: float = 60.0
    max_output_chars: int = 4000

    async def reset(self) -> Observation:
        await asyncio.to_thread(self.sandbox.start)
        for path, content in self.files.items():
            await asyncio.to_thread(self.sandbox.write_file, path, content)
        for cmd in self.setup:
            await asyncio.to_thread(self.sandbox.exec, cmd, timeout=self.exec_timeout)
        return Observation(text="", info={"setup": True})

    async def step(self, action: str) -> Observation:
        res = await asyncio.to_thread(
            self.sandbox.exec, action, timeout=self.exec_timeout
        )
        return Observation(
            text=_truncate(res.output, self.max_output_chars),
            info={"returncode": res.returncode, "timed_out": res.timed_out},
        )

    async def is_solved(self) -> bool:
        """Run the task's verifier; True == the goal was reached."""
        if self.verify is not None:
            return bool(await asyncio.to_thread(self.verify, self.sandbox))
        if self.verify_command is not None:
            res = await asyncio.to_thread(
                self.sandbox.exec, self.verify_command, timeout=self.exec_timeout
            )
            return bool(res.ok)
        return False

    async def close(self) -> None:
        await asyncio.to_thread(self.sandbox.close)

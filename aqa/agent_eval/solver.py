"""Solvers: turn a task into a model response via the served endpoint.

Ports the role of NeMo Evaluator's ``solvers/`` (chat / completion) at the E0 base
level.  A solver owns *how* the model is prompted for a task -- the simplest are
:class:`ChatSolver` (single chat turn with an optional system prompt) and
:class:`CompletionSolver`.  Richer agentic solvers (tool-use / ReAct / gym) build
on the same :class:`Solver` protocol in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .client import ModelClient


@dataclass
class Task:
    """One agentic-eval task."""

    id: str
    prompt: str
    target: str = ""
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Solver(Protocol):
    async def solve(self, task: Task) -> str:
        """Produce the model's answer for ``task``."""
        ...


@dataclass
class ChatSolver:
    """Single-turn chat solver (optional per-solver system prompt)."""

    client: ModelClient
    system: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    async def solve(self, task: Task) -> str:
        messages: list[dict[str, str]] = []
        system = task.system or self.system
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task.prompt})
        return await self.client.chat(messages, **self.params)


@dataclass
class CompletionSolver:
    """Raw text-completion solver (no chat template)."""

    client: ModelClient
    params: dict[str, Any] = field(default_factory=dict)

    async def solve(self, task: Task) -> str:
        return await self.client.complete(task.prompt, **self.params)

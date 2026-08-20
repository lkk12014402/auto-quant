"""Serving proxy: a drop-in wrapper around a :class:`ModelClient` that adds a
turn counter and an optional response cache.

Mirrors the "serving proxy" NeMo Evaluator puts in front of an endpoint for
agentic runs.  Two concerns:

  * **turn_counter** -- every model call is a "turn"; agentic loops can run for
    many turns per task, so counting (and optionally *capping*) them bounds cost
    and gives observability (``stats``).
  * **caching** -- deterministic (temperature 0) evaluation issues identical
    requests across retries and repeated tasks; caching returns the previous
    completion instead of re-hitting the endpoint.

The proxy exposes the same async ``chat`` / ``complete`` surface as
:class:`ModelClient`, so any :class:`~aqa.agent_eval.solver.Solver` (single-turn
or ReAct) can use it interchangeably.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..logging_util import AqaError, get_logger

_log = get_logger("agent_eval.proxy")


@dataclass
class ProxyStats:
    """Observability counters for a :class:`ServingProxy`."""

    turns: int = 0          # model calls that actually hit the endpoint
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def calls(self) -> int:
        """Total calls seen (endpoint hits + cache hits)."""
        return self.turns + self.cache_hits

    def to_dict(self) -> dict[str, int]:
        return {
            "turns": self.turns,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class ServingProxy:
    """Wrap a chat/complete client with a turn counter + response cache."""

    client: Any                       # ModelClient-like (async chat/complete)
    cache: bool = True
    max_turns: int | None = None      # hard cap on endpoint hits (safety budget)
    stats: ProxyStats = field(default_factory=ProxyStats)
    _cache: dict[str, str] = field(default_factory=dict, repr=False)

    @staticmethod
    def _key(kind: str, payload: Any, params: dict[str, Any]) -> str:
        blob = json.dumps(
            [kind, payload, sorted(params.items())],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    async def _call(self, kind: str, payload: Any, params: dict[str, Any]) -> str:
        key = self._key(kind, payload, params)
        if self.cache and key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        if self.max_turns is not None and self.stats.turns >= self.max_turns:
            raise AqaError(
                f"serving proxy turn budget exhausted ({self.max_turns} turns)"
            )
        self.stats.turns += 1
        self.stats.cache_misses += 1
        fn = self.client.chat if kind == "chat" else self.client.complete
        result = await fn(payload, **params)
        if self.cache:
            self._cache[key] = result
        return result

    async def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        return await self._call("chat", messages, params)

    async def complete(self, prompt: str, **params: Any) -> str:
        return await self._call("complete", prompt, params)

    def reset_cache(self) -> None:
        self._cache.clear()

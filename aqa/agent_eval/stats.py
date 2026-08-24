"""Per-trajectory token + cost accounting (Harbor-parity).

Harbor records, for every trial, an ``AgentContext`` carrying
``n_input_tokens`` (incl. cache) / ``n_cache_tokens`` / ``n_output_tokens`` /
``cost_usd`` and aggregates them across steps and runs.  We mirror that field
vocabulary so a quantized model's *quality* (resolved-rate) can be read next to
its *efficiency* (tokens + turns) and compared across agent scaffolds -- the axis
that makes multi-agent-infra evaluation meaningful.

The numbers come from whatever the endpoint / CLI already reports: OpenAI-style
``usage`` blocks for our own ReAct loop (:meth:`TokenStats.add_usage`) and the
external agents' own JSON accounting for the CLI scaffolds.  No extra model calls;
missing fields simply stay zero.  Cost is only filled when a price is supplied
(:meth:`TokenStats.price`), so we never depend on a pricing table.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class TokenStats:
    """Cumulative token + cost totals for one trajectory (Harbor field names)."""

    n_input_tokens: int = 0       # prompt tokens, *including* cached
    n_cache_tokens: int = 0       # cached prompt tokens (subset of input)
    n_output_tokens: int = 0      # completion tokens
    n_requests: int = 0           # model calls that reported usage
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens

    def is_empty(self) -> bool:
        return (self.n_input_tokens == 0 and self.n_output_tokens == 0
                and self.n_cache_tokens == 0 and self.cost_usd is None)

    def add_usage(self, usage: Mapping[str, Any] | None) -> None:
        """Fold one OpenAI-style ``usage`` block into the running totals.

        Accepts both the chat (``prompt_tokens``/``completion_tokens``) and the
        raw (``input_tokens``/``output_tokens``) spellings, and reads cached
        tokens from ``prompt_tokens_details.cached_tokens`` or the flat
        ``cached_tokens`` / ``prompt_cache_hit_tokens`` variants.
        """
        if not usage:
            return
        prompt = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion = _as_int(
            usage.get("completion_tokens") or usage.get("output_tokens"))
        cached = 0
        details = usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            cached = _as_int(details.get("cached_tokens"))
        if not cached:
            cached = _as_int(usage.get("cached_tokens")
                             or usage.get("cached_input_tokens")
                             or usage.get("prompt_cache_hit_tokens"))
        self.n_input_tokens += prompt
        self.n_output_tokens += completion
        self.n_cache_tokens += cached
        if prompt or completion:
            self.n_requests += 1

    def merge(self, other: TokenStats | None) -> None:
        """Accumulate another :class:`TokenStats` into this one."""
        if other is None:
            return
        self.n_input_tokens += other.n_input_tokens
        self.n_cache_tokens += other.n_cache_tokens
        self.n_output_tokens += other.n_output_tokens
        self.n_requests += other.n_requests
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def price(self, *, input_per_1k: float, output_per_1k: float,
              cached_per_1k: float | None = None) -> None:
        """Fill :attr:`cost_usd` from simple per-1k-token rates (uncached input).

        ``cached_per_1k`` (when given) is charged for the cached slice and the
        remaining, uncached input at ``input_per_1k`` -- matching how providers
        discount cache hits.  Left unset, cost stays ``None``.
        """
        uncached = max(0, self.n_input_tokens - self.n_cache_tokens)
        cost = uncached / 1000 * input_per_1k
        cost += self.n_output_tokens / 1000 * output_per_1k
        if cached_per_1k is not None:
            cost += self.n_cache_tokens / 1000 * cached_per_1k
        self.cost_usd = round(cost, 6)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_input_tokens": self.n_input_tokens,
            "n_cache_tokens": self.n_cache_tokens,
            "n_output_tokens": self.n_output_tokens,
            "total_tokens": self.total_tokens,
            "n_requests": self.n_requests,
        }
        if self.cost_usd is not None:
            out["cost_usd"] = self.cost_usd
        return out


_DEFAULT_USAGE_KEYS = (
    "last_token_usage", "token_usage", "total_token_usage", "usage",
)


def _find_usage(obj: Any, keys: tuple[str, ...]) -> dict | None:
    """Depth-first search for the last dict stored under any of ``keys``."""
    found: dict | None = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, dict):
                found = v
            child = _find_usage(v, keys)
            if child is not None:
                found = child
    elif isinstance(obj, list):
        for item in obj:
            child = _find_usage(item, keys)
            if child is not None:
                found = child
    return found


def parse_usage_from_text(
    text: str, *, keys: tuple[str, ...] = _DEFAULT_USAGE_KEYS
) -> TokenStats | None:
    """Fold the *last* usage block found across JSON(L) text into a TokenStats.

    Shared by the CLI-agent scaffolds and the self-heal fix-loop: both drive
    external agents that emit their own token accounting as JSON events amid plain
    log lines.  Non-JSON lines are skipped; ``None`` is returned when nothing
    usage-shaped is present, so token accounting stays best-effort.
    """
    if not text:
        return None
    usage: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        hit = _find_usage(doc, keys)
        if hit is not None:
            usage = hit
    if usage is None:
        return None
    stats = TokenStats()
    stats.add_usage(usage)
    return stats if not stats.is_empty() else None

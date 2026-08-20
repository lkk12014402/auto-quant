"""Tier ladder: map tier names to an agent backend + model + env overrides.

Ports ``agent_tier_activate`` from ``local_dispatch/agent/agent_backends.sh``.  A
tier is a named rung on the escalation ladder (cheapest first).  Resolving a tier
yields the backend to invoke, the wire model, and any extra environment the
backend process needs (e.g. Copilot BYOK provider settings for MiniMax).

Secrets are never embedded here -- only the *names* of env vars the backend reads
(``MINIMAX_API_KEY``, ``COPILOT_GITHUB_TOKEN``) are referenced at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    backend: str                       # "openclaw" | "copilot"
    model: str | None = None           # wire model name (None => backend default)
    env: dict[str, str] = field(default_factory=dict)  # extra env for the backend
    byok: bool = False                 # Copilot BYOK (custom provider, no GitHub auth)

    def describe(self) -> str:
        return f"{self.name}(backend={self.backend}, model={self.model or 'default'})"


# MiniMax BYOK defaults (Copilot driving a non-GitHub model via its own provider).
_MINIMAX_BYOK_ENV = {
    "COPILOT_MINIMAX": "1",
    "COPILOT_PROVIDER_BASE_URL": "https://api.minimaxi.com/anthropic",
    "COPILOT_PROVIDER_TYPE": "anthropic",
    "COPILOT_PROVIDER_MODEL_ID": "claude-sonnet-4",
}


def resolve_tier(name: str) -> Tier:
    """Map a tier name to a concrete :class:`Tier` (unknown => copilot + that model)."""
    key = name.strip().lower()
    if key == "openclaw":
        return Tier(name="openclaw", backend="openclaw")
    if key == "minimax":
        # Copilot BYOK against MiniMax-M3, reusing MINIMAX_API_KEY (no GitHub auth).
        return Tier(name="minimax", backend="copilot", model="MiniMax-M3",
                    env=dict(_MINIMAX_BYOK_ENV), byok=True)
    if key == "opus":
        return Tier(name="opus", backend="copilot", model="claude-opus-4.8")
    if key == "sonnet":
        return Tier(name="sonnet", backend="copilot", model="claude-sonnet-4.5")
    if key == "copilot":
        return Tier(name="copilot", backend="copilot")
    # Unknown name => copilot backend with that string as the model override.
    return Tier(name=key, backend="copilot", model=name)


def resolve_tiers(names: list[str]) -> list[Tier]:
    """Resolve an ordered ladder of tier names; empty => the default two-tier ladder."""
    if not names:
        names = ["openclaw", "minimax"]
    return [resolve_tier(n) for n in names]

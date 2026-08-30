"""Backend adapter registry + auto-detection for session->ATIF conversion.

Adapters are keyed by backend name; ``detect_backend`` picks one from the
artifact's filename/content signature; ``convert`` runs the full pipeline
(adapter -> ATIF -> light validation via :func:`aqa.agent_eval.atif.validate_atif`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...logging_util import AqaError, get_logger
from .base import Adapter
from .claude import ADAPTER as CLAUDE
from .codex import ADAPTER as CODEX
from .copilot import ADAPTER as COPILOT
from .openclaw import ADAPTER as OPENCLAW

_log = get_logger("agent.adapters")

ADAPTERS: dict[str, Adapter] = {a.name: a for a in (OPENCLAW, COPILOT, CLAUDE, CODEX)}


def detect_backend(path: str | Path) -> str | None:
    """Best-effort backend detection from the artifact path (name heuristics)."""
    p = Path(path)
    for name, adapter in ADAPTERS.items():
        if adapter.detect(p):
            return name
    return None


def convert(path: str | Path, *, backend: str | None = None, model_name: str = "",
            instruction: str | None = None, validate: bool = True) -> dict[str, Any]:
    """Convert a raw session artifact to an ATIF dict.

    ``backend`` forces an adapter; otherwise ``detect_backend`` is used.
    Raises AqaError when no adapter matches or the conversion is unusable.
    """
    p = Path(path)
    if not p.exists():
        raise AqaError(f"session artifact not found: {p}")
    name = backend or detect_backend(p)
    if name is None or name not in ADAPTERS:
        raise AqaError(
            f"no adapter for {p} (detected: {name}); known: {sorted(ADAPTERS)}"
        )
    atif = ADAPTERS[name].to_atif(p, model_name=model_name, instruction=instruction)
    if atif is None:
        raise AqaError(f"{name} adapter produced no usable trajectory from {p}")
    if validate:
        from ...agent_eval.atif import validate_atif

        problems = validate_atif(atif)
        if problems:
            raise AqaError(f"{name} conversion of {p} failed ATIF checks: {problems}")
    return atif

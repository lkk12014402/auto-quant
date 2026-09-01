"""Secret redaction for trajectory artifacts.

Agent session logs routinely capture commands that print credentials
(``git remote -v`` with a token-embedded URL, ``docker run -e HF_TOKEN=...``,
CI env dumps).  Anything exported from those logs -- ATIF trajectories and
every downstream dataset view -- would carry the same secrets, and GitHub
push protection rejects it.  All adapters pass through :func:`redact_atif`
(wired into ``make_trajectory``), so redaction happens once at ingestion
instead of per artifact.

The pattern set is the canonical one from
:mod:`aqa.report.scrub` (also used by the results sink), so trajectory
artifacts and run artifacts share one scrubbing rule.
"""

from __future__ import annotations

from typing import Any

from ..report.scrub import sanitize_text


def redact_text(text: str) -> str:
    """Alias of :func:`aqa.report.scrub.sanitize_text` (canonical scrubber)."""
    return sanitize_text(text)


def _walk_strings(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [_walk_strings(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _walk_strings(v) for k, v in obj.items()}
    return obj


def redact_atif(atif: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets from every string in an ATIF trajectory, in place.

    Only step contents are walked (messages, reasoning, tool calls,
    observations) -- identifiers/metadata are left untouched.
    """
    for step in atif.get("steps") or []:
        for key in ("message", "reasoning_content", "tool_calls", "observation"):
            if key in step:
                step[key] = _walk_strings(step[key])
    return atif

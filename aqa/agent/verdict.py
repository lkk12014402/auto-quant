"""Parse structured fields and the verdict from an agent's fix-attempt output.

Ports the ``agent_verdict`` / ``extract_agent_field`` logic from
``local_dispatch/agent/agent_fix_loop.sh``.  The parsing is deliberately strict:
only a line whose ``VERDICT:`` value is exactly one of the tokens counts, so prose
like *"Only declare VERDICT: UNFIXABLE for ..."* or the placeholder
``VERDICT: <FIXABLE | ...>`` is ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    FIXABLE = "fixable"
    UNFIXABLE = "unfixable"
    ESCALATE = "escalate"


# A real verdict line: optional markdown/bullets, VERDICT:, EXACTLY one token, EOL.
_VERDICT_RE = re.compile(
    r"^[\s>*\-]*VERDICT\s*:?\s*\**\s*(FIXABLE|UNFIXABLE|ESCALATE)\s*\.?\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# The single-line labels the protocol asks the agent to emit.
_LABELS = (
    "COMPONENT",
    "ERROR_CLASS",
    "ROOT_CAUSE_HYPOTHESIS",
    "EVIDENCE_CMD",
    "EVIDENCE_RESULT",
    "VERDICT",
    "UNFIXABLE_REASON",
    "FIX_TIER",
    "FIX_PLAN",
    "SMOKE_TEST",
)


def parse_verdict(text: str) -> Verdict | None:
    """Return the LAST valid verdict in ``text`` (later attempts override), or None.

    Placeholder lines (``VERDICT: <FIXABLE | UNFIXABLE | ESCALATE>``) never match
    because the captured group would have to be a single bare token.
    """
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return None
    return Verdict(matches[-1].lower())


def extract_field(text: str, label: str, max_len: int = 200) -> str:
    """Pull a labeled single-line field's value (last occurrence wins)."""
    if not text:
        return ""
    pat = re.compile(
        rf"^[\s>*\-]*{re.escape(label)}\s*:?\s*\**\s*(.+?)\s*\**\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    values = [m.group(1).strip() for m in pat.finditer(text)]
    # Drop placeholder values like "<ONE stable snake_case token ...>".
    values = [v for v in values if not (v.startswith("<") and v.endswith(">"))]
    if not values:
        return ""
    return values[-1][:max_len]


def _normalize_class(raw: str) -> str:
    """Normalize an ERROR_CLASS token to a stable snake_case identifier."""
    tok = raw.strip().split()[0] if raw.strip() else ""
    tok = tok.lower()
    tok = re.sub(r"[^a-z0-9_]", "", tok)
    return tok


@dataclass
class AgentAnalysis:
    """Structured view of one agent fix attempt's output."""

    verdict: Verdict | None = None
    component: str = ""
    error_class: str = ""
    root_cause: str = ""
    unfixable_reason: str = ""
    fix_tier: str = ""
    fix_plan: str = ""
    smoke_test: str = ""
    raw: str = field(default="", repr=False)

    @classmethod
    def parse(cls, text: str) -> AgentAnalysis:
        return cls(
            verdict=parse_verdict(text),
            component=extract_field(text, "COMPONENT", 60),
            error_class=_normalize_class(extract_field(text, "ERROR_CLASS", 60)),
            root_cause=extract_field(text, "ROOT_CAUSE_HYPOTHESIS", 400),
            unfixable_reason=extract_field(text, "UNFIXABLE_REASON", 200),
            fix_tier=extract_field(text, "FIX_TIER", 40),
            fix_plan=extract_field(text, "FIX_PLAN", 400),
            smoke_test=extract_field(text, "SMOKE_TEST", 400),
            raw=text or "",
        )

    def to_dict(self) -> dict:
        d = {
            "verdict": self.verdict.value if self.verdict else None,
            "component": self.component,
            "error_class": self.error_class,
            "root_cause": self.root_cause,
            "unfixable_reason": self.unfixable_reason,
            "fix_tier": self.fix_tier,
            "fix_plan": self.fix_plan,
            "smoke_test": self.smoke_test,
        }
        return {k: v for k, v in d.items() if v}

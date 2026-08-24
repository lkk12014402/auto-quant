"""Persist and retrieve distilled agent-fix *trajectory summaries*.

A :class:`~aqa.agent.lessons.LessonStore` records the one-line ``signature -> solution``
outcome of a fix.  That captures *what* worked but throws away the agent's
*reasoning*: which hypotheses it tried, what it ruled out and why the final fix
was the right one.  This module keeps that reasoning as a compact, structured
summary keyed by the same error *signature*, so future runs (and different
models) can be primed not just with the conclusion but with a distilled
playbook of how it was reached -- and can skip the dead ends.

Summaries are distilled for free from the fields the agent already emits
(:class:`~aqa.agent.verdict.AgentAnalysis`) plus the loop's own record of the
hypotheses that were tried and rejected before the winning attempt.  No extra
LLM call is made.  Only *successful* fixes are recorded, so every stored
trajectory is a positive exemplar.  Signature dedup keeps the file small.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .drift import classify_error, exception_type
from .lessons import extract_signature
from .verdict import AgentAnalysis


@dataclass
class TrajectorySummary:
    """A distilled, structured record of one successful fix trajectory."""

    signature: str
    error_class: str = ""
    exc_type: str = ""
    outcome: str = "fixed"
    root_cause: str = ""
    fix_rationale: str = ""
    component: str = ""
    smoke_test: str = ""
    tried_and_rejected: list[str] = field(default_factory=list)
    tier: str = ""
    steps: int = 0
    model_id: str = "unknown"
    scheme: str = ""
    method: str = ""
    run_id: str = ""
    phase: str = ""
    ts: str = ""

    def to_dict(self) -> dict:
        d = {
            "ts": self.ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": self.phase,
            "signature": self.signature,
            "error_class": self.error_class,
            "exc_type": self.exc_type,
            "outcome": self.outcome,
            "root_cause": self.root_cause,
            "fix_rationale": self.fix_rationale,
            "component": self.component,
            "smoke_test": self.smoke_test,
            "tried_and_rejected": list(self.tried_and_rejected),
            "tier": self.tier,
            "steps": self.steps,
            "model_id": self.model_id,
            "scheme": self.scheme,
            "method": self.method,
            "run_id": self.run_id,
        }
        return {k: v for k, v in d.items() if v not in ("", [], 0, None)}


def build_summary(
    phase: str,
    error_text: str,
    analysis: AgentAnalysis | None,
    *,
    tried_and_rejected: list[str] | None = None,
    tier: str = "",
    steps: int = 0,
    model_id: str = "unknown",
    scheme: str = "",
    method: str = "",
    run_id: str = "",
) -> TrajectorySummary:
    """Distil a :class:`TrajectorySummary` from the winning attempt's analysis.

    Reuses the already-parsed :class:`AgentAnalysis` fields (no extra LLM call).
    ``tried_and_rejected`` is the loop's compact record of earlier hypotheses.
    """
    err_class = (analysis.error_class if analysis and analysis.error_class
                 else classify_error(error_text))
    return TrajectorySummary(
        signature=extract_signature(error_text),
        error_class=err_class,
        exc_type=exception_type(error_text),
        root_cause=(analysis.root_cause if analysis else "")[:400],
        fix_rationale=(analysis.fix_plan if analysis else "")[:400],
        component=(analysis.component if analysis else "")[:80],
        smoke_test=(analysis.smoke_test if analysis else "")[:200],
        tried_and_rejected=[t for t in (tried_and_rejected or []) if t][:6],
        tier=tier,
        steps=steps,
        model_id=model_id,
        scheme=scheme,
        method=method,
        run_id=run_id,
        phase=phase,
    )


class TrajectoryStore:
    """Append-only per-phase JSONL trajectory summaries (dedup by signature)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, phase: str) -> Path:
        return self.root / f"{phase}.jsonl"

    def load(self, phase: str) -> list[dict]:
        path = self._path(phase)
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def save(self, phase: str, summary: TrajectorySummary) -> dict:
        """Append a summary; update in place if the signature already exists."""
        self.root.mkdir(parents=True, exist_ok=True)
        summary.phase = summary.phase or phase
        row = summary.to_dict()
        signature = row.get("signature", "")
        existing = self.load(phase)
        replaced = False
        for i, prev in enumerate(existing):
            if prev.get("signature") == signature:
                existing[i] = row
                replaced = True
                break
        if not replaced:
            existing.append(row)
        self._path(phase).write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in existing) + "\n"
        )
        return row

    def format_for_prompt(self, phase: str, *, error_class: str = "",
                          limit: int = 2) -> str:
        """Render the most relevant trajectory summaries as a reasoning playbook.

        Ranks summaries with a matching ``error_class`` first, then most recent,
        so the agent is primed with how a *similar* failure was reasoned through
        (and which dead ends to skip) rather than a generic dump.
        """
        rows = self.load(phase)
        if not rows:
            return ""
        want = error_class or ""

        def rank(r: dict) -> tuple:
            match = 0 if (want and r.get("error_class") == want) else 1
            return (match, )

        rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
        rows.sort(key=rank)
        picked = rows[:limit]
        if not picked:
            return ""
        blocks: list[str] = []
        for r in picked:
            head = f"- ({r.get('error_class', '?')}) {r.get('signature', '')}"
            lines = [head]
            if r.get("root_cause"):
                lines.append(f"  root cause: {r['root_cause']}")
            if r.get("fix_rationale"):
                lines.append(f"  fix that worked: {r['fix_rationale']}")
            rejected = r.get("tried_and_rejected") or []
            if rejected:
                lines.append("  dead ends (do NOT repeat): " + "; ".join(rejected))
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

"""Persist and retrieve fix-loop lessons (per-phase JSONL).

Ports ``save_lesson`` / ``format_lessons`` from
``local_dispatch/agent/agent_fix_loop.sh``.  A lesson records the stable error
*signature* (deepest exception line), its category, and the resolution so future
runs can be primed with what worked before.  Signature dedup keeps the file small.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .drift import classify_error
from .verdict import AgentAnalysis

# Strip leading timestamp / log-level so signatures are stable across runs.
_PREFIX_RE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[T ])?(?:\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?"
    r"(?:\s*[Zz]|\s*[+-]\d{2}:?\d{2})?\s*(?:\[[A-Za-z]+\]|[A-Z]{3,}:)?\s*"
)
_EXC_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception|Warning)|OSError)\b\s*:\s*\S")
_WRAP_RE = re.compile(r"\b(?:failed|error)\b\s*[:\-]\s*(\S.+)", re.IGNORECASE)


def _clean(line: str) -> str:
    return _PREFIX_RE.sub("", line).strip()


def extract_signature(text: str) -> str:
    """The stable one-line signature of an error (deepest exception, else last line)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "unknown error"
    exc = [_clean(ln) for ln in lines if _EXC_RE.search(_clean(ln))]
    if exc:
        return exc[-1][:150]
    for ln in reversed(lines):
        c = _clean(ln)
        m = _WRAP_RE.search(c)
        if m and m.group(1).strip():
            return c[:150]
    return _clean(lines[-1])[:150]


class LessonStore:
    """Append-only per-phase JSONL lessons under ``root`` (dedup by signature)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, phase: str) -> Path:
        return self.root / f"{phase}.jsonl"

    def load(self, phase: str) -> list[dict]:
        path = self._path(phase)
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def save(
        self,
        phase: str,
        error_context: str,
        status: str,
        solution_note: str,
        *,
        model_id: str = "unknown",
        scheme: str = "",
        method: str = "",
        analysis: AgentAnalysis | None = None,
    ) -> dict:
        """Append a lesson; update in place if the signature already exists."""
        self.root.mkdir(parents=True, exist_ok=True)
        signature = extract_signature(error_context)
        error_class = (analysis.error_class if analysis and analysis.error_class
                       else classify_error(error_context))
        lesson = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "status": status,
            "signature": signature,
            "error_class": error_class,
            "model_id": model_id,
            "scheme": scheme,
            "method": method,
            "solution": solution_note,
        }
        if analysis:
            lesson["analysis"] = analysis.to_dict()

        existing = self.load(phase)
        replaced = False
        for i, prev in enumerate(existing):
            if prev.get("signature") == signature:
                existing[i] = lesson
                replaced = True
                break
        if not replaced:
            existing.append(lesson)
        self._path(phase).write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in existing) + "\n"
        )
        return lesson

    def format_for_prompt(self, phase: str, limit: int = 8) -> str:
        """Render recent lessons as compact bullet text for the fix prompt."""
        lessons = self.load(phase)[-limit:]
        if not lessons:
            return ""
        lines = []
        for lz in lessons:
            status = lz.get("status", "?")
            sig = lz.get("signature", "")
            sol = lz.get("solution", "")
            cls = lz.get("error_class", "")
            lines.append(f"- [{status}] ({cls}) {sig} -> {sol}")
        return "\n".join(lines)

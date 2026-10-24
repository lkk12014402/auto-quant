"""Append-only session event stream (``session/events.jsonl``).

Replaces "what happened" being scattered across executor logs / transcripts with
one ordered, machine-readable log of phase transitions and decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._io import now_iso

_EVENTS_FILE = "events.jsonl"


def events_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / _EVENTS_FILE


def append_event(session_dir: str | Path, type: str, **fields: Any) -> None:
    path = events_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": now_iso(), "type": type, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_events(session_dir: str | Path) -> list[dict[str, Any]]:
    path = events_path(session_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

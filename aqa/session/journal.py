"""External-agent run journal: plan + reasoning + revisions as a durable artifact.

The pipeline already writes structured per-phase artifacts (``analysis.json``,
``quant_plan.json``, ``eval_report.json``) and an internal ``state.json``
``decisions[]`` log. What it could *not* capture is the **narrative** an external
agent (Claude / Codex / Copilot / Kimi driving aqa, like Hyperloom drives Claude
Code) produces while planning a run — above all the *plan-then-revise* history as
pits are hit. That reasoning used to live only in the chat transcript and never
reached ``results/``.

This module captures it into the run itself, co-located with ``recipe.yaml``::

    <run_dir>/journal.md            # human-readable, append-only, newest last
    <run_dir>/session/events.jsonl  # the same entries as ``journal`` events

**Append-only by design.** A correction is a *new* entry (``kind=revise`` with
``supersedes=<id>``), never an in-place edit, so the "I planned X, then changed to
Y because Z" trail is preserved — which is exactly the plan-then-revise history we
want to keep. The markdown is rebuilt deterministically from the event stream, so
``journal.md`` and ``events.jsonl`` can never drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._io import atomic_write_json, now_iso
from .events import append_event, read_events

JOURNAL_FILE = "journal.md"
_SESSION_DIR = "session"
_EVENT_TYPE = "journal"

# Narrative kinds. ``plan`` = an intended course of action; ``revise`` = a change
# of plan (carries ``supersedes``); ``decision`` = a committed choice (optionally
# mirrored into state.json decisions[]); ``note`` / ``observation`` = free prose.
KINDS = ("plan", "revise", "decision", "note", "observation")


def _session_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / _SESSION_DIR


def journal_entries(run_dir: str | Path) -> list[dict[str, Any]]:
    """All journal entries for a run, oldest first."""
    return [e for e in read_events(_session_dir(run_dir)) if e.get("type") == _EVENT_TYPE]


def _next_id(run_dir: str | Path) -> str:
    return f"j{len(journal_entries(run_dir)) + 1:03d}"


def append_journal(
    run_dir: str | Path,
    *,
    title: str,
    body: str = "",
    kind: str = "note",
    phase: str = "",
    author: str = "external-agent",
    ref: str = "",
    supersedes: str = "",
) -> dict[str, Any]:
    """Append one journal entry to a run and rebuild ``journal.md``.

    Bootstraps ``<run_dir>/session/`` if it does not exist yet (so the plan can be
    recorded *before* ``aqa run`` opens the session — the pipeline appends to the
    same stream and never clobbers it). Returns the stored entry.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not title.strip():
        raise ValueError("journal entry needs a non-empty title")
    run_dir = Path(run_dir)
    entry: dict[str, Any] = {
        "id": _next_id(run_dir),
        "kind": kind,
        "phase": phase,
        "title": title.strip(),
        "body": body.rstrip(),
        "author": author,
        "ref": ref,
        "supersedes": supersedes,
    }
    # The event carries ``ts`` (added by append_event); mirror it onto the entry so
    # a rebuilt markdown keeps timestamps without re-reading the raw event.
    ts = now_iso()
    append_event(_session_dir(run_dir), _EVENT_TYPE, ts=ts, **entry)
    entry["ts"] = ts
    _write_markdown(run_dir)
    return entry


def _render_entry(e: dict[str, Any]) -> str:
    phase = (e.get("phase") or "").strip()
    head = f"## {e.get('id', '?')} · {e.get('kind', 'note')}"
    if phase:
        head += f" · {phase}"
    head += f" — {e.get('title', '').strip()}"
    meta = [f"_{e.get('ts', '')}_", f"author: {e.get('author', '')}"]
    if e.get("supersedes"):
        meta.append(f"supersedes: {e['supersedes']}")
    if e.get("ref"):
        meta.append(f"ref: {e['ref']}")
    lines = [head, "", "  ·  ".join(m for m in meta if m.strip())]
    body = (e.get("body") or "").rstrip()
    if body:
        lines += ["", body]
    lines.append("")
    return "\n".join(lines)


def render_markdown(run_dir: str | Path) -> str:
    """Deterministically rebuild the full journal document from the event stream."""
    entries = journal_entries(run_dir)
    title = Path(run_dir).name or "run"
    out = [f"# Run journal — {title}", "",
           "> Append-only. Plan + reasoning + revisions captured by the driving agent.",
           "> A change of plan is a new `revise` entry (see `supersedes`), never an edit.",
           ""]
    if not entries:
        out.append("_(no entries yet)_")
    for e in entries:
        out.append(_render_entry(e))
    return "\n".join(out).rstrip() + "\n"


def _write_markdown(run_dir: str | Path) -> None:
    path = Path(run_dir) / JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_markdown(run_dir), encoding="utf-8")
    tmp.replace(path)


def mirror_decision(run_dir: str | Path, *, phase: str, what: str, why: str,
                    ref: str = "") -> bool:
    """Append a decision into ``session/state.json`` decisions[] (best-effort).

    Keeps the crisp what/why visible to ``aqa session report`` and to the internal
    agent's ``prompt_seed``. Returns True when the state file existed and was
    updated; False when there is no session state yet (journal event still stands).
    """
    state_path = _session_dir(run_dir) / "state.json"
    if not state_path.is_file():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    data.setdefault("decisions", []).append(
        {"phase": phase, "what": what, "why": why, "ref": ref}
    )
    data["updated_at"] = now_iso()
    atomic_write_json(state_path, data)
    return True

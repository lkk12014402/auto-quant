"""Session layer: the quantize run as a resumable, reproducible first-class object.

Borrowed from AMD-AGI/Hyperloom's session model (``manifest.json`` + ``state.json``
+ an append-only event stream), this package turns what used to live only in a
transient ``aqa_run.json`` ``{model, state}`` blob into a durable ``session/``
subtree the pipeline can ``load`` / ``commit`` / ``resume``:

    <run_dir>/session/
        manifest.json   # reproducible intent: schema, run_id, resolved run.yaml, env lock
        state.json      # the state machine: phase + phases{} + baseline/current_best + decisions
        events.jsonl    # append-only "what happened" stream

See docs/rfc/0001-reproducible-session-state-machine.md (P1).
"""

from __future__ import annotations

from .events import append_event, read_events
from .journal import append_journal, journal_entries, mirror_decision, render_markdown
from .manifest import (
    SCHEMA_VERSION,
    build_session_manifest,
    load_session_manifest,
    write_session_manifest,
)
from .session import Session, open_session
from .state import Phase, SessionState, Status

__all__ = [
    "Phase",
    "Status",
    "SessionState",
    "Session",
    "open_session",
    "SCHEMA_VERSION",
    "build_session_manifest",
    "write_session_manifest",
    "load_session_manifest",
    "append_event",
    "read_events",
    "append_journal",
    "journal_entries",
    "render_markdown",
    "mirror_decision",
]

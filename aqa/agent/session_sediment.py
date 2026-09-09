"""End-of-run session sedimentation: ATIF-ize this run's agent sessions.

Collects the structured session artifacts a run produced — fix-loop per-attempt
``*.events.jsonl`` under ``logs/agent_fixes/``, optimize sessions under
``logs/agent_optimize/``, plus anything the owner dropped into ``sessions/``
(raw backend logs: kimi wire.jsonl / copilot events / claude / codex ...) —
converts each to ATIF via the backend adapters (redaction happens at the
adapter choke point, so what lands in the store is already sanitized), stages
them under the shared session store, and the knowledge store pushes them to
the dataset's ``trajectories/`` tree (parallel to ``knowledge/``).

Runs at pipeline end (success and failure paths alike — failed runs have the
most instructive sessions).  Always best-effort: sedimentation never fails a run.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..logging_util import get_logger
from .adapters import convert

_log = get_logger("agent.session_sediment")

_EVENTS = "**/*.events.jsonl"


def default_session_store() -> Path:
    """Shared host dir for staged session trajectories ($AQA_SESSION_DIR)."""
    env = os.environ.get("AQA_SESSION_DIR")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path(os.environ.get("HOME", "/root")) / ".cache")
    return Path(cache) / "aqa" / "session_trajectories"


def collect_sessions(out_dir: Path) -> list[tuple[Path, str | None]]:
    """(session artifact path, forced backend or None=auto-detect) pairs."""
    out_dir = Path(out_dir)
    found: list[tuple[Path, str | None]] = []
    for sub, backend in (("logs/agent_fixes", "copilot"),
                         ("logs/agent_optimize", "copilot")):
        base = out_dir / sub
        if base.is_dir():
            found += [(p, backend) for p in sorted(base.glob(_EVENTS))]
    owner = out_dir / "sessions"
    if owner.is_dir():
        found += [(p, None) for p in sorted(owner.iterdir()) if p.is_file()]
    return found


def sediment_sessions(out_dir: Path, run_id: str, *,
                      stage_root: Path | None = None) -> list[Path]:
    """Convert this run's sessions to ATIF under ``<stage>/<run_id>/``.

    Returns the staged file list (empty when nothing convertible was found).
    """
    stage = (stage_root or default_session_store()) / (run_id or "norun")
    staged: list[Path] = []
    for path, backend in collect_sessions(out_dir):
        try:
            atif = convert(path, backend=backend)
        except Exception as exc:  # noqa: BLE001 - one bad session must not block
            _log.warning("session sediment: skipping %s (%s)", path, exc)
            continue
        sid = atif.get("session_id") or path.stem
        dest = stage / f"{sid}.jsonl"
        stage.mkdir(parents=True, exist_ok=True)
        import json

        dest.write_text(json.dumps(atif, ensure_ascii=False) + "\n")
        staged.append(dest)
    if staged:
        _log.info("session sediment: staged %d trajectory(ies) -> %s",
                  len(staged), stage)
    return staged

"""Shared helpers for the session layer (timestamps + atomic writes)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """UTC timestamp, second precision, e.g. ``2026-10-13T09:41:07Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON to ``path`` atomically.

    Mirrors Hyperloom's ``os.replace`` manifest write: a crash mid-write can
    never leave a half-written ``state.json`` that would poison a later resume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)

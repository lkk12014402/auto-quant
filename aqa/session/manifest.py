"""The reproducibility manifest (``session/manifest.json``).

Captures the *immutable intent* of a run — schema version, run id, the fully
resolved ``run.yaml`` and the frozen software/hardware stack — written once at
session start (not only on a successful finish, as ``run_manifest.json`` is).
Mirrors Hyperloom's ``inference_optimizer/session/manifest.py``:
``build`` / ``write`` (atomic) / ``load`` (missing ⇒ refuse to resume).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_util import AqaError
from ._io import atomic_write_json, now_iso
from .state import SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

_MANIFEST_FILE = "manifest.json"


def manifest_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / _MANIFEST_FILE


def build_session_manifest(run_cfg: "RunConfig", run_id: str) -> dict[str, Any]:
    # Reuse the stack/accelerator probes the terminal manifest already uses so
    # the session manifest's env lock is the same source of truth.
    from ..report.manifest import _accelerator, _package_versions

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": now_iso(),
        "model": run_cfg.model,
        "run_config": run_cfg.model_dump(mode="json"),
        "env_lock": _package_versions(),
        "accelerator": _accelerator(run_cfg),
    }


def write_session_manifest(session_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(session_dir)
    atomic_write_json(path, manifest)
    return path


def load_session_manifest(session_dir: str | Path) -> dict[str, Any]:
    """Load the session manifest, or refuse (``AqaError``) when it is absent.

    This is the resume precondition: like Hyperloom's CLI refusing to start when
    ``manifest.json`` is missing, aqa would rather fail loudly than fake a resume.
    """
    path = manifest_path(session_dir)
    if not path.exists():
        raise AqaError(f"no session manifest to resume at {path}")
    return json.loads(path.read_text())

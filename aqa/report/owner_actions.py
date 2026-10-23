"""Append and load the owner action log for one run."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

OWNER_ACTION_KINDS = ("env", "command", "patch", "note", "decision")
OWNER_ACTIONS_FILE = "owner_actions.jsonl"
MANIFEST_FILE = "run_manifest.json"


def load_owner_actions(run_dir: Path) -> list[dict[str, Any]] | None:
    """Load validated JSON objects from a run's append-only action log."""
    path = run_dir / OWNER_ACTIONS_FILE
    if not path.is_file():
        return None

    actions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        action = json.loads(line)
        if not isinstance(action, dict):
            raise ValueError(f"{path}:{line_number}: owner action must be a JSON object")
        actions.append(action)
    return actions or None


def append_owner_action(
    run_dir: Path,
    *,
    kind: str,
    text: str,
    command: str | None = None,
) -> dict[str, Any]:
    """Append one timestamped action and refresh a manifest if it already exists."""
    if kind not in OWNER_ACTION_KINDS:
        raise ValueError(f"unsupported owner action kind: {kind}")
    if not text.strip():
        raise ValueError("owner action text must not be empty")

    run_dir = Path(run_dir)
    path = run_dir / OWNER_ACTIONS_FILE
    action: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "text": text,
    }
    if command:
        action["command"] = command

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(action, ensure_ascii=False) + "\n")
    sync_owner_actions_to_manifest(run_dir)
    return action


def sync_owner_actions_to_manifest(run_dir: Path) -> bool:
    """Atomically refresh manifest owner_actions after a post-run action."""
    manifest_path = Path(run_dir) / MANIFEST_FILE
    if not manifest_path.is_file():
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: run manifest must be a JSON object")
    manifest["owner_actions"] = load_owner_actions(Path(run_dir))

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=".run_manifest.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return True

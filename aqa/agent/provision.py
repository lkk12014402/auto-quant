"""Provision the agent-fix backends' config into the runtime home directory.

Ports the ``~/.openclaw`` / ``~/.copilot`` setup that
``local_dispatch/remote_exec.py`` does inside the container so the fix-loop's
OpenClaw and Copilot backends are authenticated and skill-aware by default.

What it lays down (idempotent):
  * ``~/.openclaw/openclaw.json`` + ``~/.openclaw/agents/main/agent/auth-profiles.json``
    from the bundled templates, with ``MINIMAX_API_KEY`` synced into the MiniMax
    auth profiles (openclaw's tier-0 provider).
  * ``~/.copilot/settings.json`` from the bundled template (model/effort defaults).
  * The project skills copied into ``~/.copilot/skills`` and
    ``~/.openclaw/workspace/skills`` so both backends share the same recipe
    knowledge (single source of truth).

Secrets are read from the environment only (``--minimax-env`` names the var);
nothing is inlined. Run as ``python -m aqa.agent.provision`` inside the container.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ..logging_util import get_logger

_log = get_logger("agent.provision")

_ASSETS = Path(__file__).resolve().parent / "assets"
_MINIMAX_PROFILES = ("minimax-global", "minimax:cn", "minimax:global")


def _copy_tree(src: Path, dst: Path) -> None:
    """Overlay ``src`` contents onto ``dst`` (like ``cp -a src/. dst/``)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def sync_minimax_key(auth_path: Path, key: str) -> bool:
    """Write ``key`` into the MiniMax profiles of an openclaw auth-profiles file."""
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _log.warning("cannot sync MiniMax key into %s: %s", auth_path, exc)
        return False
    profiles = data.setdefault("profiles", {})
    updated = False
    for name in _MINIMAX_PROFILES:
        prof = profiles.get(name)
        if isinstance(prof, dict) and prof.get("provider") == "minimax":
            prof["key"] = key
            updated = True
    if updated:
        auth_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return updated


def provision_openclaw(home: Path, minimax_key: str | None) -> None:
    src = _ASSETS / "openclaw"
    if not src.is_dir():
        _log.warning("bundled openclaw assets missing at %s", src)
        return
    dst = home / ".openclaw"
    _copy_tree(src, dst)
    if minimax_key:
        auth = dst / "agents" / "main" / "agent" / "auth-profiles.json"
        if sync_minimax_key(auth, minimax_key):
            _log.info("synced MINIMAX_API_KEY into openclaw auth profiles")
    else:
        _log.info("MINIMAX_API_KEY not set; openclaw MiniMax tiers stay unauthenticated")


def provision_copilot(home: Path, model: str | None) -> None:
    src = _ASSETS / "copilot" / "settings.json"
    if not src.is_file():
        _log.warning("bundled copilot settings missing at %s", src)
        return
    dst_dir = home / ".copilot"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "settings.json"
    if not dst.exists():
        settings = json.loads(src.read_text(encoding="utf-8"))
        if model:
            settings["model"] = model
        dst.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")


def provision_skills(home: Path, skills_dir: Path | None) -> None:
    """Share the project skills with both agent backends (single source of truth)."""
    if not skills_dir or not skills_dir.is_dir():
        return
    for dest in (home / ".copilot" / "skills", home / ".openclaw" / "workspace" / "skills"):
        _copy_tree(skills_dir, dest)


def provision(
    *,
    home: Path | None = None,
    minimax_key: str | None = None,
    copilot_model: str | None = None,
    skills_dir: Path | None = None,
) -> None:
    home = home or Path(os.environ.get("HOME", "/root"))
    provision_openclaw(home, minimax_key)
    provision_copilot(home, copilot_model)
    provision_skills(home, skills_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision agent-fix backend config")
    parser.add_argument("--home", default=None, help="target home (default: $HOME)")
    parser.add_argument("--minimax-env", default="MINIMAX_API_KEY",
                        help="env var holding the MiniMax API key")
    parser.add_argument("--copilot-model", default=None,
                        help="override the copilot settings.json model")
    parser.add_argument("--skills-dir", default=None,
                        help="project skills dir to share with both backends")
    args = parser.parse_args(argv)
    provision(
        home=Path(args.home) if args.home else None,
        minimax_key=os.environ.get(args.minimax_env) or None,
        copilot_model=args.copilot_model,
        skills_dir=Path(args.skills_dir) if args.skills_dir else None,
    )
    _log.info("agent backend config provisioned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

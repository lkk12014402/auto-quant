"""Optional git sync for auto-pinned Recipe KB entries.

The recipe file is already written by ``RecipeKB.pin``.  This module turns that
file into a reviewable commit without touching the caller's current worktree:

* ``branch``: create a detached temporary worktree from ``HEAD``, copy the auto
  recipe, commit it on ``aqa/kb-<run_id>``, then remove the worktree.
* ``pr``: additionally push that branch to ``origin`` and open a PR with ``gh``.

Default is ``off``; CI/owners can set ``agent.recipe_kb.auto_sync`` explicitly.
All failures are returned as data, never raised into CLOSE.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(["git", "-C", str(repo), *args])


def _repo_root(curated_root: Path) -> Path | None:
    proc = _git(curated_root, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.strip())


def _default_base(repo: Path) -> str:
    """Prefer origin/HEAD; fall back to main for repos without a remote HEAD."""
    ref = _git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if ref.returncode == 0:
        value = ref.stdout.strip()
        if "/" in value:
            return value.split("/", 1)[1]
    return "main"


def _branch_name(run_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id or "run").strip("-")
    return f"aqa/kb-{slug}"[:120]


def sync_auto_recipe(*, recipe_path: str | Path, curated_root: str | Path,
                     run_id: str, mode: str, base: str | None = None) -> dict[str, Any]:
    """Commit an auto recipe on a reviewable branch; return a status dict."""
    if mode not in {"off", "branch", "pr"}:
        return {"status": "failed", "error": f"unknown auto_sync mode {mode!r}"}
    if mode == "off":
        return {"status": "off"}
    recipe = Path(recipe_path).resolve()
    curated = Path(curated_root).resolve()
    if not recipe.is_file():
        return {"status": "failed", "error": f"recipe not found: {recipe}"}
    repo = _repo_root(curated)
    if repo is None:
        return {"status": "skipped", "error": "curated root is not a git worktree"}
    try:
        rel = recipe.relative_to(repo)
    except ValueError:
        return {"status": "skipped", "error": "auto recipe is outside the git repo"}
    branch = _branch_name(run_id)

    with tempfile.TemporaryDirectory(prefix="aqa-kb-") as tmp:
        work = Path(tmp) / "repo"
        add = _git(repo, "worktree", "add", "--detach", str(work), "HEAD")
        if add.returncode != 0:
            return {"status": "failed", "error": add.stderr.strip()[:500]}
        try:
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recipe, target)
            add_file = _git(work, "add", str(rel))
            if add_file.returncode != 0:
                return {"status": "failed", "error": add_file.stderr.strip()[:500]}
            changed = _git(work, "diff", "--cached", "--quiet")
            if changed.returncode == 0:
                return {"status": "skipped", "error": "no recipe change"}
            commit = _git(
                work, "-c", "user.name=aqa", "-c", "user.email=aqa@localhost",
                "commit", "-m", f"chore(recipe-kb): pin auto recipe ({run_id})",
            )
            if commit.returncode != 0:
                return {"status": "failed", "error": commit.stderr.strip()[:500]}
            sha = _git(work, "rev-parse", "HEAD").stdout.strip()
            switch = _git(work, "switch", "-c", branch)
            if switch.returncode != 0:
                return {"status": "failed", "error": switch.stderr.strip()[:500]}
            result: dict[str, Any] = {
                "status": "branch", "branch": branch, "commit": sha,
                "recipe": str(rel),
            }
            if mode == "branch":
                return result

            push = _git(work, "push", "origin", f"HEAD:refs/heads/{branch}")
            if push.returncode != 0:
                result.update(status="failed", error=push.stderr.strip()[:500])
                return result
            if shutil.which("gh") is None:
                result.update(status="branch_pushed",
                              error="gh not installed; branch pushed without PR")
                return result
            base = base or _default_base(repo)
            pr = _run([
                "gh", "pr", "create", "--head", branch, "--base", base,
                "--title", f"Recipe KB: pin auto recipe ({run_id})",
                "--body", f"Auto-pinned by aqa run `{run_id}`.\n\n"
                          f"File: `{rel}`",
            ], cwd=repo)
            if pr.returncode != 0:
                result.update(status="branch_pushed",
                              error=pr.stderr.strip()[:500])
            else:
                result.update(status="pr", pr_url=pr.stdout.strip())
            return result
        finally:
            _git(repo, "worktree", "remove", "--force", str(work))

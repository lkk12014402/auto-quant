"""Recipe KB git auto-sync (detached worktree; no current-branch changes)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from aqa.agent.recipe_git import sync_auto_recipe


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "recipes" / "quant" / "auto").mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
         "add", ".")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
         "commit", "-m", "base")
    return repo


def test_branch_mode_commits_in_detached_worktree(tmp_path: Path):
    repo = _repo(tmp_path)
    recipe = repo / "recipes" / "quant" / "auto" / "m.json"
    recipe.write_text('{"model": "org/m"}\n')

    result = sync_auto_recipe(
        recipe_path=recipe, curated_root=repo / "recipes" / "quant",
        run_id="run-1", mode="branch")

    assert result["status"] == "branch", result
    assert result["branch"] == "aqa/kb-run-1"
    assert result["commit"]
    branches = _git(repo, "branch", "--list", "aqa/kb-run-1").stdout
    assert "aqa/kb-run-1" in branches
    assert _git(repo, "show", f"{result['branch']}:recipes/quant/auto/m.json").returncode == 0
    # The main worktree keeps the auto file as an untracked pending copy; the
    # reviewable commit lives on the branch.
    status = _git(repo, "status", "--porcelain", "-uall").stdout
    assert "recipes/quant/auto/m.json" in status
    assert not _git(repo, "worktree", "list", "--porcelain").stdout.count("worktree") > 1


def test_off_and_non_repo_are_safe(tmp_path: Path):
    repo = _repo(tmp_path)
    recipe = repo / "recipes" / "quant" / "auto" / "m.json"
    recipe.write_text("{}\n")
    assert sync_auto_recipe(
        recipe_path=recipe, curated_root=repo / "recipes" / "quant",
        run_id="run-1", mode="off")["status"] == "off"

    outside = tmp_path / "outside"
    outside.mkdir()
    assert sync_auto_recipe(
        recipe_path=recipe, curated_root=outside,
        run_id="run-1", mode="branch")["status"] == "skipped"

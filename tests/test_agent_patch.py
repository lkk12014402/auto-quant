"""Tests for patch capture (snapshot -> diff of agent edits)."""

from __future__ import annotations

from pathlib import Path

from aqa.agent.patch import PatchArea, PatchCapture


def _area(tmp_path: Path) -> tuple[PatchArea, Path]:
    src = tmp_path / "pkg"
    (src / "sub").mkdir(parents=True)
    (src / "mod.py").write_text("def f():\n    return 1\n")
    (src / "sub" / "other.py").write_text("x = 1\n")
    (src / "mod.pyc").write_text("binary-noise")  # excluded
    return PatchArea("pkg", "auto_round", src), src


def test_snapshot_and_no_change_diff(tmp_path: Path):
    area, _ = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    assert cap.snapshot() == 1
    assert cap.combined_diff() == ""  # nothing edited yet


def test_diff_captures_edit(tmp_path: Path):
    area, src = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    cap.snapshot()
    (src / "mod.py").write_text("def f():\n    return 2\n")  # agent edit
    diff = cap.combined_diff()
    assert "a/pkg/mod.py" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_diff_captures_new_file(tmp_path: Path):
    area, src = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    cap.snapshot()
    (src / "new.py").write_text("y = 2\n")
    diff = cap.combined_diff()
    assert "b/pkg/new.py" in diff
    assert "+y = 2" in diff


def test_write_patches(tmp_path: Path):
    area, src = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    cap.snapshot()
    (src / "mod.py").write_text("def f():\n    return 3\n")
    written = cap.write_patches(tmp_path / "patches")
    assert "pkg" in written and written["pkg"].exists()
    assert "combined" in written
    assert "return 3" in written["combined"].read_text()


def test_snapshot_idempotent(tmp_path: Path):
    area, _ = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    assert cap.snapshot() == 1
    assert cap.snapshot() == 0  # already snapshotted, not re-copied


def test_pyc_excluded_from_diff(tmp_path: Path):
    area, src = _area(tmp_path)
    cap = PatchCapture(tmp_path / "snap", areas=[area])
    cap.snapshot()
    (src / "mod.pyc").write_text("changed-binary")  # should be ignored
    assert cap.combined_diff() == ""

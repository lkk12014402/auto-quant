"""Capture agent source edits as unified-diff patches.

Ports ``local_dispatch/agent/patch_capture.sh``.  Before the fix-loop makes any
edit we snapshot the *editable areas* (the installed ``auto_round`` / ``vllm``
package source + the HF model custom-code cache).  After a fix we diff the live
tree against that pristine snapshot to record exactly what the agent changed --
uploaded with the results so a fix is reproducible / reviewable.

The snapshot + diff are pure filesystem operations (``difflib`` based), so this
module is fully testable without any agent or GPU.
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..logging_util import get_logger

_log = get_logger("agent.patch")

# Heavy/binary noise excluded from BOTH snapshot and diff (source edits only).
_EXCLUDE_DIRS = {"__pycache__", ".git", "build", "dist", ".mypy_cache", ".pytest_cache"}
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".so", ".o", ".a", ".bin", ".pt", ".safetensors")


@dataclass(frozen=True)
class PatchArea:
    label: str      # short name used in a/<label> b/<label> headers
    category: str   # manifest bucket (auto_round | vllm | model_code)
    path: Path      # live directory on disk


def _package_dir(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec and spec.submodule_search_locations:
        return Path(os.path.realpath(next(iter(spec.submodule_search_locations))))
    return None


def _hf_modules_dir() -> Path | None:
    hf = os.environ.get("HF_HOME") or str(Path(os.environ.get("HOME", "/root")) / ".cache" / "huggingface")
    p = Path(hf) / "modules" / "transformers_modules"
    return p if p.is_dir() else None


def find_editable_areas(*, track_vllm: bool = True) -> list[PatchArea]:
    """Discover the source trees an agent fix may edit (present ones only)."""
    areas: list[PatchArea] = []
    ar = _package_dir("auto_round")
    if ar and ar.is_dir():
        areas.append(PatchArea("auto_round", "auto_round", ar))
    if track_vllm:
        vl = _package_dir("vllm")
        if vl and vl.is_dir():
            areas.append(PatchArea("vllm", "vllm", vl))
    hf = _hf_modules_dir()
    if hf:
        areas.append(PatchArea("model_code", "model_code", hf))
    return areas


def _keep(path: Path) -> bool:
    if path.suffix in _EXCLUDE_SUFFIXES:
        return False
    return not any(part in _EXCLUDE_DIRS for part in path.parts)


def _ignore(_dir: str, names: list[str]) -> set[str]:
    out = set()
    for n in names:
        if n in _EXCLUDE_DIRS or n.endswith(_EXCLUDE_SUFFIXES):
            out.add(n)
    return out


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return []


class PatchCapture:
    """Snapshot editable areas, then produce unified diffs of the agent's edits."""

    def __init__(self, snapshot_root: Path, *, track_vllm: bool = True,
                 areas: list[PatchArea] | None = None):
        self.snapshot_root = Path(snapshot_root)
        self.track_vllm = track_vllm
        self.areas: list[PatchArea] = areas if areas is not None else []
        self._snapped: set[str] = set()

    def snapshot(self) -> int:
        """Copy every discovered (or provided) editable area to the pristine root.

        Idempotent per area: an already-snapshotted area is never overwritten, so a
        late call (after packages install) captures new areas without clobbering
        baselines -- mirrors ``patch_snapshot_ensure``.
        """
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        if not self.areas:
            self.areas = find_editable_areas(track_vllm=self.track_vllm)
        added = 0
        for area in self.areas:
            if area.label in self._snapped:
                continue
            dst = self.snapshot_root / area.label
            if dst.exists():
                self._snapped.add(area.label)
                continue
            try:
                shutil.copytree(area.path, dst, ignore=_ignore, dirs_exist_ok=True)
                self._snapped.add(area.label)
                added += 1
            except (OSError, shutil.Error) as exc:
                _log.warning("patch: snapshot of %s failed: %s", area.label, exc)
        if added:
            _log.info("patch: snapshotted %d editable area(s)", added)
        return added

    def _area_diff(self, area: PatchArea) -> str:
        snap = self.snapshot_root / area.label
        if not snap.is_dir() or not area.path.is_dir():
            return ""
        chunks: list[str] = []
        cur_files = {p.relative_to(area.path): p for p in area.path.rglob("*")
                     if p.is_file() and _keep(p)}
        snap_files = {p.relative_to(snap): p for p in snap.rglob("*")
                      if p.is_file() and _keep(p)}
        for rel in sorted(set(cur_files) | set(snap_files), key=str):
            old = _read_lines(snap_files[rel]) if rel in snap_files else []
            new = _read_lines(cur_files[rel]) if rel in cur_files else []
            if old == new:
                continue
            diff = difflib.unified_diff(
                old, new,
                fromfile=f"a/{area.label}/{rel}",
                tofile=f"b/{area.label}/{rel}",
            )
            chunks.append("".join(diff))
        return "".join(chunks)

    def combined_diff(self) -> str:
        """Unified diff across all tracked areas (empty if nothing changed)."""
        return "".join(self._area_diff(a) for a in self.areas)

    def write_patches(self, out_dir: Path) -> dict[str, Path]:
        """Write a per-area patch + a combined patch under ``out_dir``; return the paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        combined_parts: list[str] = []
        for area in self.areas:
            d = self._area_diff(area)
            if not d:
                continue
            p = out_dir / f"{area.label}.patch"
            p.write_text(d)
            written[area.label] = p
            combined_parts.append(d)
        if combined_parts:
            combined = out_dir / "combined.patch"
            combined.write_text("".join(combined_parts))
            written["combined"] = combined
        return written

    # -- deterministic replay (apply a *recorded* patch, revert on failure) ----
    def _area_by_label(self, label: str) -> PatchArea | None:
        for area in self.areas:
            if area.label == label:
                return area
        return None

    def apply_recorded(self, patch_by_label: dict[str, Path]) -> bool:
        """Apply recorded per-area ``.patch`` files onto the live tree.

        Each label must map to a currently-tracked area; the diff was written with
        ``a/<label>/<rel>`` headers, so ``patch -p2`` (strip ``a/`` + ``<label>/``)
        applies it relative to the area dir. We dry-run *every* patch first and only
        commit if all apply cleanly -- an all-or-nothing atomic replay. A patch that
        doesn't apply (different package version, model family, ...) is a normal miss,
        not an error: we return ``False`` and the caller escalates to the agent.
        """
        if shutil.which("patch") is None:
            _log.info("replay: `patch` tool unavailable; skipping deterministic replay")
            return False
        targets: list[tuple[PatchArea, Path]] = []
        for label, patch_file in patch_by_label.items():
            if label == "combined":
                continue
            area = self._area_by_label(label)
            pf = Path(patch_file)
            if area is None or not area.path.is_dir() or not pf.is_file() or pf.stat().st_size == 0:
                return False
            targets.append((area, pf))
        if not targets:
            return False
        # Phase 1: dry-run everything.
        for area, pf in targets:
            if not self._patch(area.path, pf, dry_run=True):
                return False
        # Phase 2: commit (revert-all if any step unexpectedly fails).
        applied: list[PatchArea] = []
        for area, pf in targets:
            if self._patch(area.path, pf, dry_run=False):
                applied.append(area)
            else:
                self.restore()
                return False
        _log.info("replay: applied recorded patch to %d area(s): %s",
                  len(applied), ", ".join(a.label for a in applied))
        return True

    @staticmethod
    def _patch(area_path: Path, patch_file: Path, *, dry_run: bool) -> bool:
        cmd = ["patch", "-p2", "--forward", "--no-backup-if-mismatch",
               "-d", str(area_path), "-i", str(patch_file)]
        if dry_run:
            cmd.append("--dry-run")
        try:
            rc = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120, check=False).returncode
        except (OSError, subprocess.SubprocessError) as exc:
            _log.warning("replay: patch invocation failed: %s", exc)
            return False
        return rc == 0

    def restore(self) -> int:
        """Revert every tracked area to its pristine snapshot (undo agent/replay edits).

        Used after a failed replay (or a fix that must be rolled back) so the next
        attempt starts from a clean tree. Only snapshotted areas are restored.
        """
        restored = 0
        for area in self.areas:
            snap = self.snapshot_root / area.label
            if not snap.is_dir():
                continue
            try:
                if area.path.exists():
                    shutil.rmtree(area.path, ignore_errors=True)
                shutil.copytree(snap, area.path, ignore=_ignore, dirs_exist_ok=True)
                restored += 1
            except (OSError, shutil.Error) as exc:
                _log.warning("restore of %s failed: %s", area.label, exc)
        if restored:
            _log.info("replay: restored %d area(s) to pristine snapshot", restored)
        return restored

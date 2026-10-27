"""Local-first knowledge synchronization (pull/push with an outbox).

The pipeline writes lessons/patches/references locally first; this layer records
content hashes in ``.sync_state.json`` and pushes only changed files to the
configured HF dataset.  A missing repo/token/network never fails a run: the
change stays pending and is retried on the next run (or by an explicit sync).

Session trajectories are pushed separately under the dataset's ``trajectories/``
tree (see :meth:`KnowledgeStore.push_sessions`).
"""

from __future__ import annotations

try:  # POSIX; Windows falls back to no cross-process lock
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..logging_util import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from .knowledge import KnowledgeStore

_log = get_logger("agent.knowledge_sync")

SyncMode = Literal["auto", "pull", "push", "off", "background"]
_STATE_FILE = ".sync_state.json"
_KEEP_SUFFIXES = {".jsonl", ".json", ".patch", ".md"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SyncReport:
    """Auditable result of one pull/push cycle (written to knowledge_sync.json)."""

    status: str = "local_only"
    mode: str = "auto"
    repo: str | None = None
    path_in_repo: str | None = None
    pulled: int = 0
    pushed: int = 0
    sessions: int = 0
    pending: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    last_pull: str | None = None
    last_push: str | None = None
    revision: str | None = None
    session_stage: str | None = None
    run_snapshot: str | None = None
    snapshot_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "mode": self.mode,
            "repo": self.repo,
            "path_in_repo": self.path_in_repo,
            "pulled": self.pulled,
            "pushed": self.pushed,
            "sessions": self.sessions,
            "pending": list(self.pending),
            "errors": list(self.errors),
            "last_pull": self.last_pull,
            "last_push": self.last_push,
            "revision": self.revision,
            "session_stage": self.session_stage,
            "run_snapshot": self.run_snapshot,
            "snapshot_files": self.snapshot_files,
        }


class KnowledgeSync:
    """Pull/push a :class:`KnowledgeStore` with a local sha256 outbox."""

    def __init__(self, store: "KnowledgeStore", *, mode: SyncMode = "auto"):
        self.store = store
        self.mode = mode if mode in {"auto", "pull", "push", "off", "background"} else "auto"
        self.state_path = Path(store.local_dir) / _STATE_FILE
        self._state = self._load_state()
        self._baseline: dict[str, str] = {}
        self.report = SyncReport(
            mode=self.mode,
            repo=store.repo,
            path_in_repo=store.path_in_repo,
            last_pull=self._state.get("last_pull"),
            last_push=self._state.get("last_push"),
            revision=self._state.get("revision"),
        )

    # ── state ────────────────────────────────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self) -> None:
        self._state.setdefault("schema_version", 1)
        self._state["repo"] = self.store.repo
        self._state["path_in_repo"] = self.store.path_in_repo
        self.state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @contextmanager
    def _lock(self):
        """Serialize pull/push between concurrent runs on the same host."""
        lock_path = Path(self.store.local_dir) / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as stream:
            if fcntl is not None:
                fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def _local_files(self) -> dict[str, str]:
        root = Path(self.store.local_dir)
        out: dict[str, str] = {}
        if not root.is_dir():
            return out
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == _STATE_FILE:
                continue
            if any(part.startswith(".") or part == "__pycache__"
                   for part in path.relative_to(root).parts):
                continue
            if path.suffix not in _KEEP_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            out[rel] = _sha256(path)
        return out

    def pending(self) -> list[str]:
        files = self._local_files()
        pushed = self._state.get("files")
        pushed = pushed if isinstance(pushed, dict) else {}
        return [rel for rel, digest in files.items() if pushed.get(rel) != digest]

    def mark_baseline(self) -> None:
        """Record the pre-run local state; used for a run-scoped sink snapshot."""
        self._baseline = self._local_files()

    def changed_since_baseline(self) -> list[str]:
        """Files created/changed by this run (not pre-existing shared cache)."""
        current = self._local_files()
        return sorted(rel for rel, digest in current.items()
                      if self._baseline.get(rel) != digest)

    def status(self) -> dict[str, Any]:
        # A later stage (e.g. session sediment) may append an error after a
        # successful file push; never report success while errors/pending remain.
        if (self.report.errors or self.report.pending) and self.report.status in {
                "pulled", "pushed", "up_to_date"}:
            self.report.status = "pending"
        return self.report.to_dict()

    def _settle(self, status: str) -> None:
        """Never report success while a previous step left pending/errors."""
        self.report.status = (
            "pending" if self.report.errors or self.report.pending else status)

    # ── pull ─────────────────────────────────────────────────────────────────
    def pull(self) -> SyncReport:
        """Sync lessons/patches/references down from the HF dataset."""
        if self.mode not in {"auto", "pull", "background"}:
            self.report.status = "local_only"
            return self.report
        if not self.store.repo:
            self.report.status = "local_only"
            return self.report
        try:
            with self._lock():
                ok = self.store.sync_down()
        except Exception as exc:  # noqa: BLE001 - offline must not fail a run
            self.report.errors.append(f"pull: {exc}")
            self.report.status = "pending"
            _log.warning("knowledge sync: pull failed: %s", exc)
            return self.report
        if ok:
            self.report.pulled = len(self._local_files())
            self.report.last_pull = _now()
            self._state["last_pull"] = self.report.last_pull
            self._save_state()
            self.mark_baseline()
            self._settle("pulled")
        else:
            self.report.status = "local_only"
        return self.report

    # ── push ─────────────────────────────────────────────────────────────────
    def push(self) -> SyncReport:
        """Push only changed local knowledge files; leave failures pending."""
        if self.mode == "background":
            self.report.pending = self.pending()
            self.report.status = "pending"
            return self.report
        if self.mode not in {"auto", "push"}:
            self.report.status = "local_only"
            return self.report
        if not self.store.repo or not getattr(self.store, "push", True):
            self.report.status = "local_only"
            return self.report
        pending = self.pending()
        self.report.pending = pending
        if not pending:
            self._settle("up_to_date")
            return self.report
        try:
            with self._lock():
                ok = self.store.push_up(allow_patterns=pending)
        except Exception as exc:  # noqa: BLE001 - never fail a run on a push
            self.report.errors.append(f"push: {exc}")
            self.report.status = "pending"
            _log.warning("knowledge sync: push failed: %s", exc)
            return self.report
        if not ok:
            self.report.status = "pending"
            self.report.errors.append("push: store.push_up returned false")
            return self.report
        files = self._state.get("files")
        files = files if isinstance(files, dict) else {}
        current = self._local_files()
        for rel in pending:
            if rel in current:
                files[rel] = current[rel]
        self._state["files"] = files
        self.report.pushed = len(pending)
        self.report.pending = []
        self.report.last_push = _now()
        self._settle("pushed")
        self._state["last_push"] = self.report.last_push
        self._save_state()
        _log.info("knowledge sync: pushed %d file(s) to %s",
                  len(pending), self.store.repo)
        return self.report

    # ── sessions ─────────────────────────────────────────────────────────────
    def push_sessions(self, stage_dir: str | Path | None) -> SyncReport:
        """Push staged ATIF sessions to ``<dataset>/trajectories/``."""
        if not stage_dir:
            return self.report
        stage = Path(stage_dir)
        if not stage.is_dir():
            return self.report
        count = len([p for p in stage.rglob("*") if p.is_file()])
        self.report.sessions = count
        self.report.session_stage = str(stage)
        if self.mode == "background":
            self.report.status = "pending"
            return self.report
        if self.mode not in {"auto", "push"} or not self.store.repo:
            self.report.status = "local_only" if not self.store.repo else self.report.status
            return self.report
        try:
            with self._lock():
                ok = self.store.push_sessions(stage)
        except Exception as exc:  # noqa: BLE001
            self.report.errors.append(f"sessions: {exc}")
            self.report.status = "pending"
            _log.warning("knowledge sync: session push failed: %s", exc)
            return self.report
        if ok:
            self._settle("pushed")
            self.report.last_push = _now()
            self._state["last_push"] = self.report.last_push
            self._state["sessions_pushed"] = self._state.get("sessions_pushed", 0) + count
            self._save_state()
            _log.info("knowledge sync: pushed %d session(s) to %s/trajectories",
                      count, self.store.repo)
        else:
            self.report.status = "pending"
        return self.report

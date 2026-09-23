"""Shared, persistent fix-knowledge store: lessons + replayable patches.

The generic fix-loop learns from every run -- but only if what it learns *outlives*
the run. This store keeps lessons and successful patches in a location that is
shared across runs, models and machines, keyed by the error **signature** (not the
model), so a fix discovered once can be replayed deterministically the next time
the same failure appears -- with no agent call at all.

Backends:
  * **Local** (always): a directory (default ``~/.cache/aqa/knowledge``) shared by
    every run on the host.
  * **Hugging Face dataset** (when ``repo`` is set): synced *down* into the local
    dir at run start and pushed *up* after a successful fix, giving cross-machine
    sharing across ephemeral remote containers.

Layout (mirrored locally and in ``<repo>/<path_in_repo>/``)::

    lessons/<phase>.jsonl                      # dedup-by-signature lessons
    patches/<phase>/<sig_hash>/
        <label>.patch  ...                     # per-area unified diffs
        meta.json                              # signature/error_class/model/scheme/ts

Everything is best-effort: a missing repo/token, offline host or dirty patch is a
normal miss, never a hard failure -- the caller just falls back to the agent.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_util import get_logger
from ..report.sinks import resolve_hf_token
from .drift import classify_error, exception_type
from .lessons import LessonStore, extract_signature
from .trajectories import TrajectoryStore, TrajectorySummary

_log = get_logger("agent.knowledge")


def default_local_dir() -> Path:
    """The default host-shared knowledge dir (``$AQA_KNOWLEDGE_DIR`` or XDG cache)."""
    import os

    env = os.environ.get("AQA_KNOWLEDGE_DIR")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path(os.environ.get("HOME", "/root")) / ".cache")
    return Path(cache) / "aqa" / "knowledge"


def signature_hash(signature: str) -> str:
    """Stable short id for an error signature (patch-archive directory name)."""
    return hashlib.sha1(signature.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class PatchRecord:
    """A recorded, replayable fix keyed by an error signature."""

    phase: str
    sig_hash: str
    signature: str
    error_class: str
    patch_files: dict[str, Path]  # label -> per-area .patch path (no "combined")
    exc_type: str = ""            # exact exception class name (structured key)
    model_id: str = ""
    scheme: str = ""
    method: str = ""
    ts: str = ""

    def diff_text(self, max_chars: int = 4000) -> str:
        """Concatenated unified-diff text of this record's per-area patches.

        Used to hand the recorded fix to the agent as *reference* (Option A): the
        patch is never auto-applied, so the agent reads these diffs and decides
        whether the mechanism applies to the current model/checkpoint. Large diffs
        are summarised + pointed-to instead of inlined (see :meth:`stat_summary` /
        :meth:`patch_paths`) so a multi-file source patch never bloats the prompt
        or, worse, gets hard-truncated mid-hunk.
        """
        chunks: list[str] = []
        for label, path in sorted(self.patch_files.items()):
            try:
                body = Path(path).read_text()
            except OSError:
                continue
            if body.strip():
                chunks.append(f"# area: {label}\n{body.rstrip()}")
        text = "\n".join(chunks)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text

    def stat_summary(self) -> str:
        """Compact 'N files [paths], +A/-D lines' description of this record's diff."""
        files: list[str] = []
        adds = dels = 0
        for line in self.diff_text(max_chars=200_000).splitlines():
            if line.startswith("+++ "):
                files.append(line[4:].strip())
            elif line.startswith("+") and not line.startswith("+++"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1
        flist = ", ".join(files) if files else "?"
        return f"touches {len(files)} file(s) [{flist}], +{adds}/-{dels} lines"

    def patch_paths(self) -> list[str]:
        """On-disk paths of the per-area .patch files (for on-demand agent reading)."""
        return [str(self.patch_files[k]) for k in sorted(self.patch_files)]

@dataclass
class KnowledgeStore:
    """Local (optionally HF-dataset-backed) shared lessons + patch archive."""

    local_dir: Path
    repo: str | None = None
    path_in_repo: str = "knowledge"
    token_env: str | None = None
    endpoint: str | None = None
    push: bool = True
    lessons: LessonStore = field(init=False)
    trajectories: TrajectoryStore = field(init=False)

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir).expanduser()
        (self.local_dir / "lessons").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "patches").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        self.lessons = LessonStore(self.local_dir / "lessons")
        self.trajectories = TrajectoryStore(self.local_dir / "trajectories")

    @classmethod
    def from_config(cls, cfg) -> KnowledgeStore | None:
        """Build a store from a ``KnowledgeConfig`` (returns None when disabled)."""
        if not getattr(cfg, "enabled", True):
            return None
        local = Path(cfg.local_dir).expanduser() if cfg.local_dir else default_local_dir()
        return cls(
            local_dir=local,
            repo=cfg.repo,
            path_in_repo=cfg.path_in_repo,
            token_env=cfg.token_env,
            endpoint=cfg.endpoint,
            push=cfg.push,
        )

    # -- HF sync ---------------------------------------------------------
    def sync_down(self) -> bool:
        """Pull the knowledge subtree from the HF dataset into the local dir.

        No-op (returns False) without a ``repo``; best-effort otherwise -- an
        offline host or a fresh/empty repo simply leaves the local cache as-is.
        """
        if not self.repo:
            return False
        token = resolve_hf_token(None, self.token_env)
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            _log.info("knowledge: huggingface_hub not installed; local-only mode")
            return False
        try:
            snap = snapshot_download(
                repo_id=self.repo, repo_type="dataset", token=token,
                endpoint=self.endpoint, allow_patterns=[f"{self.path_in_repo}/**"],
            )
        except Exception as exc:  # noqa: BLE001 - offline / empty repo / auth: soft-fail
            _log.info("knowledge: sync_down skipped (%s)", exc)
            return False
        src = Path(snap) / self.path_in_repo
        if not src.is_dir():
            return False
        merged = self._merge_tree(src, self.local_dir)
        _log.info("knowledge: synced down from %s (%d file(s))", self.repo, merged)
        return True

    def _merge_tree(self, src: Path, dst: Path) -> int:
        """Copy ``src`` into ``dst``; lessons JSONL is merged (dedup by signature)."""
        count = 0
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if (rel.parts and rel.parts[0] in ("lessons", "trajectories")
                    and target.exists()):
                self._merge_lessons(target, p)
            else:
                shutil.copy2(p, target)
            count += 1
        return count

    @staticmethod
    def _merge_lessons(local: Path, remote: Path) -> None:
        by_sig: dict[str, dict] = {}
        for src in (local, remote):  # remote wins on conflict (freshest shared truth)
            for line in src.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_sig[obj.get("signature", line)] = obj
        local.write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in by_sig.values()) + "\n"
        )

    def push_up(self) -> bool:
        """Upload the local knowledge dir back to the HF dataset (best-effort)."""
        if not self.repo or not self.push:
            return False
        token = resolve_hf_token(None, self.token_env)
        if not token:
            _log.info("knowledge: no write token; not pushing to %s", self.repo)
            return False
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return False
        try:
            api = HfApi(token=token, endpoint=self.endpoint)
            api.upload_folder(
                repo_id=self.repo, repo_type="dataset", folder_path=str(self.local_dir),
                path_in_repo=self.path_in_repo, token=token,
                commit_message="aqa: update shared fix-knowledge",
            )
        except Exception as exc:  # noqa: BLE001 - never fail a run on a knowledge push
            _log.warning("knowledge: push_up to %s failed: %s", self.repo, exc)
            return False
        _log.info("knowledge: pushed lessons/patches to %s", self.repo)
        return True

    # -- patch archive ---------------------------------------------------
    def _patch_root(self, phase: str) -> Path:
        return self.local_dir / "patches" / phase

    def find_patches(self, phase: str, error_text: str,
                     error_class: str = "") -> list[PatchRecord]:
        """Recorded patches for this failure, best-match first.

        Ranking tiers (finest first): exact signature > exact exception class +
        semantic category > exact exception class > semantic category > other.
        Newest ``ts`` breaks ties within a tier. Cross-model by design -- the key
        is the error, not the model. ``patch --dry-run`` is the real safety net,
        so a looser tier only ever affects *which candidate is tried first*.
        """
        root = self._patch_root(phase)
        if not root.is_dir():
            return []
        signature = extract_signature(error_text)
        want_class = error_class or classify_error(error_text)
        want_exc = exception_type(error_text)
        records: list[PatchRecord] = []
        for meta_path in root.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            patch_files = {
                p.stem: p for p in meta_path.parent.glob("*.patch") if p.stem != "combined"
            }
            if not patch_files:
                continue
            records.append(PatchRecord(
                phase=phase, sig_hash=meta_path.parent.name,
                signature=meta.get("signature", ""), error_class=meta.get("error_class", ""),
                patch_files=patch_files, exc_type=meta.get("exc_type", ""),
                model_id=meta.get("model_id", ""),
                scheme=meta.get("scheme", ""), method=meta.get("method", ""),
                ts=meta.get("ts", ""),
            ))

        def rank(r: PatchRecord) -> int:
            if r.signature and r.signature == signature:
                return 0
            exc_ok = bool(want_exc) and r.exc_type == want_exc
            cls_ok = want_class not in ("", "unknown") and r.error_class == want_class
            if exc_ok and cls_ok:
                return 1
            if exc_ok:
                return 2
            if cls_ok:
                return 3
            return 4

        # Newest-first within each match tier: sort by ts desc, then stably by tier.
        records.sort(key=lambda r: r.ts, reverse=True)
        records.sort(key=rank)
        return records

    def record(self, phase: str, error_text: str, patch_files: dict[str, Path], *,
               error_class: str = "", model_id: str = "", scheme: str = "",
               method: str = "") -> PatchRecord | None:
        """Archive a successful fix's patches keyed by the error signature.

        ``patch_files`` is the ``{label: path}`` map from
        :meth:`PatchCapture.write_patches` (the ``combined`` entry is ignored).
        Returns the stored record, or None if there were no per-area patches.
        """
        per_area = {k: v for k, v in patch_files.items() if k != "combined"}
        if not per_area:
            return None
        signature = extract_signature(error_text)
        err_class = error_class or classify_error(error_text)
        exc = exception_type(error_text)
        sig_hash = signature_hash(signature)
        dest = self._patch_root(phase) / sig_hash
        dest.mkdir(parents=True, exist_ok=True)
        stored: dict[str, Path] = {}
        for label, src in per_area.items():
            src = Path(src)
            if not src.is_file():
                continue
            out = dest / f"{label}.patch"
            shutil.copy2(src, out)
            stored[label] = out
        if not stored:
            return None
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (dest / "meta.json").write_text(json.dumps({
            "signature": signature, "error_class": err_class, "exc_type": exc,
            "model_id": model_id, "scheme": scheme, "method": method, "ts": ts,
        }, ensure_ascii=False, indent=2))
        _log.info("knowledge: recorded patch for '%s'/%s (%s) -> %s",
                  err_class, exc or "-", signature[:60], sig_hash)
        return PatchRecord(
            phase=phase, sig_hash=sig_hash, signature=signature, error_class=err_class,
            patch_files=stored, exc_type=exc, model_id=model_id, scheme=scheme,
            method=method, ts=ts,
        )

    # -- replay statistics (埋点) -----------------------------------------
    _STAT_OUTCOMES = ("replay_hit", "reference_used", "agent_fixed", "unresolved")

    def record_trajectory(self, summary: TrajectorySummary) -> dict | None:
        """Persist one distilled fix-trajectory summary (best-effort).

        Complements :meth:`record` (the replayable patch) with the *reasoning*
        behind it, keyed by the same signature. Never raises -- knowledge
        capture must not fail a run.
        """
        try:
            return self.trajectories.save(summary.phase, summary)
        except OSError as exc:
            _log.warning("knowledge: could not record trajectory: %s", exc)
            return None

    def _stats_path(self) -> Path:
        return self.local_dir / "stats" / "events.jsonl"

    def log_event(self, phase: str, outcome: str, *, sig_hash: str = "",
                  error_class: str = "", model_id: str = "", tier: str = "") -> None:
        """Append one replay/heal outcome event (best-effort telemetry).

        ``outcome`` is one of :data:`_STAT_OUTCOMES`: ``replay_hit`` (a recorded
        patch resolved the failure with no agent call -- legacy, auto-apply is
        disabled under reference-only replay so this stays 0), ``reference_used``
        (an agent tier resolved it *while* a matching recorded fix was injected as
        reference, i.e. knowledge demonstrably contributed), ``agent_fixed`` (an
        agent tier resolved it with no matching recorded fix available), or
        ``unresolved`` (gave up). Drives the hit-rate metric.
        """
        path = self._stats_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "phase": phase, "outcome": outcome, "sig_hash": sig_hash,
                "error_class": error_class, "model_id": model_id, "tier": tier,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            _log.warning("knowledge: could not log stat event: %s", exc)

    def stats(self) -> dict:
        """Aggregate replay telemetry: counts per outcome + knowledge hit-rate.

        ``hit_rate`` = (replay_hit + reference_used) / (all outcomes), i.e. the
        fraction of *fixable* failures where recorded knowledge demonstrably
        contributed -- either auto-applied (``replay_hit``, legacy/0 under
        reference-only replay) or consulted-and-adopted by the agent
        (``reference_used``). ``agent_fixed`` counts fixes the agent found with no
        matching recorded knowledge available.
        """
        counts = dict.fromkeys(self._STAT_OUTCOMES, 0)
        path = self._stats_path()
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    outcome = json.loads(line).get("outcome", "")
                except json.JSONDecodeError:
                    continue
                if outcome in counts:
                    counts[outcome] += 1
        total = sum(counts.values())
        counts["total"] = total
        knowledge_hits = counts["replay_hit"] + counts["reference_used"]
        counts["hit_rate"] = round(knowledge_hits / total, 4) if total else 0.0
        counts["trajectories"] = self._count_trajectories()
        return counts

    def _count_trajectories(self) -> int:
        """Total distilled trajectory summaries stored across all phases."""
        root = self.local_dir / "trajectories"
        if not root.is_dir():
            return 0
        n = 0
        for p in root.glob("*.jsonl"):
            for line in p.read_text(errors="replace").splitlines():
                if line.strip():
                    n += 1
        return n

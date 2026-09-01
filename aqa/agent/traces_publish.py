"""Layered trajectory publishing to a HF dataset repo (M4).

Unlike ``traces export --push`` (flat dataset rows), ``publish`` archives each
agent session as a directory tree so the *complete* session, its ATIF
trajectory, and both dataset views stay addressable side by side::

    trajectories/<session_id>/trajectory.json    # full ATIF session
    trajectories/<session_id>/dataset_sft.jsonl  # SFT row (OpenAI messages)
    trajectories/<session_id>/dataset_rl.jsonl   # RL row (NeMo-Gym task)
    knowledge/<phase>.jsonl                      # optional distilled entries

Upload is ``HfApi.upload_folder`` onto the dataset repo; tokens come from the
environment only (``--token-env`` or HF_TOKEN...).  Staging is local-first:
``--stage-only`` builds the tree without uploading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..logging_util import AqaError, get_logger
from .traces_export import atif_to_rl_row, atif_to_sft_row

_log = get_logger("agent.traces_publish")

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_session_id(raw: str) -> str:
    """Session ids become path segments -- sanitize (no traversal, no spaces)."""
    cleaned = _SAFE.sub("_", (raw or "").strip()).strip("._")
    return cleaned or "unknown"


def _iter_atif(paths: Iterable[str | Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "trajectories.jsonl"
        if not p.is_file():
            _log.warning("no trajectories.jsonl under %s; skipped", raw)
            continue
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield p, json.loads(line)
            except json.JSONDecodeError:
                _log.warning("%s: skipping malformed line", p)


def stage_trajectory_archive(
    paths: Iterable[str | Path], dest: str | Path, *,
    views: tuple[str, ...] = ("sft", "rl"),
) -> dict[str, Any]:
    """Stage the layered archive under ``dest``; returns a staging manifest.

    One directory per session: full ATIF + one dataset file per requested view.
    Sessions whose trajectory converts to no dataset row still keep their raw
    trajectory.json (the archive is the complete record; datasets are views).
    """
    dest = Path(dest)
    written: list[str] = []
    for src_path, atif in _iter_atif(paths):
        session = _safe_session_id(str(atif.get("session_id") or atif.get("trajectory_id") or ""))
        sess_dir = dest / "trajectories" / session
        sess_dir.mkdir(parents=True, exist_ok=True)
        # trajectories.jsonl 可能早于适配器侧脱敏；归档前统一再过一遍
        from .redact import redact_atif

        (sess_dir / "trajectory.json").write_text(
            json.dumps(redact_atif(atif), indent=2, ensure_ascii=False) + "\n")
        written.append(f"trajectories/{session}/trajectory.json")
        for view in views:
            row = (atif_to_sft_row(atif, run=str(src_path.parent)) if view == "sft"
                   else atif_to_rl_row(atif, run=str(src_path.parent)))
            if row is None:
                continue
            (sess_dir / f"dataset_{view}.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n")
            written.append(f"trajectories/{session}/dataset_{view}.jsonl")
    manifest = {"sessions": len({w.split('/')[1] for w in written}),
                "files": written}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish_archive(
    staged: str | Path, repo: str, *, token_env: str | None = None,
    path_in_repo: str = "",
) -> str:
    """Upload the staged tree onto the HF dataset repo (token from env only)."""
    from ..report.sinks import resolve_hf_token

    token = resolve_hf_token(None, token_env)
    if not token:
        raise AqaError(
            "publishing traces needs an HF token in the environment "
            "(HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or --token-env <VAR>; never inline)"
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dep
        raise AqaError("publishing needs `huggingface_hub`: pip install huggingface_hub") from exc
    staged = Path(staged)
    if not (staged / "manifest.json").exists():
        raise AqaError(f"{staged} is not a staged archive (no manifest.json)")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(staged), path_in_repo=path_in_repo,
                    repo_id=repo, repo_type="dataset")
    url = f"https://huggingface.co/datasets/{repo}/tree/main/{path_in_repo}".rstrip("/")
    _log.info("archive uploaded to %s", url)
    return url

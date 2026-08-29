"""Assemble SFT training datasets from aqa trajectory stores.

Harbor ``traces export`` parity: two trajectory sources become rows of
OpenAI-style ``{"messages": [...], "metadata": {...}}`` (optionally with a
ShareGPT ``conversations`` column added):

* **agent_eval runs** -- the ``trajectories.jsonl`` (ATIF dicts) a run dumps
  next to its results (:func:`aqa.agent_eval.atif.dump_trajectories_jsonl`);
* **the fix-loop knowledge root** -- distilled *successful-fix* summaries
  (:class:`~aqa.agent.trajectories.TrajectoryStore`) plus the patch diffs
  archived by :class:`~aqa.agent.knowledge.KnowledgeStore`, rendered as
  persona/system + error-context user + structured-analysis assistant rows.

Heavy deps (``datasets``, ``huggingface_hub``) are imported lazily so the JSONL
path works with the base install; tokens come from the environment only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_util import AqaError, get_logger
from ..report.sinks import resolve_hf_token

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

_log = get_logger("agent.traces_export")

# Persona for fix-loop SFT rows: the assistant answers as a quant-fix expert.
_FIXLOOP_SYSTEM = (
    "You are an expert quantization engineer diagnosing failures in an automated "
    "quantization pipeline (auto-round / vllm / lm-eval). Given an error "
    "signature and run context, explain the root cause and the fix that worked."
)

# ShareGPT role names for each OpenAI message role (tool output folds to human).
_SHAREGPT_FROM = {"system": "system", "user": "human", "assistant": "gpt", "tool": "human"}


# ── source auto-detection ────────────────────────────────────────────────────
def detect_source(path: str | Path) -> str | None:
    """Classify ``path`` as ``"agent_eval"`` / ``"fixloop"`` (None = neither)."""
    p = Path(path)
    if p.is_file():
        return "agent_eval" if p.suffix == ".jsonl" else None
    if (p / "trajectories.jsonl").is_file():
        return "agent_eval"
    if (p / "trajectories").is_dir() or (p / "patches").is_dir():
        return "fixloop"  # knowledge-root layout (trajectories/ + patches/)
    return None


# ── agent_eval: ATIF -> messages ─────────────────────────────────────────────
def atif_to_messages(atif: dict[str, Any]) -> list[dict[str, Any]]:
    """Map ATIF steps to OpenAI chat messages (tool calls + tool results)."""
    messages: list[dict[str, Any]] = []
    for step in atif.get("steps", []):
        source = step.get("source")
        if source in ("system", "user"):
            messages.append({"role": source, "content": step.get("message", "")})
        elif source == "agent":
            msg: dict[str, Any] = {"role": "assistant", "content": step.get("message", "")}
            calls = step.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("tool_call_id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("function_name", ""),
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    }
                    for tc in calls
                ]
            messages.append(msg)
            for res in (step.get("observation") or {}).get("results", []):
                messages.append({
                    "role": "tool",
                    "tool_call_id": res.get("source_call_id") or "",
                    "content": res.get("content", ""),
                })
    return messages


def _atif_files(paths: Iterable[str | Path]) -> list[Path]:
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "trajectories.jsonl"
        if p.is_file():
            files.append(p)
        else:
            _log.warning("no trajectories.jsonl under %s; skipped", raw)
    return files


def export_agent_eval(
    paths: Iterable[str | Path], *, success_only: bool = True
) -> list[dict[str, Any]]:
    """Read ``trajectories.jsonl`` files into ``messages`` rows.

    Rows whose assistant turns are all empty carry no training signal and are
    skipped.  ``success_only`` keeps trajectories with ``solved``/``reward>0``
    (rows lacking outcome metadata are kept -- nothing to judge them by).
    """
    rows: list[dict[str, Any]] = []
    for path in _atif_files(paths):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                atif = json.loads(line)
            except json.JSONDecodeError:
                _log.warning("%s: skipping malformed line", path)
                continue
            extra = atif.get("extra") or {}
            # Rows lacking outcome metadata are kept -- nothing to judge them by.
            if (success_only and ("solved" in extra or "reward" in extra)
                    and not (extra.get("solved") or (extra.get("reward") or 0) > 0)):
                continue
            messages = atif_to_messages(atif)
            assistants = [m for m in messages if m["role"] == "assistant"]
            if not any(m["content"].strip() or m.get("tool_calls") for m in assistants):
                continue
            rows.append({
                "messages": messages,
                "metadata": {
                    "task": extra.get("task_id") or atif.get("session_id", ""),
                    "model": (atif.get("agent") or {}).get("model_name", ""),
                    "reward": extra.get("reward"),
                    "solved": extra.get("solved"),
                    "run": str(path.parent),
                    "source": "agent_eval",
                },
            })
    return rows


# ── fix-loop knowledge store -> SFT rows ─────────────────────────────────────
def _patch_diff(knowledge_root: Path, phase: str, signature: str) -> str:
    """Concatenated diff archived for ``signature`` under the knowledge root."""
    if not signature or not phase:
        return ""
    from .knowledge import PatchRecord, signature_hash

    dest = knowledge_root / "patches" / phase / signature_hash(signature)
    patch_files = {p.stem: p for p in dest.glob("*.patch") if p.stem != "combined"}
    if not patch_files:
        return ""
    record = PatchRecord(phase=phase, sig_hash=dest.name, signature=signature,
                         error_class="", patch_files=patch_files)
    return record.diff_text()


def _fixloop_row(summary: dict[str, Any], patch_diff: str) -> dict[str, Any]:
    user_lines = [f"Error signature: {summary.get('signature', '')}"]
    for key, label in (("error_class", "error class"), ("exc_type", "exception"),
                       ("component", "component"), ("model_id", "model")):
        if summary.get(key):
            user_lines.append(f"{label}: {summary[key]}")
    quant = " ".join(x for x in (summary.get("scheme"), summary.get("method")) if x)
    if quant:
        user_lines.append(f"quant: {quant}")

    assistant_lines = []
    if summary.get("root_cause"):
        assistant_lines.append(f"Root cause: {summary['root_cause']}")
    if summary.get("fix_rationale"):
        assistant_lines.append(f"Fix: {summary['fix_rationale']}")
    if summary.get("smoke_test"):
        assistant_lines.append(f"Smoke test: {summary['smoke_test']}")
    rejected = summary.get("tried_and_rejected") or []
    if rejected:
        assistant_lines.append("Dead ends avoided:\n" + "\n".join(f"- {t}" for t in rejected))
    if patch_diff:
        assistant_lines.append(f"Patch:\n```diff\n{patch_diff}\n```")

    return {
        "messages": [
            {"role": "system", "content": _FIXLOOP_SYSTEM},
            {"role": "user", "content": "\n".join(user_lines)},
            {"role": "assistant", "content": "\n\n".join(assistant_lines)},
        ],
        "metadata": {
            "signature": summary.get("signature", ""),
            "tier": summary.get("tier", ""),
            "phase": summary.get("phase", ""),
            "run_id": summary.get("run_id", ""),
            "model": summary.get("model_id", ""),
            "solved": True,
            "source": "fixloop",
        },
    }


def export_fixloop(
    root: str | Path | None = None, *, success_only: bool = True
) -> list[dict[str, Any]]:
    """SFT rows from the fix-loop's distilled trajectory summaries.

    ``root`` is the knowledge root (with ``trajectories/`` + ``patches/``
    subdirs) or a bare TrajectoryStore dir; defaults to the shared host store.
    The store holds only successful fixes, so every row is a positive exemplar
    (``success_only`` is kept for interface parity and just re-checks outcome).
    """
    from .knowledge import default_local_dir
    from .trajectories import TrajectoryStore

    root = Path(root).expanduser() if root else default_local_dir()
    traj_root = root / "trajectories" if (root / "trajectories").is_dir() else root
    store = TrajectoryStore(traj_root)
    rows: list[dict[str, Any]] = []
    for phase_file in sorted(traj_root.glob("*.jsonl")):
        phase = phase_file.stem
        for summary in store.load(phase):
            if success_only and summary.get("outcome", "fixed") != "fixed":
                continue
            diff = _patch_diff(root, phase, summary.get("signature", ""))
            rows.append(_fixloop_row(summary, diff))
    return rows


# ── output formats ───────────────────────────────────────────────────────────
def add_sharegpt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a ShareGPT ``conversations`` column derived from ``messages``."""
    for row in rows:
        row["conversations"] = [
            {"from": _SHAREGPT_FROM.get(m["role"], "human"), "value": m["content"]}
            for m in row["messages"]
        ]
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write rows as JSONL (one dataset row per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    return path


def write_parquet(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write rows as a parquet file (lazy ``datasets`` dependency)."""
    try:
        import datasets
    except ImportError as exc:
        raise AqaError(
            "parquet export needs the `datasets` package: pip install datasets"
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))
    return path


def push_to_hub(
    rows: list[dict[str, Any]], repo: str, *, token_env: str | None = None
) -> None:
    """Push rows to a HF dataset repo (token from the environment only)."""
    token = resolve_hf_token(None, token_env)
    if not token:
        raise AqaError(
            "pushing traces needs an HF token in the environment "
            "(HF_TOKEN / HUGGING_FACE_HUB_TOKEN; never inline)"
        )
    try:
        import datasets
    except ImportError as exc:
        raise AqaError(
            "pushing traces needs the `datasets` package: pip install datasets"
        ) from exc
    datasets.Dataset.from_list(rows).push_to_hub(repo, token=token)
    _log.info("traces: pushed %d row(s) to hf.co/datasets/%s", len(rows), repo)

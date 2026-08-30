"""Knowledge distillation from agent trajectories (M3).

Extracts reusable fix knowledge from ATIF trajectories and writes it back into
the shared knowledge store (:class:`~aqa.agent.trajectories.TrajectoryStore`),
so a later fix-loop run starts with the pattern instead of rediscovering it.

Rule-based by default (no LLM needed): an error observation followed by
successful progress yields one entry -- signature/error_class classify the
error, root_cause/fix_rationale come from the surrounding agent messages,
tried_and_rejected collects the failed commands.  An optional ``summarizer``
callable (trajectory -> fields) can refine entries with an agent pass; the
rule-based fields are the fallback and the floor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from ..logging_util import get_logger
from .drift import classify_error, exception_type
from .lessons import extract_signature
from .trajectories import TrajectoryStore, TrajectorySummary

_log = get_logger("agent.knowledge_extract")

# Observation content that reads as an error (tool stderr / exceptions / non-zero
# exits).  Conservative on purpose: a false positive just yields a weak entry.
_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\b[A-Za-z_.]*(?:Error|Exception)\b|"
    r"exit_code[\"']?\s*[:=]\s*[1-9]|is_error[\"']?\s*[:=]\s*true|"
    r"command failed|No such file|Permission denied|CUDA out of memory",
    re.IGNORECASE,
)

_MAX_FIELD = 400  # per-field text cap, mirrors build_summary


def _is_error_observation(content: Any, extra: dict[str, Any] | None = None) -> bool:
    if extra and (extra.get("is_error") is True):
        return True
    if extra and isinstance(extra.get("exit_code"), int) and extra["exit_code"] != 0:
        return True
    return isinstance(content, str) and bool(_ERROR_RE.search(content))


def _commands_of(step: dict[str, Any]) -> list[str]:
    out = []
    for tc in step.get("tool_calls") or []:
        args = tc.get("arguments") or {}
        cmd = args.get("command") or args.get("cmd") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        if cmd:
            out.append(str(cmd))
    return out


def extract_from_atif(
    atif: dict[str, Any],
    *,
    model_id: str = "",
    run_id: str = "",
    summarizer: Callable[[dict[str, Any]], dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Rule-extract fix-knowledge entries from one ATIF trajectory.

    One entry per (error, subsequent-progress) pair; empty when the trajectory
    has no errors or nothing recovered (no signal to reuse).
    """
    steps = atif.get("steps") or []
    extra = atif.get("extra") or {}
    solved = bool(extra.get("solved")) or (extra.get("reward") or 0) > 0
    entries: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        if step.get("source") != "agent":
            continue
        results = (step.get("observation") or {}).get("results") or []
        errors = [r for r in results
                  if _is_error_observation(r.get("content"), r.get("extra"))]
        if not errors:
            continue

        # The commands that produced the error are the dead ends.
        rejected = _commands_of(step)
        # Resolution = the next agent step(s) after this error.
        later = [s for s in steps[i + 1:] if s.get("source") == "agent"]
        if not later:
            continue
        next_msg = (later[0].get("message") or "").strip()
        fix_cmds = [c for s in later[:3] for c in _commands_of(s)]
        if not solved and not next_msg and not fix_cmds:
            continue  # error with no visible follow-up: nothing to learn

        err_text = "\n".join(str(r.get("content", "")) for r in errors)
        entry = TrajectorySummary(
            signature=extract_signature(err_text),
            error_class=classify_error(err_text),
            exc_type=exception_type(err_text),
            outcome="fixed" if solved else "progressed",
            root_cause=next_msg[:_MAX_FIELD],
            fix_rationale=("\n".join(f"- {c}" for c in fix_cmds))[:_MAX_FIELD],
            tried_and_rejected=rejected[:6],
            steps=len(steps),
            model_id=model_id or (atif.get("agent") or {}).get("model_name", "unknown"),
            run_id=run_id,
            phase="trajectory",
        ).to_dict()
        if summarizer is not None:
            try:
                refined = summarizer(atif) or {}
                for key in ("root_cause", "fix_rationale", "smoke_test"):
                    if refined.get(key):
                        entry[key] = str(refined[key])[:_MAX_FIELD]
            except Exception as exc:  # noqa: BLE001 - summarizer is best-effort
                _log.warning("summarizer failed, keeping rule-based entry: %s", exc)
        entries.append(entry)
    return entries


def extract_from_file(path: str | Path, **kw: Any) -> list[dict[str, Any]]:
    """Extract entries from every ATIF line of a trajectories.jsonl file."""
    path = Path(path)
    if path.is_dir():
        path = path / "trajectories.jsonl"
    entries: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            atif = json.loads(line)
        except json.JSONDecodeError:
            _log.warning("%s: skipping malformed line", path)
            continue
        entries.extend(extract_from_atif(atif, run_id=str(path.parent), **kw))
    return entries


def write_back(entries: Iterable[dict[str, Any]], root: str | Path,
               *, phase: str = "trajectory") -> int:
    """Append extracted entries to the knowledge store's ``<phase>.jsonl``.

    Returns the number of entries written.  The store layout is the one the
    fix-loop already replays from (``trajectories/<phase>.jsonl``), so distilled
    knowledge is immediately visible to the next failing run.
    """
    store = TrajectoryStore(Path(root) / "trajectories" if (Path(root) / "trajectories").is_dir()
                            else Path(root))
    written = 0
    for entry in entries:
        summary = TrajectorySummary(
            signature=entry.get("signature", ""),
            error_class=entry.get("error_class", ""),
            exc_type=entry.get("exc_type", ""),
            outcome=entry.get("outcome", "fixed"),
            root_cause=entry.get("root_cause", ""),
            fix_rationale=entry.get("fix_rationale", ""),
            component=entry.get("component", ""),
            smoke_test=entry.get("smoke_test", ""),
            tried_and_rejected=entry.get("tried_and_rejected", []),
            tier=entry.get("tier", "distill"),
            steps=entry.get("steps", 0),
            model_id=entry.get("model_id", "unknown"),
            scheme=entry.get("scheme", ""),
            method=entry.get("method", ""),
            run_id=entry.get("run_id", ""),
            phase=phase,
        )
        store.save(phase, summary)
        written += 1
    return written

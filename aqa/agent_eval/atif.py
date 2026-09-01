"""ATIF (Agent Trajectory Interchange Format) export for agent_eval runs.

Converts aqa's ReAct :class:`~aqa.agent_eval.react.Trajectory` into an
ATIF-v1.7 dict (Harbor's interchange format -- see
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md),
so aqa runs can be consumed by the Harbor toolchain and by SFT dataset builders
(:mod:`aqa.agent.traces_export`).  Runs dump one ATIF dict per line to
``trajectories.jsonl`` next to the run results (:func:`dump_trajectories_jsonl`).

Field mapping: the task prompt becomes the leading ``user`` step (the aqa
Trajectory does not retain it, so callers thread it through); each
:class:`ReActStep` becomes an ``agent`` step whose bash action maps to a
``bash`` tool call and whose observation maps to an ``ObservationResult``
correlated by ``tool_call_id``; per-step metrics are unavailable (TokenStats is
trajectory-cumulative) and therefore omitted, with trajectory totals going to
``final_metrics``.  Scaffold (codex/swe-agent) summary trajectories export what
exists: one agent step with the output tail + exit-code observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_util import get_logger
from .react import Command

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from .loop import AgentEvalResult
    from .react import Trajectory
    from .solver import Task

_log = get_logger("agent_eval.atif")

SCHEMA_VERSION = "ATIF-v1.7"
_SOURCES = ("user", "agent", "system")


def trajectory_to_atif(
    traj: Trajectory,
    *,
    agent_name: str = "aqa-react",
    model_name: str = "",
    system_prompt: str = "",
    task_prompt: str = "",
) -> dict[str, Any]:
    """Convert one ReAct :class:`Trajectory` into an ATIF-v1.7 dict."""
    from .. import __version__

    steps: list[dict[str, Any]] = []
    step_id = 1  # ATIF step_ids are 1-based ordinals
    if system_prompt:
        steps.append({"step_id": step_id, "source": "system", "message": system_prompt})
        step_id += 1
    steps.append({
        "step_id": step_id,
        "source": "user",
        # The Trajectory does not retain the prompt; the caller threads it
        # through from the Task.  Fall back to a traceable placeholder.
        "message": task_prompt or f"(task prompt unavailable; task_id={traj.task_id})",
    })
    step_id += 1

    for react_step in traj.steps:
        step: dict[str, Any] = {
            "step_id": step_id,
            "source": "agent",
            "message": react_step.output,
            "extra": {"action": "bash" if isinstance(react_step.action, Command) else "finish"},
        }
        step_id += 1
        if model_name:
            step["model_name"] = model_name
        call_id = None
        if isinstance(react_step.action, Command):
            call_id = f"call_{step['step_id']}"
            step["tool_calls"] = [{
                "tool_call_id": call_id,
                "function_name": "bash",
                "arguments": {"command": react_step.action.cmd},
            }]
        if react_step.observation is not None:
            result: dict[str, Any] = {"content": react_step.observation}
            if call_id is not None:
                result["source_call_id"] = call_id
            step["observation"] = {"results": [result]}
        steps.append(step)

    # Closing agent step carrying the clean final answer (the Finish step's raw
    # output still contains the ``Final Answer:`` prefix line).
    if traj.final_answer:
        step = {"step_id": step_id, "source": "agent", "message": traj.final_answer,
                "extra": {"action": "final_answer"}}
        if model_name:
            step["model_name"] = model_name
        steps.append(step)

    atif: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": str(traj.task_id),
        "agent": {
            "name": agent_name,
            "version": __version__,
            **({"model_name": model_name} if model_name else {}),
        },
        "steps": steps,
        "extra": {
            "task_id": traj.task_id,
            "reward": traj.reward,
            "solved": traj.solved,
            "hit_turn_limit": traj.hit_turn_limit,
            "turns": traj.turns,
            **({"verdict_category": traj.verdict.category} if traj.verdict else {}),
        },
    }
    stats = traj.stats
    if not stats.is_empty():
        final_metrics: dict[str, Any] = {
            "total_prompt_tokens": stats.n_input_tokens,
            "total_completion_tokens": stats.n_output_tokens,
            "total_cached_tokens": stats.n_cache_tokens,
        }
        if stats.cost_usd is not None:
            final_metrics["total_cost_usd"] = stats.cost_usd
        atif["final_metrics"] = final_metrics
    # aqa 自身 ReAct 轨迹与外部 backend 适配器共用同一脱敏规则
    from ..agent.redact import redact_atif

    return redact_atif(atif)


def validate_atif(data: dict[str, Any]) -> list[str]:
    """Light self-check of an ATIF dict; returns a list of problems ([] = ok).

    Not a full schema validator (Harbor's pydantic models are that) -- just the
    invariants aqa relies on: required keys, the source enum, monotonic step_ids,
    and tool_call/observation correlation.
    """
    errors: list[str] = []
    version = data.get("schema_version")
    if not (isinstance(version, str) and version.startswith("ATIF-")):
        errors.append("schema_version must be an 'ATIF-*' string")
    agent = data.get("agent")
    if not (isinstance(agent, dict) and agent.get("name") and agent.get("version")):
        errors.append("agent must be an object with name + version")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty array")
        return errors
    prev_id = 0
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] must be an object")
            continue
        sid = step.get("step_id")
        if not isinstance(sid, int) or sid <= prev_id:
            errors.append(f"steps[{i}].step_id must be a strictly increasing integer")
        else:
            prev_id = sid
        if step.get("source") not in _SOURCES:
            errors.append(f"steps[{i}].source must be one of {_SOURCES}")
        if not isinstance(step.get("message"), (str, list)):
            errors.append(f"steps[{i}].message must be a string (or content parts)")
        call_ids = set()
        for tc in step.get("tool_calls") or []:
            if not all(k in tc for k in ("tool_call_id", "function_name", "arguments")):
                errors.append(f"steps[{i}].tool_calls entries need tool_call_id/"
                              "function_name/arguments")
            else:
                call_ids.add(tc["tool_call_id"])
        for res in (step.get("observation") or {}).get("results", []):
            if "content" not in res:
                errors.append(f"steps[{i}].observation results need content")
            scid = res.get("source_call_id")
            if scid is not None and scid not in call_ids:
                errors.append(f"steps[{i}].observation source_call_id {scid!r} has no "
                              "matching tool_call")
    return errors


def dump_trajectories_jsonl(
    result: AgentEvalResult,
    out_dir: str | Path,
    *,
    tasks: Iterable[Task] = (),
    agent_name: str = "aqa-react",
    model_name: str = "",
    system_prompt: str = "",
) -> Path | None:
    """Write ``<out_dir>/trajectories.jsonl`` (one ATIF dict per line).

    Returns the written path, or ``None`` when the result carries no
    trajectories (e.g. the single-turn solver path).  Task prompts are looked
    up by task id so the exported user step carries the real instruction.
    """
    trajs = getattr(result, "trajectories", None) or []
    if not trajs:
        return None
    prompts = {t.id: t.prompt for t in tasks}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trajectories.jsonl"
    lines = [
        json.dumps(
            trajectory_to_atif(traj, agent_name=agent_name, model_name=model_name,
                               system_prompt=system_prompt,
                               task_prompt=prompts.get(traj.task_id, "")),
            ensure_ascii=False,
        )
        for traj in trajs
    ]
    path.write_text("\n".join(lines) + "\n")
    return path

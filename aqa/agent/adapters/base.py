"""Multi-backend agent session -> ATIF adapters.

Each backend's raw session log (jsonl) is converted into an ATIF dict -- the
same interchange format aqa's own ReAct trajectories use
(:mod:`aqa.agent_eval.atif`), so downstream dataset builders
(:mod:`aqa.agent.traces_export`) work uniformly across agent backends.

One adapter per backend, modeled on Harbor's per-backend ``_convert_*_to_atif``
converters (harbor-framework/harbor, ``src/harbor/agents/installed/*.py``);
the format contract for each backend is documented in
``docs/trajectory_data_plan.md``.  Adapters are deliberately lossy where the
source is (e.g. openclaw logs carry no model name; pass it via ``model_name``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from ...logging_util import get_logger

_log = get_logger("agent.adapters")

SCHEMA_VERSION = "ATIF-v1.7"


class Adapter(Protocol):
    """A backend adapter: detect a raw session artifact and convert it to ATIF."""

    name: str

    def detect(self, path: Path) -> bool:
        """True when ``path`` looks like this backend's session artifact."""
        ...

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        """Convert the session at ``path`` to an ATIF dict (None = unusable)."""
        ...


# ── shared builders ──────────────────────────────────────────────────────────
def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a jsonl file tolerantly (skip blank/malformed lines)."""
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            _log.debug("%s: skipping non-JSON line", path.name)
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def flatten_content(content: Any) -> str:
    """Normalize content (str | parts list | None) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
        )
    return ""


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments may be a dict or a JSON string (or neither)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"input": raw}
    return {"input": raw} if raw is not None else {}


def make_trajectory(*, session_id: str, agent_name: str, steps: list[dict[str, Any]],
                    model_name: str = "", agent_version: str = "",
                    final_metrics: dict[str, Any] | None = None,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the ATIF root dict (steps renumbered 1..N)."""
    for i, step in enumerate(steps, start=1):
        step["step_id"] = i
    traj: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version or "unknown",
            **({"model_name": model_name} if model_name else {}),
        },
        "steps": steps,
    }
    if final_metrics:
        traj["final_metrics"] = final_metrics
    if extra:
        traj["extra"] = extra
    return traj


class StepBuilder:
    """Accumulates ATIF steps with tool_call/observation correlation.

    ``add_agent`` starts an agent step carrying text + tool_calls;
    ``add_tool_result`` appends an ObservationResult onto the step that issued
    the call (unmatched results become standalone agent steps, mirroring
    Harbor's orphan handling).
    """

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}  # call_id -> step

    def add_user(self, text: str, **kw: Any) -> dict[str, Any]:
        step = {"source": "user", "message": text, **kw}
        self.steps.append(step)
        return step

    def add_system(self, text: str, **kw: Any) -> dict[str, Any]:
        step = {"source": "system", "message": text, **kw}
        self.steps.append(step)
        return step

    def add_agent(self, text: str, tool_calls: list[dict[str, Any]] | None = None,
                  **kw: Any) -> dict[str, Any]:
        step: dict[str, Any] = {"source": "agent", "message": text,
                                "llm_call_count": 1, **kw}
        if tool_calls:
            step["tool_calls"] = tool_calls
            step["observation"] = {"results": []}
            for tc in tool_calls:
                self._pending[tc["tool_call_id"]] = step
        self.steps.append(step)
        return step

    def add_tool_result(self, call_id: str, content: str,
                        extra: dict[str, Any] | None = None) -> bool:
        """Attach a tool result to its issuing step; False when orphaned."""
        step = self._pending.pop(call_id, None)
        if step is None:
            return False
        result: dict[str, Any] = {"source_call_id": call_id, "content": content}
        if extra:
            result["extra"] = extra
        step.setdefault("observation", {"results": []})["results"].append(result)
        return True


def first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def env_or(default: str, *names: str) -> str:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default

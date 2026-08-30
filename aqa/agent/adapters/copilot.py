"""Copilot CLI session adapter: copilot-cli.jsonl -> ATIF.

Source format (harbor copilot_cli.py contract): jsonl of copilot
``--output-format=json`` events, two coexisting schemas:

* flat (Anthropic models): ``type`` in message / tool_use / tool_result /
  usage, with role/content/id/name/input/tool_use_id;
* session-event (OpenAI models): namespaced ``type`` (``user.message`` /
  ``assistant.message`` / ``tool.execution_complete`` / ``result``) with the
  payload under ``data``.

Session id is not carried in the log -- Harbor hardcodes "copilot-cli".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (StepBuilder, env_or, flatten_content, iter_jsonl,
                   make_trajectory, parse_arguments)

_RESULT_KEYS = ("content", "output", "stdout", "text", "message")


def _result_text(data: dict[str, Any]) -> str:
    result = data.get("result")
    if isinstance(result, dict):
        for key in _RESULT_KEYS:
            if result.get(key):
                return str(result[key])
        return json_dumps(result)
    if result is not None:
        return str(result)
    if data.get("error"):
        return str(data["error"])
    return ""


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


class CopilotAdapter:
    name = "copilot"

    def detect(self, path: Path) -> bool:
        return "copilot" in path.name and path.suffix == ".jsonl"

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        events = iter_jsonl(path)
        sb = StepBuilder()
        total_in = total_out = 0
        model_seen = model_name
        first_user = True
        for ev in events:
            etype = ev.get("type", "")
            ts = ev.get("timestamp") if isinstance(ev.get("timestamp"), str) else None
            tsk = {"timestamp": ts} if ts else {}

            if etype == "message":  # flat schema
                role = ev.get("role", "user")
                text = flatten_content(ev.get("content"))
                if role == "assistant":
                    kw: dict[str, Any] = dict(tsk)
                    if ev.get("model"):
                        kw["model_name"] = ev["model"]
                        model_seen = model_seen or ev["model"]
                    sb.add_agent(text, **kw)
                else:
                    if first_user and instruction:
                        text = instruction
                    first_user = False
                    sb.add_user(text, **tsk)
            elif etype == "tool_use":
                name = ev.get("name", "")
                call_id = ev.get("id", "")
                sb.add_agent(f"Executed {name}", [{
                    "tool_call_id": call_id, "function_name": name,
                    "arguments": parse_arguments(ev.get("input")),
                }], **tsk)
            elif etype == "tool_result":
                sb.add_tool_result(ev.get("tool_use_id", ""),
                                   flatten_content(ev.get("content")) or json_dumps(ev.get("content")),
                                   extra={k: v for k, v in ev.items()
                                          if k not in ("type", "tool_use_id", "content", "timestamp")})
            elif etype == "usage":
                total_in += ev.get("input_tokens", 0)
                total_out += ev.get("output_tokens", 0)

            elif etype == "user.message":  # session-event schema
                data = ev.get("data") or {}
                text = data.get("content", "")
                if first_user and instruction:
                    text = instruction
                first_user = False
                sb.add_user(text, **tsk)
            elif etype == "assistant.message":
                data = ev.get("data") or {}
                text = data.get("content", "")
                calls = [
                    {"tool_call_id": tc.get("toolCallId", ""),
                     "function_name": tc.get("name", ""),
                     "arguments": parse_arguments(tc.get("arguments"))}
                    for tc in data.get("toolRequests") or []
                ]
                kw = dict(tsk)
                if data.get("model"):
                    kw["model_name"] = data["model"]
                    model_seen = model_seen or data["model"]
                if data.get("outputTokens"):
                    total_out += data["outputTokens"]
                    kw["metrics"] = {"completion_tokens": data["outputTokens"]}
                if text or calls:
                    sb.add_agent(text, calls or None, **kw)
            elif etype == "tool.execution_complete":
                data = ev.get("data") or {}
                sb.add_tool_result(data.get("toolCallId", ""), _result_text(data),
                                   extra={k: v for k, v in data.items()
                                          if k not in ("toolCallId", "result")})
            # result / tool.execution_* / assistant.turn_* / session.* ignored

        if len(sb.steps) < 2:
            return None
        final = None
        if total_in or total_out:
            final = {"total_prompt_tokens": total_in or None,
                     "total_completion_tokens": total_out or None}
            final = {k: v for k, v in final.items() if v is not None}
        return make_trajectory(
            session_id="copilot-cli",  # logs carry no session id
            agent_name="copilot",
            model_name=model_seen or env_or("", "COPILOT_MODEL"),
            steps=sb.steps, final_metrics=final or None,
        )


ADAPTER = CopilotAdapter()

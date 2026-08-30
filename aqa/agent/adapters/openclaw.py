"""OpenClaw session adapter: openclaw.session.jsonl -> ATIF.

Source format (harbor openclaw.py contract): jsonl of
``{"type":"message","timestamp":...,"message":{role, content, ...}}`` rows.
Roles consumed: ``user`` / ``assistant`` / ``toolResult``.  Assistant content
is a parts array (``text`` / ``toolCall{id,name,arguments}``); toolResult rows
carry ``toolCallId`` + ``details.aggregated`` (fallback: text parts).  Usage
lives per assistant row (``usage.{input,output,cacheRead,cacheWrite}``).

OpenClaw logs carry no model name -- pass ``model_name`` (``provider/model``)
explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (SCHEMA_VERSION, StepBuilder, env_or, flatten_content,
                   iter_jsonl, make_trajectory, parse_arguments)


class OpenclawAdapter:
    name = "openclaw"

    def detect(self, path: Path) -> bool:
        return path.name.endswith(".jsonl") and ("openclaw" in path.name or "session" in path.name)

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        events = iter_jsonl(path)
        sb = StepBuilder()
        first_user = True
        total_input = total_output = total_cache_read = total_cache_write = 0
        for ev in events:
            if ev.get("type") != "message":
                continue
            msg = ev.get("message") or {}
            role = msg.get("role")
            ts = ev.get("timestamp") if isinstance(ev.get("timestamp"), str) else None
            if role == "user":
                text = flatten_content(msg.get("content"))
                if first_user and instruction:
                    text = instruction
                first_user = False
                sb.add_user(text, **({"timestamp": ts} if ts else {}))
            elif role == "assistant":
                parts = msg.get("content") or []
                text = "".join(p.get("text", "") for p in parts
                               if isinstance(p, dict) and p.get("type") == "text")
                if not text:
                    text = msg.get("errorMessage") or "(no assistant text)"
                calls = [
                    {"tool_call_id": p.get("id", ""), "function_name": p.get("name", ""),
                     "arguments": parse_arguments(p.get("arguments"))}
                    for p in parts if isinstance(p, dict) and p.get("type") == "toolCall"
                ]
                usage = msg.get("usage") or {}
                metrics = None
                if usage:
                    total_input += usage.get("input", 0) + usage.get("cacheRead", 0)
                    total_output += usage.get("output", 0)
                    total_cache_read += usage.get("cacheRead", 0)
                    total_cache_write += usage.get("cacheWrite", 0)
                    metrics = {
                        "prompt_tokens": usage.get("input", 0) + usage.get("cacheRead", 0),
                        "completion_tokens": usage.get("output", 0),
                        "cached_tokens": usage.get("cacheRead", 0),
                        "extra": {"cache_write_tokens": usage.get("cacheWrite", 0)},
                    }
                kw: dict[str, Any] = {"metrics": metrics} if metrics else {}
                if model_name:
                    kw["model_name"] = model_name
                if ts:
                    kw["timestamp"] = ts
                sb.add_agent(text, calls or None, **kw)
            elif role == "toolResult":
                call_id = msg.get("toolCallId", "")
                content = (msg.get("details") or {}).get("aggregated") \
                    or flatten_content(msg.get("content"))
                sb.add_tool_result(call_id, content)

        if len(sb.steps) < 2:
            return None
        final = None
        if total_input or total_output:
            final = {"total_prompt_tokens": total_input,
                     "total_completion_tokens": total_output,
                     "total_cached_tokens": total_cache_read,
                     "extra": {"cache_write_tokens": total_cache_write}}
        return make_trajectory(
            session_id=path.stem.replace(".session", ""),
            agent_name="openclaw",
            model_name=model_name or env_or("", "OPENCLAW_MODEL"),
            steps=sb.steps, final_metrics=final,
        )


ADAPTER = OpenclawAdapter()

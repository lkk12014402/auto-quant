"""Codex CLI session adapter: sessions/<Y>/<M>/<D>/rollout-*.jsonl -> ATIF.

Source format (harbor codex.py contract): jsonl of ``{"timestamp","type",
"payload"}`` events.  Consumed: ``session_meta`` (session id, cli_version),
``turn_context`` (model), ``response_item`` payloads of type ``message`` /
``reasoning`` / ``function_call`` / ``custom_tool_call`` /
``function_call_output`` / ``custom_tool_call_output``, and ``event_msg`` with
``token_count`` (per-call + total usage).  Tool outputs join their calls by
``call_id``; output text may be a JSON string carrying {"output","metadata"}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import (StepBuilder, flatten_content, iter_jsonl, make_trajectory,
                   parse_arguments)


def _output_text(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            inner = json.loads(raw)
            if isinstance(inner, dict) and "output" in inner:
                return str(inner["output"])
        except json.JSONDecodeError:
            pass
        return raw
    return str(raw)


class CodexAdapter:
    name = "codex"

    def detect(self, path: Path) -> bool:
        return path.name.startswith("rollout-") and path.suffix == ".jsonl"

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        events = iter_jsonl(path)
        sb = StepBuilder()
        session_id = ""
        cli_version = ""
        model_seen = model_name
        first_user = True
        pending_reasoning = ""
        pending_calls: dict[str, dict[str, Any]] = {}  # call_id -> call dict
        order: list[str] = []
        last_usage: dict[str, Any] | None = None
        total_usage: dict[str, Any] | None = None

        def flush_calls(text: str, reasoning: str, ts: str | None) -> None:
            calls = [pending_calls[k] for k in order]
            kw: dict[str, Any] = {"llm_call_count": 1}
            if ts:
                kw["timestamp"] = ts
            if model_seen:
                kw["model_name"] = model_seen
            if reasoning:
                kw["reasoning_content"] = reasoning
            if last_usage:
                kw["metrics"] = {
                    "prompt_tokens": last_usage.get("input_tokens", 0),
                    "completion_tokens": last_usage.get("output_tokens", 0),
                    "cached_tokens": last_usage.get("cached_input_tokens", 0),
                    "extra": {"reasoning_output_tokens": last_usage.get("reasoning_output_tokens", 0)},
                }
            step = sb.add_agent(text, calls or None, **kw)
            for call_id in order:
                call = pending_calls[call_id]
                if "_output" in call:
                    sb._pending.pop(call_id, None)
                    step.setdefault("observation", {"results": []})["results"].append({
                        "source_call_id": call_id,
                        "content": call["_output"],
                        **({"extra": call["_meta"]} if call.get("_meta") else {}),
                    })

        for ev in events:
            etype = ev.get("type")
            ts = ev.get("timestamp") if isinstance(ev.get("timestamp"), str) else None
            payload = ev.get("payload") or {}
            if etype == "session_meta":
                session_id = payload.get("id", "") or session_id
                cli_version = payload.get("cli_version", "") or cli_version
            elif etype == "turn_context":
                model_seen = model_seen or payload.get("model", "")
            elif etype == "response_item":
                ptype = payload.get("type")
                if ptype == "message":
                    role = payload.get("role", "user")
                    text = flatten_content(payload.get("content"))
                    if role == "assistant":
                        flush_calls(text, pending_reasoning, ts)
                        pending_reasoning = ""
                        pending_calls, order = {}, []
                    else:
                        if first_user and instruction:
                            text = instruction
                        first_user = False
                        sb.add_user(text, **({"timestamp": ts} if ts else {}))
                elif ptype == "reasoning":
                    summary = payload.get("summary")
                    if isinstance(summary, list):
                        pending_reasoning += "".join(
                            s.get("text", "") for s in summary if isinstance(s, dict))
                    elif isinstance(summary, str):
                        pending_reasoning += summary
                elif ptype in ("function_call", "custom_tool_call"):
                    call_id = payload.get("call_id", "")
                    raw = payload.get("arguments", payload.get("input"))
                    pending_calls[call_id] = {
                        "tool_call_id": call_id,
                        "function_name": payload.get("name", ""),
                        "arguments": parse_arguments(raw),
                    }
                    order.append(call_id)
                elif ptype in ("function_call_output", "custom_tool_call_output"):
                    call_id = payload.get("call_id", "")
                    out_text = _output_text(payload.get("output"))
                    if call_id in pending_calls:
                        pending_calls[call_id]["_output"] = out_text
                        try:
                            inner = json.loads(payload["output"]) if isinstance(payload.get("output"), str) else None
                            if isinstance(inner, dict) and inner.get("metadata"):
                                pending_calls[call_id]["_meta"] = inner["metadata"]
                        except json.JSONDecodeError:
                            pass
                    else:
                        sb.add_agent(f"Tool output (orphan call {call_id})",
                                     [{"tool_call_id": call_id, "function_name": "unknown",
                                       "arguments": {}}],
                                     **({"timestamp": ts} if ts else {}))
                        sb.add_tool_result(call_id, out_text)
            elif etype == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info") or {}
                last_usage = info.get("last_token_usage") or last_usage
                total_usage = info.get("total_token_usage") or total_usage

        # trailing open calls / reasoning
        if order or pending_reasoning:
            flush_calls("", pending_reasoning, None)

        if len(sb.steps) < 2:
            return None
        final = None
        if total_usage:
            final = {
                "total_prompt_tokens": total_usage.get("input_tokens"),
                "total_completion_tokens": total_usage.get("output_tokens"),
                "total_cached_tokens": total_usage.get("cached_input_tokens"),
            }
            final = {k: v for k, v in final.items() if v is not None}
        return make_trajectory(
            session_id=session_id or path.stem,
            agent_name="codex",
            agent_version=cli_version,
            model_name=model_seen,
            steps=sb.steps, final_metrics=final or None,
        )


ADAPTER = CodexAdapter()

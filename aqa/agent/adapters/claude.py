"""Claude Code session adapter: projects/<slug>/<uuid>.jsonl -> ATIF.

Source format (harbor claude_code.py contract): jsonl events with top-level
``type`` (``user``/``assistant`` consumed), ``message`` payload, ``sessionId``,
``version``.  Assistant messages aggregate streaming rows by ``message.id``;
content blocks: ``text`` / ``thinking`` (-> reasoning_content) / ``tool_use``
(``id``/``name``/``input``); user events may carry ``tool_result`` blocks
(correlated by ``tool_use_id``) plus a top-level ``toolUseResult``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (StepBuilder, flatten_content, iter_jsonl, make_trajectory,
                   parse_arguments)


def _format_tool_use_result(tur: dict[str, Any]) -> str:
    parts = []
    if tur.get("stdout"):
        parts.append(f"[stdout]\n{tur['stdout']}")
    if tur.get("stderr"):
        parts.append(f"[stderr]\n{tur['stderr']}")
    code = tur.get("exitCode", tur.get("exit_code"))
    if code is not None:
        parts.append(f"[exit_code] {code}")
    return "\n".join(parts)


class ClaudeAdapter:
    name = "claude"

    def detect(self, path: Path) -> bool:
        return path.suffix == ".jsonl" and ("projects" in str(path) or path.parent.parent.name == "projects")

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        events = iter_jsonl(path)
        sb = StepBuilder()
        session_id = ""
        cli_version = ""
        model_seen = model_name
        first_user = True
        # Aggregate streaming rows by message.id: msgid -> (step, usage)
        by_msgid: dict[str, dict[str, Any]] = {}

        for ev in events:
            etype = ev.get("type")
            ts = ev.get("timestamp") if isinstance(ev.get("timestamp"), str) else None
            session_id = session_id or ev.get("sessionId", "")
            cli_version = cli_version or ev.get("version", "")
            msg = ev.get("message") or {}

            if etype == "user":
                content = msg.get("content")
                blocks = content if isinstance(content, list) else []
                tool_results = [b for b in blocks if isinstance(b, dict)
                                and b.get("type") == "tool_result"]
                if tool_results:
                    tur_extra = _format_tool_use_result(ev.get("toolUseResult") or {})
                    for block in tool_results:
                        text = flatten_content(block.get("content"))
                        if tur_extra:
                            text = (text + "\n" + tur_extra) if text else tur_extra
                        sb.add_tool_result(block.get("tool_use_id", ""), text,
                                           extra={"is_error": block.get("is_error")})
                else:
                    text = flatten_content(content)
                    if text:
                        if first_user and instruction:
                            text = instruction
                        first_user = False
                        sb.add_user(text, **({"timestamp": ts} if ts else {}))
            elif etype == "assistant":
                msgid = msg.get("id", "")
                content = msg.get("content") or []
                text = "".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
                reasoning = "".join(
                    b.get("thinking", "") or b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") in ("thinking", "reasoning", "analysis")
                )
                calls = [
                    {"tool_call_id": b.get("id", ""), "function_name": b.get("name", ""),
                     "arguments": parse_arguments(b.get("input"))}
                    for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                model_seen = model_seen or msg.get("model", "")

                if msgid and msgid in by_msgid:
                    # streaming continuation: merge into the existing step
                    step = by_msgid[msgid]["step"]
                    if text:
                        step["message"] = (step["message"] + "\n" + text).strip()
                    if reasoning:
                        step["reasoning_content"] = (step.get("reasoning_content", "") + reasoning)
                    for tc in calls:
                        step.setdefault("tool_calls", []).append(tc)
                        sb._pending[tc["tool_call_id"]] = step
                    if msg.get("usage"):
                        by_msgid[msgid]["usage"] = msg["usage"]
                        step["metrics"] = _metrics(msg["usage"])
                    continue

                kw: dict[str, Any] = {"llm_call_count": 1}
                if ts:
                    kw["timestamp"] = ts
                if msg.get("model"):
                    kw["model_name"] = msg["model"]
                if reasoning:
                    kw["reasoning_content"] = reasoning
                if msg.get("usage"):
                    kw["metrics"] = _metrics(msg["usage"])
                step = sb.add_agent(text, calls or None, **kw)
                if msgid:
                    by_msgid[msgid] = {"step": step, "usage": msg.get("usage")}

        if len(sb.steps) < 2:
            return None
        return make_trajectory(
            session_id=session_id or path.stem,
            agent_name="claude-code",
            agent_version=cli_version,
            model_name=model_seen,
            steps=sb.steps,
        )


def _metrics(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": (usage.get("input_tokens", 0)
                          + usage.get("cache_read_input_tokens", 0)
                          + usage.get("cache_creation_input_tokens", 0)),
        "completion_tokens": usage.get("output_tokens", 0),
        "cached_tokens": usage.get("cache_read_input_tokens", 0),
    }


ADAPTER = ClaudeAdapter()

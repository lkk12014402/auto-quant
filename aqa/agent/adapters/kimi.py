"""Kimi Code CLI session adapter: ``.../session_<id>/agents/<agent>/wire.jsonl`` -> ATIF.

Source format (kimi-code wire protocol): jsonl events.  Consumed:
``metadata`` (protocol version), ``profile.bind`` (modelAlias -> model name),
``context.append_message`` (committed user messages; ``turn.prompt`` duplicates
them and is skipped), and ``context.append_loop_event`` sub-events:
``content.part`` (``think``/``text`` parts), ``tool.call`` (name + args dict),
``tool.result`` (correlated by ``toolCallId``, arriving between steps), and
``step.end`` (per-call token usage).  Each ``step.end`` flushes one ATIF agent
step; tool results attach to the pending calls of the step that issued them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...logging_util import get_logger
from .base import StepBuilder, flatten_content, iter_jsonl, make_trajectory

_log = get_logger("agent.adapters.kimi")


class KimiAdapter:
    name = "kimi"

    def detect(self, path: Path) -> bool:
        if path.name != "wire.jsonl":
            return False
        try:
            with open(path) as f:
                head = json.loads(f.readline())
        except Exception:  # noqa: BLE001 - detection must never raise
            return False
        return head.get("type") == "metadata" and "protocol_version" in head

    def to_atif(self, path: Path, *, model_name: str = "",
                instruction: str | None = None) -> dict[str, Any] | None:
        events = iter_jsonl(path)
        sb = StepBuilder()
        model_seen = model_name
        version = ""
        first_user = True

        cur_text: list[str] = []
        cur_think: list[str] = []
        cur_calls: list[dict[str, Any]] = []
        results: dict[str, str] = {}  # toolCallId -> output (arrives before step.end)
        total = {"prompt": 0, "completion": 0, "cached": 0}

        def flush(usage: dict[str, Any] | None) -> None:
            if not (cur_text or cur_think or cur_calls):
                return
            kw: dict[str, Any] = {}
            if model_seen:
                kw["model_name"] = model_seen
            if cur_think:
                kw["reasoning_content"] = "".join(cur_think)
            if usage:
                kw["metrics"] = {
                    "prompt_tokens": usage.get("inputOther", 0) + usage.get("inputCacheCreation", 0),
                    "completion_tokens": usage.get("output", 0),
                    "cached_tokens": usage.get("inputCacheRead", 0),
                }
            step = sb.add_agent("".join(cur_text), list(cur_calls) or None, **kw)
            for call in cur_calls:
                cid = call["tool_call_id"]
                if cid in results:
                    sb._pending.pop(cid, None)
                    step.setdefault("observation", {"results": []})["results"].append(
                        {"source_call_id": cid, "content": results[cid]})
            cur_text.clear()
            cur_think.clear()
            cur_calls.clear()
            results.clear()

        for ev in events:
            etype = ev.get("type")
            if etype == "metadata":
                version = f"wire-{ev.get('protocol_version', '')}" or version
            elif etype == "profile.bind":
                model_seen = model_seen or ev.get("modelAlias", "")
            elif etype == "context.append_message":
                msg = ev.get("message") or {}
                if msg.get("role") != "user":
                    continue
                text = flatten_content(msg.get("content"))
                if first_user and instruction:
                    text = instruction
                first_user = False
                sb.add_user(text)
            elif etype == "context.append_loop_event":
                e = ev.get("event") or {}
                kind = e.get("type")
                if kind == "content.part":
                    part = e.get("part") or {}
                    if part.get("type") == "think":
                        cur_think.append(part.get("think", ""))
                    elif part.get("type") == "text":
                        cur_text.append(part.get("text", ""))
                elif kind == "tool.call":
                    cur_calls.append({
                        "tool_call_id": e.get("toolCallId", ""),
                        "function_name": e.get("name", ""),
                        "arguments": e.get("args") if isinstance(e.get("args"), dict) else {"input": e.get("args")},
                    })
                elif kind == "tool.result":
                    result = e.get("result") or {}
                    content = result.get("output", result) if isinstance(result, dict) else str(result)
                    cid = e.get("toolCallId", "")
                    if any(c["tool_call_id"] == cid for c in cur_calls):
                        results[cid] = str(content)
                    else:
                        _log.debug("tool result for unknown call %s", cid)
                elif kind == "step.end":
                    usage = e.get("usage") or {}
                    flush(usage)
                    total["prompt"] += usage.get("inputOther", 0) + usage.get("inputCacheCreation", 0)
                    total["completion"] += usage.get("output", 0)
                    total["cached"] += usage.get("inputCacheRead", 0)

        flush(None)  # trailing open step

        if len(sb.steps) < 2:
            return None
        # session id from .../session_<id>/agents/<agent>/wire.jsonl；子代理
        # （agent-N）与 main 共用 session_<id>，归档目录会撞名，补上代理名
        session_id = path.stem
        for parent in path.parents:
            if parent.name.startswith("session_"):
                session_id = parent.name
                break
        if (path.parent.parent.name == "agents" and path.parent.name != "main"
                and session_id.startswith("session_")):
            session_id = f"{session_id}_{path.parent.name}"
        final = {
            "total_prompt_tokens": total["prompt"],
            "total_completion_tokens": total["completion"],
            "total_cached_tokens": total["cached"],
        }
        return make_trajectory(
            session_id=session_id,
            agent_name="kimi",
            agent_version=version or "unknown",
            model_name=model_seen,
            steps=sb.steps,
            final_metrics=final,
        )


ADAPTER = KimiAdapter()

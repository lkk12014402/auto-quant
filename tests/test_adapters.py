"""Backend adapter tests: synthetic session jsonl -> ATIF, validated.

Fixtures mirror each backend's documented log format (see
``docs/trajectory_data_plan.md`` and Harbor's per-backend converters).  No real
agent sessions exist on this machine; the fixtures cover the four core event
kinds (user / agent text / tool call / tool result) per backend.
"""

import json
from pathlib import Path

import pytest

from aqa.agent.adapters import ADAPTERS, convert, detect_backend
from aqa.agent_eval.atif import validate_atif
from aqa.logging_util import AqaError


def _write(tmp_path: Path, name: str, lines: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


# ── openclaw ─────────────────────────────────────────────────────────────────
def test_openclaw_adapter(tmp_path):
    path = _write(tmp_path, "openclaw.session.jsonl", [
        {"type": "message", "timestamp": "2026-08-29T01:00:00Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "Fix the bug"}]}},
        {"type": "message", "timestamp": "2026-08-29T01:00:01Z",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "Let me read."},
                                 {"type": "toolCall", "id": "call_1", "name": "read_file",
                                  "arguments": "{\"path\": \"app.py\"}"}],
                     "usage": {"input": 1200, "output": 40, "cacheRead": 800, "cacheWrite": 0}}},
        {"type": "message", "timestamp": "2026-08-29T01:00:02Z",
         "message": {"role": "toolResult", "toolCallId": "call_1",
                     "details": {"aggregated": "def main(): pass"}}},
        {"type": "message", "timestamp": "2026-08-29T01:00:04Z",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "Fixed."}],
                     "usage": {"input": 2100, "output": 25, "cacheRead": 1200, "cacheWrite": 100}}},
    ])
    atif = convert(path, model_name="nvidia/test-model")
    assert validate_atif(atif) == []
    assert atif["agent"]["name"] == "openclaw"
    assert atif["agent"]["model_name"] == "nvidia/test-model"
    assert atif["schema_version"] == "ATIF-v1.7"
    agent_steps = [s for s in atif["steps"] if s["source"] == "agent"]
    assert agent_steps[0]["tool_calls"][0]["function_name"] == "read_file"
    assert agent_steps[0]["tool_calls"][0]["arguments"] == {"path": "app.py"}
    obs = agent_steps[0]["observation"]["results"][0]
    assert obs["source_call_id"] == "call_1"
    assert obs["content"] == "def main(): pass"
    assert agent_steps[0]["metrics"]["prompt_tokens"] == 2000  # input + cacheRead
    assert atif["final_metrics"]["total_completion_tokens"] == 65


# ── copilot (flat schema) ────────────────────────────────────────────────────
def test_copilot_adapter_flat(tmp_path):
    path = _write(tmp_path, "copilot-cli.jsonl", [
        {"type": "message", "role": "user", "content": "Fix the bug",
         "timestamp": "2026-08-29T01:00:00Z"},
        {"type": "tool_use", "id": "toolu_01", "name": "bash",
         "input": {"command": "cat app.py"}, "model": "claude-sonnet-4.5",
         "timestamp": "2026-08-29T01:00:01Z"},
        {"type": "tool_result", "tool_use_id": "toolu_01",
         "content": "def main(): pass", "is_error": False,
         "timestamp": "2026-08-29T01:00:02Z"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "text", "text": "Fixed."}], "model": "claude-sonnet-4.5",
         "timestamp": "2026-08-29T01:00:03Z"},
        {"type": "usage", "input_tokens": 1500, "output_tokens": 60,
         "timestamp": "2026-08-29T01:00:04Z"},
    ])
    atif = convert(path)
    assert validate_atif(atif) == []
    assert atif["agent"]["name"] == "copilot"
    assert atif["agent"]["model_name"] == "claude-sonnet-4.5"
    tool_step = next(s for s in atif["steps"] if s.get("tool_calls"))
    assert tool_step["tool_calls"][0]["function_name"] == "bash"
    assert tool_step["observation"]["results"][0]["content"] == "def main(): pass"
    assert atif["final_metrics"]["total_prompt_tokens"] == 1500


# ── copilot (session-event schema) ───────────────────────────────────────────
def test_copilot_adapter_session_events(tmp_path):
    path = _write(tmp_path, "copilot-cli.jsonl", [
        {"type": "session.start",
         "data": {"sessionId": "04e43240-test", "copilotVersion": "1.0.80"},
         "timestamp": "2026-08-29T00:59:59Z"},
        {"type": "user.message", "data": {"content": "Fix the bug"},
         "timestamp": "2026-08-29T01:00:00Z"},
        {"type": "assistant.message",
         "data": {"content": "Reading the file.",
                  "toolRequests": [{"toolCallId": "tc_1", "name": "bash",
                                    "arguments": {"command": "cat app.py"}}],
                  "outputTokens": 42, "model": "gpt-5.2"},
         "timestamp": "2026-08-29T01:00:01Z"},
        {"type": "tool.execution_complete",
         "data": {"toolCallId": "tc_1",
                  "result": {"content": "def main(): pass", "exitCode": 0}, "success": True},
         "timestamp": "2026-08-29T01:00:02Z"},
        {"type": "result", "exitCode": 0, "timestamp": "2026-08-29T01:00:04Z"},
    ])
    atif = convert(path)
    assert validate_atif(atif) == []
    assert atif["session_id"] == "04e43240-test"
    assert atif["agent"]["version"] == "1.0.80"
    agent_step = next(s for s in atif["steps"] if s["source"] == "agent")
    assert agent_step["model_name"] == "gpt-5.2"
    assert agent_step["metrics"]["completion_tokens"] == 42
    assert agent_step["tool_calls"][0]["tool_call_id"] == "tc_1"
    assert agent_step["observation"]["results"][0]["content"] == "def main(): pass"


# ── claude code ──────────────────────────────────────────────────────────────
def test_claude_adapter(tmp_path):
    sessions = tmp_path / "projects" / "-app"
    sessions.mkdir(parents=True)
    path = _write(sessions, "7f3a9b2c.jsonl", [
        {"type": "user", "uuid": "u1", "timestamp": "2026-08-29T01:00:00Z",
         "sessionId": "sess-1", "version": "2.1.0",
         "message": {"role": "user", "content": "Fix the bug"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-08-29T01:00:01Z",
         "sessionId": "sess-1",
         "message": {"model": "claude-sonnet-4-5", "id": "msg_01", "role": "assistant",
                     "content": [
                         {"type": "thinking", "thinking": "Need to inspect first"},
                         {"type": "text", "text": "Let me read."},
                         {"type": "tool_use", "id": "toolu_01", "name": "Read",
                          "input": {"file_path": "/app/app.py"}}],
                     "usage": {"input_tokens": 120, "output_tokens": 35,
                               "cache_read_input_tokens": 1800,
                               "cache_creation_input_tokens": 50}}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-08-29T01:00:02Z",
         "sessionId": "sess-1",
         "toolUseResult": {"stdout": "def main(): pass", "stderr": "", "exitCode": 0},
         "message": {"role": "user",
                     "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                  "content": "def main(): pass", "is_error": False}]}},
        {"type": "assistant", "uuid": "a2", "timestamp": "2026-08-29T01:00:04Z",
         "sessionId": "sess-1",
         "message": {"model": "claude-sonnet-4-5", "id": "msg_02", "role": "assistant",
                     "content": [{"type": "text", "text": "Fixed."}],
                     "usage": {"input_tokens": 140, "output_tokens": 20,
                               "cache_read_input_tokens": 1900}}},
    ])
    atif = convert(path)
    assert validate_atif(atif) == []
    assert atif["agent"]["name"] == "claude-code"
    assert atif["agent"]["version"] == "2.1.0"
    assert atif["session_id"] == "sess-1"
    agent_step = next(s for s in atif["steps"] if s["source"] == "agent")
    assert agent_step["reasoning_content"] == "Need to inspect first"
    assert agent_step["tool_calls"][0]["arguments"] == {"file_path": "/app/app.py"}
    obs = agent_step["observation"]["results"][0]
    assert obs["source_call_id"] == "toolu_01"
    assert "def main(): pass" in obs["content"]
    assert agent_step["metrics"]["prompt_tokens"] == 120 + 1800 + 50
    assert agent_step["metrics"]["cached_tokens"] == 1800


# ── codex ────────────────────────────────────────────────────────────────────
def test_codex_adapter(tmp_path):
    day = tmp_path / "sessions" / "2026" / "08" / "29"
    day.mkdir(parents=True)
    path = _write(day, "rollout-2026-08-29T01-00-00-sess9.jsonl", [
        {"timestamp": "2026-08-29T01:00:00Z", "type": "session_meta",
         "payload": {"id": "sess-9", "cli_version": "0.118.0"}},
        {"timestamp": "2026-08-29T01:00:00Z", "type": "turn_context",
         "payload": {"turn_id": "turn-1", "model": "gpt-5.2"}},
        {"timestamp": "2026-08-29T01:00:00Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "Fix the bug"}]}},
        {"timestamp": "2026-08-29T01:00:01Z", "type": "response_item",
         "payload": {"type": "reasoning",
                     "summary": [{"type": "summary_text", "text": "read first"}]}},
        {"timestamp": "2026-08-29T01:00:01Z", "type": "response_item",
         "payload": {"type": "function_call", "call_id": "call_1", "name": "shell",
                     "arguments": "{\"command\": [\"cat\", \"app.py\"]}",
                     "status": "completed"}},
        {"timestamp": "2026-08-29T01:00:02Z", "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": "call_1",
                     "output": "{\"output\": \"def main(): pass\", \"metadata\": {\"exit_code\": 0}}"}},
        {"timestamp": "2026-08-29T01:00:03Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Fixed."}]}},
        {"timestamp": "2026-08-29T01:00:03Z", "type": "event_msg",
         "payload": {"type": "token_count",
                     "info": {"last_token_usage": {"input_tokens": 1500, "output_tokens": 60,
                                                   "cached_input_tokens": 1000},
                              "total_token_usage": {"input_tokens": 1500, "output_tokens": 60,
                                                    "cached_input_tokens": 1000}}}},
    ])
    atif = convert(path)
    assert validate_atif(atif) == []
    assert atif["agent"]["name"] == "codex"
    assert atif["agent"]["version"] == "0.118.0"
    assert atif["session_id"] == "sess-9"
    agent_step = next(s for s in atif["steps"] if s["source"] == "agent"
                      and s.get("tool_calls"))
    assert agent_step["tool_calls"][0]["function_name"] == "shell"
    assert agent_step["tool_calls"][0]["arguments"] == {"command": ["cat", "app.py"]}
    obs = agent_step["observation"]["results"][0]
    assert obs["content"] == "def main(): pass"
    assert obs["extra"] == {"exit_code": 0}
    assert atif["final_metrics"]["total_prompt_tokens"] == 1500


# ── detection / errors ───────────────────────────────────────────────────────
def test_detect_backend_by_filename(tmp_path):
    assert detect_backend(tmp_path / "openclaw.session.jsonl") == "openclaw"
    assert detect_backend(tmp_path / "copilot-cli.jsonl") == "copilot"
    assert detect_backend(tmp_path / "rollout-2026-01-01T00-00-00-x.jsonl") == "codex"
    assert detect_backend(tmp_path / "random.txt") is None


def test_convert_unknown_backend_raises(tmp_path):
    path = _write(tmp_path, "random.jsonl", [{"foo": 1}])
    with pytest.raises(AqaError, match="no adapter"):
        convert(path)


def test_convert_missing_file_raises(tmp_path):
    with pytest.raises(AqaError, match="not found"):
        convert(tmp_path / "nope.jsonl")


def test_adapter_registry_complete():
    assert set(ADAPTERS) == {"openclaw", "copilot", "claude", "codex", "kimi"}


# ── kimi ─────────────────────────────────────────────────────────────────────
def test_kimi_adapter(tmp_path):
    sess = tmp_path / "sessions" / "wd_x" / "session_abc123" / "agents" / "main"
    sess.mkdir(parents=True)
    path = _write(sess, "wire.jsonl", [
        {"type": "metadata", "protocol_version": "1.5", "created_at": 1},
        {"type": "profile.bind", "agentId": "main", "modelAlias": "kimi-code/k3"},
        {"type": "turn.prompt", "agentId": "main",
         "input": [{"type": "text", "text": "Fix the bug"}], "origin": {"kind": "user"}},
        {"type": "context.append_message", "agentId": "main",
         "message": {"role": "user", "content": [{"type": "text", "text": "Fix the bug"}]}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "step.begin", "uuid": "s1", "turnId": "0", "step": 1}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "content.part", "stepUuid": "s1",
                   "part": {"type": "think", "think": "read first"}}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "tool.call", "stepUuid": "s1", "toolCallId": "tool_1",
                   "name": "Bash", "args": {"command": "cat app.py"}}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "tool.result", "toolCallId": "tool_1",
                   "result": {"output": "def main(): pass"}}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "step.end", "uuid": "s1", "finishReason": "tool_use",
                   "usage": {"inputOther": 500, "output": 40,
                             "inputCacheRead": 1000, "inputCacheCreation": 0}}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "step.begin", "uuid": "s2", "turnId": "0", "step": 2}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "content.part", "stepUuid": "s2",
                   "part": {"type": "text", "text": "Fixed."}}},
        {"type": "context.append_loop_event", "agentId": "main",
         "event": {"type": "step.end", "uuid": "s2", "finishReason": "stop",
                   "usage": {"inputOther": 600, "output": 10,
                             "inputCacheRead": 1100, "inputCacheCreation": 0}}},
    ])
    assert detect_backend(path) == "kimi"
    atif = convert(path)
    assert validate_atif(atif) == []
    assert atif["agent"]["name"] == "kimi"
    assert atif["agent"]["model_name"] == "kimi-code/k3"
    assert atif["session_id"] == "session_abc123"
    steps = atif["steps"]
    # turn.prompt must not duplicate the committed user message
    assert [s["message"] for s in steps if s["source"] == "user"] == ["Fix the bug"]
    agent_step = next(s for s in steps if s["source"] == "agent" and s.get("tool_calls"))
    assert agent_step["reasoning_content"] == "read first"
    assert agent_step["tool_calls"][0] == {"tool_call_id": "tool_1", "function_name": "Bash",
                                           "arguments": {"command": "cat app.py"}}
    assert agent_step["observation"]["results"][0]["content"] == "def main(): pass"
    assert agent_step["metrics"]["prompt_tokens"] == 500
    assert agent_step["metrics"]["cached_tokens"] == 1000
    assert atif["final_metrics"]["total_completion_tokens"] == 50


def test_kimi_adapter_rejects_non_wire(tmp_path):
    path = _write(tmp_path, "wire.jsonl", [{"foo": 1}])
    assert detect_backend(path) is None
    with pytest.raises(AqaError, match="no adapter"):
        convert(path)

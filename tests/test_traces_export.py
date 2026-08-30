"""Tests for trajectory dataset assembly (aqa traces export)."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from aqa.agent import traces_export as te
from aqa.agent.knowledge import signature_hash
from aqa.agent.trajectories import TrajectoryStore, TrajectorySummary
from aqa.agent_eval.atif import trajectory_to_atif
from aqa.agent_eval.react import Command, Finish, ReActStep, Trajectory
from aqa.agent_eval.verifier import EvalVerdict
from aqa.cli import main
from aqa.logging_util import AqaError


def _traj(task_id, *, solved, reward):
    return Trajectory(
        task_id=task_id,
        steps=[
            ReActStep("inspect\n```bash\nls\n```", Command("ls"), "file1"),
            ReActStep(f"Final Answer: {task_id} done", Finish(f"{task_id} done")),
        ],
        final_answer=f"{task_id} done",
        solved=solved,
        reward=reward,
        verdict=EvalVerdict(reward=reward, solved=solved,
                            category="ok" if solved else "model_wrong"),
    )


def _write_run(run_dir, trajs):
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(trajectory_to_atif(t, model_name="m",
                                      task_prompt=f"prompt for {t.task_id}"),
                   ensure_ascii=False)
        for t in trajs
    ]
    (run_dir / "trajectories.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir


# ── agent_eval export ────────────────────────────────────────────────────────
def test_export_agent_eval_rows(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0),
                                         _traj("b", solved=False, reward=0.0)])
    rows = te.export_agent_eval([run], success_only=False)
    assert len(rows) == 2
    row = rows[0]
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "assistant"]
    # OpenAI tool-call shape on the assistant message
    call = row["messages"][1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls"}
    # observation mapped to a tool message correlated by tool_call_id
    tool_msg = row["messages"][2]
    assert tool_msg["tool_call_id"] == call["id"]
    assert tool_msg["content"] == "file1"
    assert row["metadata"] == {"task": "a", "model": "m", "reward": 1.0,
                              "solved": True, "run": str(run), "source": "agent_eval"}


def test_export_agent_eval_success_only_filters(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0),
                                         _traj("b", solved=False, reward=0.0)])
    rows = te.export_agent_eval([run], success_only=True)
    assert [r["metadata"]["task"] for r in rows] == ["a"]


def test_export_agent_eval_skips_empty_assistant(tmp_path):
    empty = Trajectory(task_id="empty", steps=[ReActStep("", Finish(""))])
    run = _write_run(tmp_path / "run1", [empty])
    assert te.export_agent_eval([run], success_only=False) == []


def test_sharegpt_column(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0)])
    rows = te.export_agent_eval([run])
    te.add_sharegpt(rows)
    conv = rows[0]["conversations"]
    assert conv[0] == {"from": "human", "value": "prompt for a"}
    assert {c["from"] for c in conv} == {"human", "gpt"}
    assert conv[1]["from"] == "gpt"
    assert conv[1]["value"] == rows[0]["messages"][1]["content"]


def test_write_jsonl_round_trip(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0)])
    rows = te.export_agent_eval([run])
    out = te.write_jsonl(rows, tmp_path / "traces.jsonl")
    back = [json.loads(x) for x in out.read_text().splitlines()]
    assert back == rows


# ── fix-loop export ──────────────────────────────────────────────────────────
def _knowledge_root(tmp_path, *, with_patch=True):
    root = tmp_path / "knowledge"
    store = TrajectoryStore(root / "trajectories")
    store.save("quantize", TrajectorySummary(
        signature="quantize.py:42 RuntimeError oom",
        error_class="oom",
        exc_type="RuntimeError",
        root_cause="vllm preallocates KV cache",
        fix_rationale="lower gpu_memory_utilization to 0.8",
        component="evaluate",
        smoke_test="run piqa on 1 sample",
        tried_and_rejected=["reduce batch size"],
        tier="openclaw",
        steps=3,
        model_id="org/m",
        scheme="MXFP4",
        method="model_free",
        run_id="r1",
    ))
    if with_patch:
        dest = root / "patches" / "quantize" / signature_hash("quantize.py:42 RuntimeError oom")
        dest.mkdir(parents=True)
        (dest / "meta.json").write_text(json.dumps({
            "signature": "quantize.py:42 RuntimeError oom", "error_class": "oom"}))
        (dest / "evaluate.patch").write_text(
            "--- a/aqa/eval/runner.py\n+++ b/aqa/eval/runner.py\n@@\n-gpu_memory_utilization=0.9\n+gpu_memory_utilization=0.8\n")
    return root


def test_export_fixloop_rows_with_patch(tmp_path):
    rows = te.export_fixloop(_knowledge_root(tmp_path))
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "quantization" in msgs[0]["content"]
    assert "quantize.py:42 RuntimeError oom" in msgs[1]["content"]
    assert "oom" in msgs[1]["content"]
    assistant = msgs[2]["content"]
    assert "vllm preallocates KV cache" in assistant
    assert "lower gpu_memory_utilization to 0.8" in assistant
    assert "Dead ends avoided:\n- reduce batch size" in assistant
    assert "```diff" in assistant and "gpu_memory_utilization=0.8" in assistant
    meta = rows[0]["metadata"]
    assert meta["signature"] == "quantize.py:42 RuntimeError oom"
    assert meta["tier"] == "openclaw"
    assert meta["phase"] == "quantize"
    assert meta["run_id"] == "r1"
    assert meta["source"] == "fixloop"


def test_export_fixloop_without_patch(tmp_path):
    rows = te.export_fixloop(_knowledge_root(tmp_path, with_patch=False))
    assert len(rows) == 1
    assert "```diff" not in rows[0]["messages"][2]["content"]


def test_detect_source(tmp_path):
    assert te.detect_source(_write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1)])) \
        == "agent_eval"
    assert te.detect_source(_knowledge_root(tmp_path)) == "fixloop"
    f = tmp_path / "run1" / "trajectories.jsonl"
    assert te.detect_source(f) == "agent_eval"
    (tmp_path / "empty").mkdir()
    assert te.detect_source(tmp_path / "empty") is None


# ── output formats / deps ────────────────────────────────────────────────────
def test_parquet_missing_datasets_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # force ImportError
    with pytest.raises(AqaError, match="datasets"):
        te.write_parquet([{"messages": []}], tmp_path / "x.parquet")


def test_push_requires_env_token(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AqaError, match="token"):
        te.push_to_hub([{"messages": []}], "org/ds")


def test_push_missing_datasets_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(AqaError, match="datasets"):
        te.push_to_hub([{"messages": []}], "org/ds")


# ── CLI smoke ────────────────────────────────────────────────────────────────
def test_cli_traces_export_agent_eval(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0),
                                         _traj("b", solved=False, reward=0.0)])
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(
        main, ["traces", "export", "--path", str(run), "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1  # default --filter success
    assert rows[0]["metadata"]["task"] == "a"
    assert "exported 1 row(s)" in result.output


def test_cli_traces_export_fixloop_with_sharegpt(tmp_path):
    root = _knowledge_root(tmp_path)
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(
        main, ["traces", "export", "--path", str(root), "--sharegpt",
               "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["conversations"][1]["from"] == "human"
    assert rows[0]["conversations"][2]["from"] == "gpt"


def test_cli_traces_export_no_rows(tmp_path):
    (tmp_path / "empty").mkdir()
    result = CliRunner().invoke(main, ["traces", "export", "--path", str(tmp_path / "empty")])
    assert result.exit_code == 0
    assert "no trajectory rows found" in result.output


# ── SFT view (Nemotron-Agentic shape) & RL view (NeMo-Gym shape) ────────────
def _atif_with_reasoning():
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "sess-1",
        "agent": {"name": "aqa-react", "version": "0.1", "model_name": "m",
                  "tool_definitions": [
                      {"type": "function",
                       "function": {"name": "bash", "description": "run shell",
                                    "parameters": {"type": "object", "properties": {}}}},
                  ]},
        "steps": [
            {"step_id": 1, "source": "user", "message": "fix the bug"},
            {"step_id": 2, "source": "agent", "message": "reading first",
             "reasoning_content": "need context",
             "tool_calls": [{"tool_call_id": "c1", "function_name": "bash",
                             "arguments": {"command": "cat app.py"}}],
             "observation": {"results": [{"source_call_id": "c1",
                                          "content": "def main(): pass"}]}},
            {"step_id": 3, "source": "agent", "message": "fixed"},
        ],
        "extra": {"task_id": "t1", "solved": True, "reward": 1.0},
    }


def test_sft_row_nemotron_shape():
    row = te.atif_to_sft_row(_atif_with_reasoning(), run="run1")
    assert row is not None
    assert set(row) == {"uuid", "messages", "tools", "metadata"}
    # reasoning_content is separate from content (never folded in)
    asst = row["messages"][1]
    assert asst["reasoning_content"] == "need context"
    assert asst["content"] == "reading first"
    assert asst["tool_calls"][0]["function"]["name"] == "bash"
    tool_msg = row["messages"][2]
    assert tool_msg == {"role": "tool", "tool_call_id": "c1",
                        "content": "def main(): pass"}
    assert row["tools"][0]["function"]["name"] == "bash"
    assert row["metadata"]["solved"] is True
    assert row["metadata"]["reward"] == 1.0


def test_sft_row_skips_empty_assistant():
    atif = _atif_with_reasoning()
    atif["steps"][1:] = []  # only the user step remains
    assert te.atif_to_sft_row(atif) is None


def test_rl_row_nemo_gym_shape():
    row = te.atif_to_rl_row(_atif_with_reasoning(), run="run1")
    assert row is not None
    rcp = row["responses_create_params"]
    # task input = everything before the first assistant turn (the user prompt)
    assert [m["role"] for m in rcp["input"]] == ["user"]
    assert rcp["input"][0]["content"] == "fix the bug"
    assert rcp["tools"][0]["function"]["name"] == "bash"
    assert rcp["parallel_tool_calls"] is True
    assert row["environment_name"] == "agent_run"
    assert row["agent_ref"] == {"type": "aqa", "name": "aqa-react"}
    # trajectory outcome rides in verifier materials (offline RL / rejection sampling)
    assert row["verifier"]["reward"] == 1.0
    assert row["verifier"]["solved"] is True
    assert row["metadata"]["model"] == "m"


def test_rl_row_requires_prompt():
    atif = _atif_with_reasoning()
    atif["steps"] = [s for s in atif["steps"] if s["source"] == "agent"]
    assert te.atif_to_rl_row(atif) is None


def test_export_agent_eval_view_rl(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0)])
    sft = te.export_agent_eval([run], view="sft")
    rl = te.export_agent_eval([run], view="rl")
    assert "messages" in sft[0] and "tools" in sft[0]
    assert "responses_create_params" in rl[0] and "verifier" in rl[0]
    assert rl[0]["verifier"]["reward"] == 1.0


def test_export_agent_eval_unknown_view_raises(tmp_path):
    run = _write_run(tmp_path / "run1", [_traj("a", solved=True, reward=1.0)])
    with pytest.raises(AqaError, match="unknown view"):
        te.export_agent_eval([run], view="banana")

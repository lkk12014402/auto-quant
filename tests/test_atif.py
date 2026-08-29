"""Tests for ATIF (Harbor trajectory format) conversion + the run-time dump hook."""

from __future__ import annotations

import asyncio
import json

from aqa.agent_eval import (
    Finish,
    ReActStep,
    Task,
    Trajectory,
    run_tool_use_eval,
)
from aqa.agent_eval.atif import (
    dump_trajectories_jsonl,
    trajectory_to_atif,
    validate_atif,
)
from aqa.agent_eval.loop import AgentEvalResult
from aqa.agent_eval.react import Command
from aqa.agent_eval.stats import TokenStats
from aqa.agent_eval.verifier import EvalVerdict


def _stats():
    stats = TokenStats()
    stats.add_usage({"prompt_tokens": 100, "completion_tokens": 40,
                     "prompt_tokens_details": {"cached_tokens": 30}})
    stats.cost_usd = 0.001
    return stats


def _traj():
    return Trajectory(
        task_id="task-1",
        steps=[
            ReActStep("let me look\n```bash\nls -la\n```", Command("ls -la"),
                      "total 3\nfile1"),
            ReActStep("fixed it\nFinal Answer: patched", Finish("patched")),
        ],
        final_answer="patched",
        solved=True,
        reward=1.0,
        verdict=EvalVerdict(reward=1.0, solved=True, category="ok"),
        stats=_stats(),
    )


def test_trajectory_to_atif_shape_and_validity():
    atif = trajectory_to_atif(_traj(), model_name="m", system_prompt="sys",
                              task_prompt="Fix the bug")
    assert validate_atif(atif) == []
    assert atif["schema_version"] == "ATIF-v1.7"
    assert atif["session_id"] == "task-1"
    assert atif["agent"]["name"] == "aqa-react"
    assert atif["agent"]["model_name"] == "m"
    assert atif["agent"]["version"]

    steps = atif["steps"]
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent", "agent"]
    assert [s["step_id"] for s in steps] == [1, 2, 3, 4, 5]  # 1-based, monotonic
    assert steps[0]["message"] == "sys"
    assert steps[1]["message"] == "Fix the bug"  # task prompt threaded through


def test_tool_call_and_observation_matching():
    atif = trajectory_to_atif(_traj(), task_prompt="p")
    # no system prompt: steps = [user, agent(bash), agent(finish), closing]
    step = atif["steps"][1]
    call = step["tool_calls"][0]
    assert call["function_name"] == "bash"
    assert call["arguments"] == {"command": "ls -la"}
    result = step["observation"]["results"][0]
    assert result["source_call_id"] == call["tool_call_id"]
    assert result["content"] == "total 3\nfile1"
    # the Finish step carries no tool call; the closing step has the clean answer
    assert "tool_calls" not in atif["steps"][2]
    assert atif["steps"][-1]["message"] == "patched"
    assert atif["steps"][-1]["extra"]["action"] == "final_answer"


def test_final_metrics_and_extra_mapping():
    atif = trajectory_to_atif(_traj(), task_prompt="p")
    fm = atif["final_metrics"]
    assert fm["total_prompt_tokens"] == 100
    assert fm["total_completion_tokens"] == 40
    assert fm["total_cached_tokens"] == 30
    assert fm["total_cost_usd"] == 0.001
    extra = atif["extra"]
    assert extra["reward"] == 1.0
    assert extra["solved"] is True
    assert extra["verdict_category"] == "ok"
    assert extra["task_id"] == "task-1"


def test_empty_stats_omit_final_metrics():
    traj = Trajectory(task_id="t", steps=[ReActStep("a", Finish("a"))], final_answer="a")
    atif = trajectory_to_atif(traj, task_prompt="p")
    assert "final_metrics" not in atif
    assert validate_atif(atif) == []


def test_prompt_placeholder_when_unavailable():
    atif = trajectory_to_atif(_traj())
    assert "task-1" in atif["steps"][0]["message"]  # traceable placeholder


def test_scaffold_summary_trajectory_exports_what_exists():
    # codex/swe-agent scaffolds record one output-tail step + exit observation
    traj = Trajectory(
        task_id="swebench-x",
        steps=[ReActStep("...cli tail...", Finish(""), observation="exit=0")],
        solved=False,
        reward=0.0,
        verdict=EvalVerdict(reward=0.0, solved=False, category="model_wrong"),
    )
    atif = trajectory_to_atif(traj, agent_name="codex", task_prompt="issue text")
    assert validate_atif(atif) == []
    agent_step = atif["steps"][1]
    assert "tool_calls" not in agent_step
    # observation without a tool call has no source_call_id (allowed by ATIF)
    assert agent_step["observation"]["results"][0] == {"content": "exit=0"}
    assert atif["agent"]["name"] == "codex"


def test_validate_atif_catches_errors():
    good = trajectory_to_atif(_traj(), task_prompt="p")
    bad_source = json.loads(json.dumps(good))
    bad_source["steps"][1]["source"] = "robot"
    assert any("source" in e for e in validate_atif(bad_source))

    bad_ids = json.loads(json.dumps(good))
    bad_ids["steps"][2]["step_id"] = 1  # not monotonic
    assert any("step_id" in e for e in validate_atif(bad_ids))

    orphan = json.loads(json.dumps(good))
    orphan["steps"][1]["observation"]["results"][0]["source_call_id"] = "nope"
    assert any("source_call_id" in e for e in validate_atif(orphan))

    assert validate_atif({"schema_version": "ATIF-v1.7"})  # no agent/steps


class _FakeAgent:
    name = "aqa-react"

    def __init__(self, traj):
        self._traj = traj

    async def run(self, task):
        return self._traj


def test_run_tool_use_eval_keeps_trajectories():
    traj = _traj()
    result = asyncio.run(run_tool_use_eval(
        "t", [Task(id="task-1", prompt="Fix the bug")], _FakeAgent(traj)))
    assert result.trajectories == [traj]
    # to_dict stays a compact score report (no trajectory payloads)
    assert "trajectories" not in result.to_dict()


def test_dump_trajectories_jsonl(tmp_path):
    traj = _traj()
    result = asyncio.run(run_tool_use_eval(
        "t", [Task(id="task-1", prompt="Fix the bug")], _FakeAgent(traj)))
    path = dump_trajectories_jsonl(result, tmp_path,
                                   tasks=[Task(id="task-1", prompt="Fix the bug")],
                                   model_name="m", system_prompt="sys")
    assert path == tmp_path / "trajectories.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    atif = json.loads(lines[0])
    assert validate_atif(atif) == []
    assert atif["steps"][1]["message"] == "Fix the bug"


def test_dump_returns_none_without_trajectories(tmp_path):
    assert dump_trajectories_jsonl(AgentEvalResult(name="x"), tmp_path) is None
    assert not (tmp_path / "trajectories.jsonl").exists()


def test_swebench_run_dumps_trajectories(tmp_path):
    """The swebench hook writes trajectories.jsonl next to report.json."""
    from test_swebench import FakeSandbox, ScriptedClient, _inst

    from aqa.swebench import run_swebench

    insts = [_inst("a__a-1")]
    logs = {"a__a-1": "PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep"}

    def factory(inst):
        return FakeSandbox(patch=f"diff {inst.instance_id}\n", test_log=logs[inst.instance_id])

    run_swebench(insts, ScriptedClient(["```bash\ngit status\n```", "Final Answer: done"]),
                 sandbox_factory=factory, spec_provider=lambda i: None,
                 model_name="m1", out_dir=tmp_path, concurrency=1, max_turns=3)
    lines = (tmp_path / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 1
    atif = json.loads(lines[0])
    assert validate_atif(atif) == []
    assert atif["steps"][1]["message"]  # the issue instruction as user step
    assert atif["agent"]["model_name"] == "m1"

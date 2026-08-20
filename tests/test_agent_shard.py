"""Tests for E1 sharded agentic eval (round-robin across endpoint pool)."""

from __future__ import annotations

import asyncio

from aqa.agent_eval import ModelClient, Task, run_sharded_eval, split_gpu_shards
from aqa.agent_eval.solver import ChatSolver


def _tagging_transport(tag: str):
    async def transport(url, payload, headers):
        user = payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": f"{tag}:{user}"}}]}

    return transport


def _solver(tag: str) -> ChatSolver:
    return ChatSolver(client=ModelClient(base_url="http://h", model="m",
                                         transport=_tagging_transport(tag)))


def test_split_gpu_shards():
    assert split_gpu_shards([0, 1, 2, 3], 2) == [[0, 1], [2, 3]]
    assert split_gpu_shards([0, 1, 2, 3, 4], 2) == [[0, 1], [2, 3], [4]]


def test_sharded_eval_round_robin_and_order():
    tasks = [Task(id=str(i), prompt=str(i)) for i in range(6)]
    solvers = [_solver("s0"), _solver("s1")]

    # score = shard index the task was routed to (parsed from the tag).
    def score(task, response):
        return 0.0 if response.startswith("s0") else 1.0

    result = asyncio.run(run_sharded_eval("t", tasks, solvers, score,
                                          concurrency_per_shard=2))
    # even indices -> shard 0 (s0 -> 0.0), odd -> shard 1 (s1 -> 1.0)
    assert result.per_sample() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert result.num_tasks == 6
    assert [o.task_id for o in result.outcomes] == ["0", "1", "2", "3", "4", "5"]


def test_sharded_eval_single_solver_equivalent():
    tasks = [Task(id=str(i), prompt=str(i), target="x") for i in range(4)]
    result = asyncio.run(run_sharded_eval("t", tasks, [_solver("s")],
                                          lambda t, r: 1.0))
    assert result.mean_score() == 1.0
    assert result.num_tasks == 4


def test_sharded_eval_requires_solver():
    try:
        asyncio.run(run_sharded_eval("t", [], [], lambda t, r: 1.0))
    except Exception as exc:  # noqa: BLE001
        assert "at least one solver" in str(exc)
    else:
        raise AssertionError("expected AqaError for empty solver pool")

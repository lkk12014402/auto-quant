"""Async evaluation loop for endpoint-centric (agentic) benchmarks.

Ports the shape of NeMo Evaluator's async ``eval_loop``: drive a set of tasks
through a :class:`~aqa.agent_eval.solver.Solver` against a served endpoint, score
each response, and aggregate -- all with bounded concurrency.  Scoring reuses the
native scorer library so agentic and lm_eval paths share one metric vocabulary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..logging_util import get_logger
from .solver import Solver, Task

_log = get_logger("agent_eval.loop")

# score_fn(task, response) -> correctness in [0, 1].
ScoreFn = Callable[[Task, str], float]


@dataclass
class TaskOutcome:
    task_id: str
    response: str
    score: float
    error: str | None = None


@dataclass
class AgentEvalResult:
    name: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def num_tasks(self) -> int:
        return len(self.outcomes)

    def mean_score(self) -> float:
        scored = [o.score for o in self.outcomes if o.error is None]
        return sum(scored) / len(scored) if scored else 0.0

    def per_sample(self) -> list[float]:
        return [o.score for o in self.outcomes if o.error is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_tasks": self.num_tasks,
            "mean_score": self.mean_score(),
            "num_errors": sum(1 for o in self.outcomes if o.error),
            "outcomes": [
                {"task_id": o.task_id, "score": o.score, "error": o.error}
                for o in self.outcomes
            ],
        }


async def run_agentic_eval(
    name: str,
    tasks: list[Task],
    solver: Solver,
    score_fn: ScoreFn,
    *,
    concurrency: int = 8,
) -> AgentEvalResult:
    """Solve + score every task with bounded concurrency; preserves task order."""
    sem = asyncio.Semaphore(max(1, concurrency))
    outcomes: list[TaskOutcome | None] = [None] * len(tasks)

    async def _one(idx: int, task: Task) -> None:
        async with sem:
            try:
                response = await solver.solve(task)
            except Exception as exc:  # noqa: BLE001 - one task failing must not abort the run
                _log.warning("task %s failed: %s", task.id, exc)
                outcomes[idx] = TaskOutcome(task.id, "", 0.0, error=str(exc))
                return
            try:
                score = float(score_fn(task, response))
            except Exception as exc:  # noqa: BLE001 - scoring error is per-task
                outcomes[idx] = TaskOutcome(task.id, response, 0.0, error=f"score: {exc}")
                return
            outcomes[idx] = TaskOutcome(task.id, response, score)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(tasks)))
    result = AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])
    _log.info("agentic eval %s: %d tasks, mean=%.4f, errors=%d",
              name, result.num_tasks, result.mean_score(),
              sum(1 for o in result.outcomes if o.error))
    return result

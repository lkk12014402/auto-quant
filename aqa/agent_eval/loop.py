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
from .stats import TokenStats
from .verifier import EvalVerdict, coerce_verdict, verdict_from_exception

_log = get_logger("agent_eval.loop")

# score_fn(task, response) -> correctness in [0, 1].
ScoreFn = Callable[[Task, str], float]


@dataclass
class TaskOutcome:
    task_id: str
    response: str
    score: float
    error: str | None = None
    verdict: EvalVerdict | None = None
    stats: TokenStats | None = None
    turns: int = 0

    @property
    def is_scored(self) -> bool:
        """Counts toward mean/denominator: no hard error and a scored verdict.

        Legacy outcomes carry no verdict; those count as scored (unchanged
        behaviour) as long as they have no ``error``.
        """
        if self.error is not None:
            return False
        return self.verdict is None or self.verdict.is_scored


@dataclass
class AgentEvalResult:
    name: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    # Full ReAct trajectories of the tool-use path, kept for ATIF/SFT export
    # (``aqa.agent_eval.atif``).  Typed Any to avoid a loop<->react import
    # cycle; excluded from to_dict (too large for the score report).
    trajectories: list[Any] = field(default_factory=list)

    @property
    def num_tasks(self) -> int:
        return len(self.outcomes)

    def mean_score(self) -> float:
        scored = [o.score for o in self.outcomes if o.is_scored]
        return sum(scored) / len(scored) if scored else 0.0

    def per_sample(self) -> list[float]:
        return [o.score for o in self.outcomes if o.is_scored]

    def by_category(self) -> dict[str, int]:
        """Count outcomes by verdict category (infra/parse failures visible).

        Errored outcomes without a verdict are bucketed as ``infra_error``.
        """
        counts: dict[str, int] = {}
        for o in self.outcomes:
            cat = o.verdict.category if o.verdict else ("infra_error" if o.error else "ok")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def num_scored(self) -> int:
        return sum(1 for o in self.outcomes if o.is_scored)

    def token_totals(self) -> TokenStats:
        """Sum token + cost usage across every outcome (Harbor-parity totals)."""
        total = TokenStats()
        for o in self.outcomes:
            total.merge(o.stats)
        return total

    def total_turns(self) -> int:
        return sum(o.turns for o in self.outcomes)

    def efficiency(self) -> dict[str, Any]:
        """Cost/effort summary (tokens + turns) alongside the score.

        The same shape the reproducibility manifest embeds, so a quantized
        model's *quality* can always be read next to its *efficiency*.
        """
        n = self.num_tasks or 1
        eff: dict[str, Any] = {
            "num_tasks": self.num_tasks,
            "total_turns": self.total_turns(),
            "avg_turns": round(self.total_turns() / n, 2),
        }
        tokens = self.token_totals()
        if not tokens.is_empty():
            eff["tokens"] = tokens.to_dict()
        return eff

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_tasks": self.num_tasks,
            "num_scored": self.num_scored(),
            "mean_score": self.mean_score(),
            "num_errors": sum(1 for o in self.outcomes if o.error),
            "by_category": self.by_category(),
            "efficiency": self.efficiency(),
            "outcomes": [
                {
                    "task_id": o.task_id,
                    "score": o.score,
                    "error": o.error,
                    **({"turns": o.turns} if o.turns else {}),
                    **({"tokens": o.stats.to_dict()}
                       if o.stats is not None and not o.stats.is_empty() else {}),
                    **({"verdict": o.verdict.to_dict()} if o.verdict else {}),
                }
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
                outcomes[idx] = TaskOutcome(task.id, "", 0.0, error=str(exc),
                                            verdict=verdict_from_exception(exc))
                return
            try:
                score = float(score_fn(task, response))
            except Exception as exc:  # noqa: BLE001 - scoring error is per-task
                outcomes[idx] = TaskOutcome(task.id, response, 0.0,
                                            error=f"score: {exc}",
                                            verdict=verdict_from_exception(exc))
                return
            outcomes[idx] = TaskOutcome(task.id, response, score,
                                        verdict=coerce_verdict(score))

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(tasks)))
    result = AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])
    _log.info("agentic eval %s: %d tasks, mean=%.4f, errors=%d",
              name, result.num_tasks, result.mean_score(),
              sum(1 for o in result.outcomes if o.error))
    return result

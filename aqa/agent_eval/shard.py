"""Shard an agentic eval across a pool of served endpoints (E1).

Ports the role of NeMo Evaluator's ``ray_launcher.py``: distribute tasks across
several model endpoints for throughput.  The core is dependency-free -- a pool of
:class:`~aqa.agent_eval.solver.Solver` (each backed by its own endpoint) with tasks
assigned round-robin and run under bounded per-shard concurrency.  Ray is only an
*optional* cluster backend (:func:`ray_shard_endpoints`); the asyncio pool below
covers the common single-host multi-GPU case with no heavy dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from ..logging_util import AqaError, get_logger
from .loop import AgentEvalResult, ScoreFn, TaskOutcome
from .solver import Solver, Task

_log = get_logger("agent_eval.shard")


async def run_sharded_eval(
    name: str,
    tasks: list[Task],
    solvers: Sequence[Solver],
    score_fn: ScoreFn,
    *,
    concurrency_per_shard: int = 8,
) -> AgentEvalResult:
    """Round-robin tasks across ``solvers`` (one per endpoint); aggregate results.

    Each shard runs its slice with its own concurrency limit, so total in-flight
    requests is ``len(solvers) * concurrency_per_shard``.  Task order is preserved
    in the aggregated result.
    """
    if not solvers:
        raise AqaError("run_sharded_eval needs at least one solver")
    outcomes: list[TaskOutcome | None] = [None] * len(tasks)
    sems = [asyncio.Semaphore(max(1, concurrency_per_shard)) for _ in solvers]

    async def _one(idx: int, task: Task) -> None:
        shard = idx % len(solvers)
        async with sems[shard]:
            try:
                response = await solvers[shard].solve(task)
            except Exception as exc:  # noqa: BLE001 - isolate per-task failure
                _log.warning("task %s (shard %d) failed: %s", task.id, shard, exc)
                outcomes[idx] = TaskOutcome(task.id, "", 0.0, error=str(exc))
                return
            try:
                score = float(score_fn(task, response))
            except Exception as exc:  # noqa: BLE001
                outcomes[idx] = TaskOutcome(task.id, response, 0.0, error=f"score: {exc}")
                return
            outcomes[idx] = TaskOutcome(task.id, response, score)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(tasks)))
    result = AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])
    _log.info("sharded eval %s: %d tasks over %d shard(s), mean=%.4f, errors=%d",
              name, result.num_tasks, len(solvers), result.mean_score(),
              sum(1 for o in result.outcomes if o.error))
    return result


def split_gpu_shards(gpu_ids: Sequence[int], per_shard: int) -> list[list[int]]:
    """Partition GPU ids into shards of ``per_shard`` (e.g. tensor-parallel groups)."""
    if per_shard < 1:
        raise AqaError("per_shard must be >= 1")
    ids = list(gpu_ids)
    return [ids[i:i + per_shard] for i in range(0, len(ids), per_shard)]


# solver_factory(base_url, served_name) -> Solver
SolverFactory = Callable[[str, str], Solver]


def ray_shard_endpoints(
    name: str,
    tasks: list[Task],
    endpoints: Sequence[tuple[str, str]],
    solver_factory: SolverFactory,
    score_fn: ScoreFn,
    *,
    concurrency_per_shard: int = 8,
) -> AgentEvalResult:
    """Optional Ray backend: fan tasks out to endpoints as Ray tasks (needs ``[ray]``).

    ``endpoints`` is a list of ``(base_url, served_name)``.  Falls back to nothing
    special beyond distributing shards as Ray remote tasks; the per-shard eval reuses
    :func:`run_sharded_eval`.  For single-host use, prefer :func:`run_sharded_eval`
    directly (no Ray dependency).
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - only without ray
        raise AqaError("ray is required for ray_shard_endpoints; "
                       "`pip install autoquant-agent[ray]`") from exc

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    shards = [tasks[i::len(endpoints)] for i in range(len(endpoints))]
    index_map = [list(range(i, len(tasks), len(endpoints))) for i in range(len(endpoints))]

    @ray.remote
    def _run_shard(base_url: str, served: str, shard_tasks: list[Task]) -> AgentEvalResult:
        solver = solver_factory(base_url, served)
        return asyncio.run(run_sharded_eval(
            name, shard_tasks, [solver], score_fn,
            concurrency_per_shard=concurrency_per_shard))

    futures = [
        _run_shard.remote(base_url, served, shard)
        for (base_url, served), shard in zip(endpoints, shards, strict=True)
    ]
    shard_results = ray.get(futures)

    # Reassemble into original task order.
    outcomes: list[TaskOutcome | None] = [None] * len(tasks)
    for idxs, res in zip(index_map, shard_results, strict=True):
        for pos, outcome in zip(idxs, res.outcomes, strict=True):
            outcomes[pos] = outcome
    return AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])

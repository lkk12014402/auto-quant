"""High-level convenience: serve a model then run an agentic eval against it.

Ties the E0 primitives together for real usage: launch ``vllm serve`` for a
(quantized) model, build a :class:`ModelClient` against its endpoint, and run a
solver over a task set with :func:`run_agentic_eval`.  Kept separate from the
primitives so they stay independently testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from .client import ModelClient
from .endpoint import VllmEndpoint
from .loop import AgentEvalResult, ScoreFn, run_agentic_eval
from .solver import ChatSolver, Solver, Task

# solver_factory(client) -> Solver
SolverFactory = Callable[[ModelClient], Solver]


def serve_and_eval(
    model: str,
    tasks: list[Task],
    score_fn: ScoreFn,
    *,
    name: str = "agentic",
    system: str | None = None,
    num_gpus: int = 1,
    concurrency: int = 8,
    served_name: str | None = None,
    endpoint_extra_args: list[str] | None = None,
    request_params: dict | None = None,
    log_dir: Path | None = None,
    solver_factory: SolverFactory | None = None,
) -> AgentEvalResult:
    """Serve ``model`` via vllm, then solve+score ``tasks`` against the endpoint.

    ``solver_factory`` lets callers plug a richer (tool-use/ReAct) solver; by
    default a single-turn :class:`ChatSolver` is used.
    """
    log_path = (log_dir / "vllm_serve.log") if log_dir else None
    with VllmEndpoint(model, served_name=served_name, num_gpus=num_gpus,
                      extra_args=endpoint_extra_args, log_path=log_path) as ep:
        client = ModelClient(base_url=ep.base_url, model=ep.served_name,
                             default_params=request_params or {})
        solver: Solver = (solver_factory(client) if solver_factory
                          else ChatSolver(client=client, system=system))
        return asyncio.run(
            run_agentic_eval(name, tasks, solver, score_fn, concurrency=concurrency))

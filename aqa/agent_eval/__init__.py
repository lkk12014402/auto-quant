"""Endpoint-centric (agentic) evaluation -- E0 base.

Provides the foundation for evaluating a served (quantized) model on agentic and
terminal benchmarks, which drive the model as an OpenAI-compatible endpoint rather
than via the lm_eval subprocess path:

  * :class:`ModelClient`   -- async chat/completion client (injectable transport).
  * :class:`VllmEndpoint`  -- launch/manage ``vllm serve`` and wait for ``/health``.
  * :class:`Solver` / :class:`ChatSolver` / :class:`CompletionSolver` -- prompt shape.
  * :func:`run_agentic_eval` -- bounded-concurrency async solve + score + aggregate.

Later phases (ray sharding, sandboxes, tool-use/terminal environments) build on
these primitives.
"""

from __future__ import annotations

from .client import ModelClient, Transport
from .endpoint import VllmEndpoint
from .environment import Environment, Observation, SandboxEnvironment
from .loop import AgentEvalResult, ScoreFn, TaskOutcome, run_agentic_eval
from .proxy import ProxyStats, ServingProxy
from .react import (
    Action,
    Command,
    Finish,
    ReActStep,
    ToolUseSolver,
    Trajectory,
    parse_action,
    run_tool_use_eval,
)
from .run import SolverFactory, serve_and_eval
from .shard import ray_shard_endpoints, run_sharded_eval, split_gpu_shards
from .solver import ChatSolver, CompletionSolver, Solver, Task

__all__ = [
    "Action",
    "AgentEvalResult",
    "ChatSolver",
    "Command",
    "CompletionSolver",
    "Environment",
    "Finish",
    "ModelClient",
    "Observation",
    "ProxyStats",
    "ReActStep",
    "SandboxEnvironment",
    "ScoreFn",
    "ServingProxy",
    "Solver",
    "SolverFactory",
    "Task",
    "TaskOutcome",
    "ToolUseSolver",
    "Trajectory",
    "Transport",
    "VllmEndpoint",
    "parse_action",
    "ray_shard_endpoints",
    "run_agentic_eval",
    "run_sharded_eval",
    "run_tool_use_eval",
    "serve_and_eval",
    "split_gpu_shards",
]

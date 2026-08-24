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
from .scaffold import (
    AgentScaffold,
    CliAgentScaffold,
    CodexScaffold,
    ScaffoldContext,
    SweAgentScaffold,
    available_scaffolds,
    register_scaffold,
    resolve_scaffold,
)
from .shard import ray_shard_endpoints, run_sharded_eval, split_gpu_shards
from .solver import ChatSolver, CompletionSolver, Solver, Task
from .stats import TokenStats
from .verifier import (
    EvalVerdict,
    InfraError,
    ParseError,
    Verifier,
    VerifierError,
    coerce_verdict,
    verdict_from_exception,
)

__all__ = [
    "Action",
    "AgentEvalResult",
    "AgentScaffold",
    "ChatSolver",
    "CliAgentScaffold",
    "CodexScaffold",
    "Command",
    "CompletionSolver",
    "Environment",
    "EvalVerdict",
    "Finish",
    "InfraError",
    "ModelClient",
    "Observation",
    "ParseError",
    "ProxyStats",
    "ReActStep",
    "SandboxEnvironment",
    "ScaffoldContext",
    "ScoreFn",
    "ServingProxy",
    "Solver",
    "SolverFactory",
    "SweAgentScaffold",
    "Task",
    "TaskOutcome",
    "TokenStats",
    "ToolUseSolver",
    "Trajectory",
    "Transport",
    "Verifier",
    "VerifierError",
    "VllmEndpoint",
    "available_scaffolds",
    "coerce_verdict",
    "parse_action",
    "ray_shard_endpoints",
    "register_scaffold",
    "resolve_scaffold",
    "run_agentic_eval",
    "run_sharded_eval",
    "run_tool_use_eval",
    "serve_and_eval",
    "split_gpu_shards",
    "verdict_from_exception",
]

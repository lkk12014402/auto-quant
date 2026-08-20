"""Run a terminal-benchmark suite on the sandbox + tool-use agent stack.

Bridges E4 :class:`TerminalTask` definitions to the E3 :class:`ToolUseSolver`:
each task becomes an agentic :class:`~aqa.agent_eval.solver.Task`, and a per-task
:class:`~aqa.agent_eval.environment.SandboxEnvironment` is built from its sandbox
kind / image / setup / files / verifier.  Scoring is the task reward (1.0 when the
verifier passes), aggregated into the same :class:`AgentEvalResult` as every other
agentic path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent_eval.environment import SandboxEnvironment
from ..agent_eval.loop import AgentEvalResult
from ..agent_eval.react import ToolUseSolver, run_tool_use_eval
from ..agent_eval.solver import Task
from ..sandbox import create_sandbox

if TYPE_CHECKING:  # pragma: no cover
    from .task import TerminalTask


def to_agent_task(task: TerminalTask) -> Task:
    """Project a terminal task into the agentic :class:`Task` the solver drives."""
    return Task(id=task.id, prompt=task.instruction, metadata={"terminal": True})


def build_env_factory(
    tasks: list[TerminalTask],
    *,
    default_sandbox: str = "local",
    sandbox_options: dict[str, Any] | None = None,
):
    """Return an ``env_factory(agent_task) -> SandboxEnvironment`` for the suite."""
    by_id = {t.id: t for t in tasks}
    base_opts = dict(sandbox_options or {})

    def env_factory(agent_task: Task) -> SandboxEnvironment:
        task = by_id[agent_task.id]
        kind = task.sandbox or default_sandbox
        opts = dict(base_opts)
        if kind == "docker" and task.image:
            opts.setdefault("image", task.image)
        sandbox = create_sandbox(kind, **opts)
        return SandboxEnvironment(
            sandbox=sandbox,
            setup=task.setup,
            files=task.files,
            verify=task.verify.build(),
            exec_timeout=task.timeout,
        )

    return env_factory


async def run_terminal_bench(
    name: str,
    tasks: list[TerminalTask],
    client: Any,
    *,
    default_sandbox: str = "local",
    sandbox_options: dict[str, Any] | None = None,
    concurrency: int = 4,
    max_turns: int | None = None,
) -> AgentEvalResult:
    """Evaluate ``client`` on a terminal-task suite; returns solve-rate result.

    ``max_turns`` overrides every task's budget; when omitted the largest task
    budget in the suite is used (a single shared agent drives all tasks).
    """
    env_factory = build_env_factory(
        tasks, default_sandbox=default_sandbox, sandbox_options=sandbox_options
    )
    budget = max_turns if max_turns is not None else max(
        (t.max_turns for t in tasks), default=15
    )
    agent = ToolUseSolver(client=client, env_factory=env_factory, max_turns=budget)
    agent_tasks = [to_agent_task(t) for t in tasks]
    return await run_tool_use_eval(name, agent_tasks, agent, concurrency=concurrency)

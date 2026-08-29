"""SWE-bench runner: evaluate a served model on SWE-bench instances.

Wires the pieces together: each :class:`~aqa.swebench.dataset.SWEBenchInstance`
becomes an agentic :class:`~aqa.agent_eval.solver.Task` (prompt = the issue), a
per-instance :class:`~aqa.swebench.environment.SWEBenchEnvironment` runs inside
the released eval container, and the ReAct
:class:`~aqa.agent_eval.react.ToolUseSolver` drives the model.  Rewards *are* the
official resolution verdict, so the aggregate mean equals the resolved-rate.

After the run, each environment's captured patch + grading report is harvested
into :class:`~aqa.swebench.report.InstanceResult` records and (optionally) written
as ``predictions.jsonl`` + ``report.json``.

The sandbox is created through an injectable ``sandbox_factory`` so the whole
runner is unit-tested with a fake sandbox + fake client — no Docker required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent_eval.react import run_tool_use_eval
from ..agent_eval.scaffold import ScaffoldContext, resolve_scaffold
from ..agent_eval.solver import Task
from ..logging_util import get_logger
from ..sandbox import create_sandbox
from .environment import SWEBenchEnvironment, SWEBenchVerifier
from .report import (
    InstanceResult,
    SWEBenchRunReport,
    write_predictions,
    write_report,
)
from .spec import TESTBED, build_instruction, docker_image_for

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ..agent_eval.loop import AgentEvalResult
    from ..sandbox.base import Sandbox
    from .dataset import SWEBenchInstance
    from .spec import TestSpec

_log = get_logger("swebench.run")

SWE_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a real bug in a large codebase. "
    "You work by running shell commands in a checked-out repository: inspect the "
    "code, make focused edits to the source (never the tests), and verify. Emit "
    "exactly one ```bash``` command per turn, then wait for its output. When the "
    "issue is fixed, reply with `Final Answer:` and a short summary."
)


def default_sandbox_factory(
    *, namespace: str = "swebench", arch: str = "x86_64", tag: str = "latest",
    sandbox_options: dict[str, Any] | None = None,
) -> Callable[[SWEBenchInstance], Sandbox]:
    """A factory building a Docker sandbox on each instance's eval image."""
    base_opts = dict(sandbox_options or {})

    def factory(instance: SWEBenchInstance) -> Sandbox:
        opts = dict(base_opts)
        opts.setdefault("image", docker_image_for(
            instance, namespace=namespace, arch=arch, tag=tag))
        opts.setdefault("workdir", TESTBED)
        return create_sandbox("docker", **opts)

    return factory


async def run_swebench_async(
    instances: list[SWEBenchInstance],
    client: Any,
    *,
    name: str = "swebench",
    model_name: str = "aqa-agent",
    sandbox_factory: Callable[[SWEBenchInstance], Sandbox] | None = None,
    spec_provider: Callable[[SWEBenchInstance], TestSpec | None] | None = None,
    concurrency: int = 4,
    max_turns: int = 40,
    scaffold: str = "aqa-react",
    exec_timeout: float = 900.0,
    test_timeout: float = 1800.0,
    out_dir: str | Path | None = None,
    dump_trajectories: bool = True,
) -> tuple[AgentEvalResult, SWEBenchRunReport]:
    """Evaluate ``client`` on ``instances``; return (agent result, SWE report)."""
    factory = sandbox_factory or default_sandbox_factory()
    by_id = {i.instance_id: i for i in instances}
    envs: dict[str, SWEBenchEnvironment] = {}

    def env_factory(task: Task) -> SWEBenchEnvironment:
        inst = by_id[task.id]
        env = SWEBenchEnvironment(
            inst, factory(inst), spec_provider=spec_provider,
            exec_timeout=exec_timeout, test_timeout=test_timeout,
        )
        envs[task.id] = env
        return env

    tasks = [
        Task(id=i.instance_id, prompt=build_instruction(i), metadata={"swebench": True})
        for i in instances
    ]
    agent = resolve_scaffold(scaffold, ScaffoldContext(
        client=client, env_factory=env_factory,
        verifier=SWEBenchVerifier(), system_prompt=SWE_SYSTEM_PROMPT,
        max_turns=max_turns, command_timeout=exec_timeout,
    ))
    _log.info("swebench: scaffold=%s over %d instance(s)", scaffold, len(instances))
    agent_result = await run_tool_use_eval(name, tasks, agent, concurrency=concurrency)

    errors = {o.task_id: o.error for o in agent_result.outcomes if o.error}
    stats_by_id = {o.task_id: o.stats for o in agent_result.outcomes}
    turns_by_id = {o.task_id: o.turns for o in agent_result.outcomes}
    results: list[InstanceResult] = []
    for inst in instances:
        env = envs.get(inst.instance_id)
        report = env.report if env is not None else None
        results.append(InstanceResult(
            instance_id=inst.instance_id,
            resolved=bool(report and report.resolved),
            model_patch=env.model_patch if env is not None else "",
            report=report,
            error=errors.get(inst.instance_id),
            stats=stats_by_id.get(inst.instance_id),
            turns=turns_by_id.get(inst.instance_id, 0),
        ))

    run_report = SWEBenchRunReport(model_name=model_name, results=results)
    if out_dir is not None:
        out = Path(out_dir)
        write_predictions(out / "predictions.jsonl", results, model_name)
        write_report(out / "report.json", results, model_name)
        if dump_trajectories:
            # ATIF export for the Harbor toolchain / SFT datasets; a failure
            # here must never fail the run itself.
            try:
                from ..agent_eval.atif import dump_trajectories_jsonl

                tpath = dump_trajectories_jsonl(
                    agent_result, out, tasks=tasks, agent_name=scaffold,
                    model_name=model_name,
                    system_prompt=getattr(agent, "system_prompt", ""),
                )
                if tpath is not None:
                    _log.info("swebench: trajectories (ATIF) -> %s", tpath)
            except Exception as exc:  # noqa: BLE001 - export is best-effort
                _log.warning("swebench: trajectory export failed: %s", exc)
        _log.info("swebench: resolved %d/%d (%.1f%%) -> %s",
                  sum(r.resolved for r in results), len(results),
                  100 * run_report.resolved_rate, out)
    return agent_result, run_report


def run_swebench(instances: list[SWEBenchInstance], client: Any, **kwargs: Any):
    """Synchronous wrapper around :func:`run_swebench_async`."""
    return asyncio.run(run_swebench_async(instances, client, **kwargs))

"""Pluggable agent *scaffolds* (a.k.a. agent infra) for agentic evaluation.

Where :class:`~aqa.agent_eval.react.ToolUseSolver` is *our* hand-written ReAct
loop, a benchmark score for a coding task depends as much on the **scaffold**
(the harness that turns a chat model into an agent) as on the model itself.
Harbor's differentiator is exactly this: run the same task + same served model
under different agent infra (Codex CLI, SWE-agent, OpenHands, ...) and compare.

This module adds that seam.  An :class:`AgentScaffold` is anything that, given a
:class:`~aqa.agent_eval.solver.Task`, drives the model to a
:class:`~aqa.agent_eval.react.Trajectory` -- so it is a drop-in for
:func:`~aqa.agent_eval.react.run_tool_use_eval` (which only needs ``run(task)``).

Two families ship:

* ``aqa-react`` -- our built-in :class:`ToolUseSolver` (default; unchanged path).
* external CLI agents (``codex``, ``swe-agent``) -- driven by
  :class:`CliAgentScaffold`, which runs the third-party agent binary *inside the
  same sandbox* pointed at the served OpenAI-compatible endpoint, lets it edit the
  repo, then grades the working tree with the *same* verifier.  The verdict is
  therefore produced by identical scoring regardless of scaffold, so numbers are
  comparable across agent infra.

The external binaries are assumed present in the eval image; their exact
invocation is a template (``build_command``) so flag drift is a config change,
not a code change.
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..logging_util import AqaError, get_logger
from .react import Finish, ReActStep, ToolUseSolver, Trajectory
from .stats import TokenStats, parse_usage_from_text
from .verifier import EvalVerdict, Verifier, verdict_from_exception

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .client import ModelClient
    from .environment import Environment
    from .solver import Task

_log = get_logger("agent_eval.scaffold")


# ── protocol ───────────────────────────────────────────────────────────────
@runtime_checkable
class AgentScaffold(Protocol):
    """Drives a model to a graded :class:`Trajectory` for one task."""

    name: str

    async def run(self, task: Task) -> Trajectory: ...


# ── shared context + endpoint plumbing ───────────────────────────────────────
@dataclass
class ScaffoldContext:
    """Everything a scaffold factory needs from the runner (injectable for tests)."""

    client: ModelClient
    env_factory: Callable[[Task], Environment]
    verifier: Verifier | None = None
    system_prompt: str = ""
    max_turns: int = 40
    command_timeout: float = 1800.0
    extra_args: list[str] = field(default_factory=list)


def _endpoint_url(base_url: str) -> str:
    """Normalise an endpoint to include the OpenAI ``/v1`` suffix (once)."""
    base = (base_url or "").rstrip("/")
    if base and not base.endswith("/v1"):
        base = base + "/v1"
    return base


def _endpoint_env(ctx: ScaffoldContext) -> dict[str, str]:
    """OpenAI-compatible env every CLI agent understands (base url + key + model)."""
    c = ctx.client
    env = {
        "OPENAI_BASE_URL": _endpoint_url(getattr(c, "base_url", "")),
        "OPENAI_API_BASE": _endpoint_url(getattr(c, "base_url", "")),
        "OPENAI_API_KEY": getattr(c, "api_key", None) or "sk-noauth",
    }
    model = getattr(c, "model", "")
    if model:
        env["OPENAI_MODEL"] = model
    return env


# ── external-CLI scaffold base ────────────────────────────────────────────────
@dataclass
class CliAgentScaffold:
    """Run a third-party CLI agent inside the sandbox, then grade the repo.

    Subclasses implement :meth:`build_command` (how to invoke their binary against
    the issue text + endpoint).  Everything else -- reset, one-shot execution with
    the endpoint env exported, verdict via the shared verifier, and mapping crashes
    to non-scored infra verdicts -- is identical so scaffolds stay comparable.
    """

    name: str
    ctx: ScaffoldContext

    def build_command(self, task: Task) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def parse_usage(self, output: str) -> TokenStats | None:
        """Extract token usage from the agent's own output (default: none).

        Subclasses whose CLI reports token counts (e.g. Codex ``--json``) override
        this so scaffold runs carry the same token/cost totals our ReAct loop does.
        """
        return None

    async def run(self, task: Task) -> Trajectory:
        env = self.ctx.env_factory(task)
        traj = Trajectory(task_id=task.id)
        try:
            await env.reset()
            sandbox = getattr(env, "sandbox", None)
            if sandbox is None:
                raise RuntimeError(
                    f"scaffold '{self.name}' requires a sandbox-backed environment")
            cmd = self.build_command(task)
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            _log.info("[scaffold=%s] %s: %s", self.name, task.id, cmd_str)
            res = await asyncio.to_thread(
                sandbox.exec, cmd_str, timeout=self.ctx.command_timeout,
                env=_endpoint_env(self.ctx),
            )
            output = getattr(res, "output", "") or ""
            traj.stats.merge(self.parse_usage(output))
            traj.steps.append(ReActStep(
                output=output[-4000:], action=Finish(answer=""),
                observation=f"exit={getattr(res, 'returncode', '?')}"))
            verdict = await self._grade(task, traj, env)
            self._settle(traj, verdict)
        except Exception as exc:  # noqa: BLE001 - a scaffold crash is infra, not model-wrong
            _log.warning("[scaffold=%s] %s crashed: %s", self.name, task.id, exc)
            self._settle(traj, verdict_from_exception(exc))
        finally:
            await env.close()
        _log.info("[scaffold=%s] %s: solved=%s reward=%.3f cat=%s tok=%d/%d",
                  self.name, task.id, traj.solved, traj.reward,
                  traj.verdict.category if traj.verdict else "-",
                  traj.stats.n_input_tokens, traj.stats.n_output_tokens)
        return traj

    async def _grade(self, task: Task, traj: Trajectory, env: Environment) -> EvalVerdict:
        if self.ctx.verifier is not None:
            try:
                return await self.ctx.verifier.verify(task, traj, env)
            except Exception as exc:  # noqa: BLE001 - grading crash != model wrong
                _log.warning("verifier failed for %s: %s", task.id, exc)
                return verdict_from_exception(exc)
        is_solved = getattr(env, "is_solved", None)
        if callable(is_solved):
            solved = bool(await is_solved())
            return EvalVerdict.from_reward(1.0 if solved else 0.0, solved=solved)
        return EvalVerdict.from_reward(0.0, solved=False)

    @staticmethod
    def _settle(traj: Trajectory, verdict: EvalVerdict) -> None:
        traj.verdict = verdict
        traj.solved = verdict.solved
        traj.reward = verdict.reward


@dataclass
class CodexScaffold(CliAgentScaffold):
    """OpenAI **Codex CLI** as the agent infra (``codex exec`` non-interactive).

    ``codex exec`` runs headless: it reads the prompt, edits files in the cwd and
    exits.  We point it at the served model via ``OPENAI_BASE_URL`` + ``--model``
    and disable its own sandbox/approvals (the docker sandbox is already the jail).
    """

    def build_command(self, task: Task) -> list[str]:
        model = getattr(self.ctx.client, "model", "") or "aqa-agent"
        cmd = [
            "codex", "exec",
            "--json",
            "--model", model,
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        cmd += list(self.ctx.extra_args)
        cmd.append(task.prompt)
        return cmd

    def parse_usage(self, output: str) -> TokenStats | None:
        """Read the last ``token_count`` event Codex ``exec --json`` emits.

        Codex streams JSONL events; token-usage events carry a
        ``last_token_usage`` (cumulative) block with ``input_tokens`` /
        ``output_tokens`` / ``cached_input_tokens`` -- the same fields Harbor's
        Codex adapter consumes.  The last such block is the run total.
        """
        return parse_usage_from_text(output)


@dataclass
class SweAgentScaffold(CliAgentScaffold):
    """Princeton **SWE-agent** as the agent infra (``sweagent run``).

    SWE-agent operates on a checked-out repo; we run it against the sandbox cwd
    with the problem statement inline, pointed at the served model.  Flags evolve
    across releases, so anything in ``extra_args`` is appended verbatim.
    """

    def build_command(self, task: Task) -> list[str]:
        model = getattr(self.ctx.client, "model", "") or "aqa-agent"
        cmd = [
            "sweagent", "run",
            "--agent.model.name", model,
            "--agent.model.api_base", _endpoint_url(
                getattr(self.ctx.client, "base_url", "")),
            "--env.repo.path", ".",
            "--problem_statement.text", task.prompt,
        ]
        cmd += list(self.ctx.extra_args)
        return cmd


# ── registry ─────────────────────────────────────────────────────────────────
def _build_react(ctx: ScaffoldContext) -> ToolUseSolver:
    kwargs: dict[str, Any] = {
        "client": ctx.client,
        "env_factory": ctx.env_factory,
        "max_turns": ctx.max_turns,
        "verifier": ctx.verifier,
    }
    if ctx.system_prompt:
        kwargs["system_prompt"] = ctx.system_prompt
    solver = ToolUseSolver(**kwargs)
    solver.name = "aqa-react"  # type: ignore[attr-defined]
    return solver


_SCAFFOLDS: dict[str, Callable[[ScaffoldContext], AgentScaffold]] = {
    "aqa-react": _build_react,
    "codex": lambda ctx: CodexScaffold(name="codex", ctx=ctx),
    "swe-agent": lambda ctx: SweAgentScaffold(name="swe-agent", ctx=ctx),
}


def available_scaffolds() -> list[str]:
    """Names of the registered agent scaffolds (sorted, ``aqa-react`` first)."""
    names = sorted(_SCAFFOLDS)
    names.remove("aqa-react")
    return ["aqa-react", *names]


def register_scaffold(
    name: str, factory: Callable[[ScaffoldContext], AgentScaffold]
) -> None:
    """Register a custom scaffold factory (later registration overrides)."""
    _SCAFFOLDS[name] = factory


def resolve_scaffold(name: str, ctx: ScaffoldContext) -> AgentScaffold:
    """Build the named scaffold from a :class:`ScaffoldContext`."""
    try:
        factory = _SCAFFOLDS[name]
    except KeyError:
        raise AqaError(
            f"unknown agent scaffold {name!r}; available: "
            f"{', '.join(available_scaffolds())}"
        ) from None
    return factory(ctx)

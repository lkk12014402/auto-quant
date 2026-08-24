"""ReAct / tool-use solver: drive the model through a multi-turn act loop.

Where :class:`~aqa.agent_eval.solver.ChatSolver` answers in one turn, a tool-use
agent *acts*: it emits a shell command, the :class:`~aqa.agent_eval.environment`
executes it, the observation is fed back, and the loop repeats until the model
declares a final answer or the turn budget is exhausted.  This is the classic
ReAct loop, and the base an E4 terminal benchmark is scored on.

The action protocol is text-based (portable across any served model, no function
-calling schema required):

  * a fenced `````bash`` / `````sh`` / ``````` block  -> run it as a command;
  * a ``Final Answer: ...`` line                       -> finish;
  * anything else                                      -> treated as the answer.

The resulting :class:`Trajectory` carries the transcript, the final answer, and
the environment's terminal reward.  :func:`run_tool_use_eval` aggregates
trajectories into the same :class:`~aqa.agent_eval.loop.AgentEvalResult` used by
the single-turn path (so ``per_sample`` feeds the statistical gate identically).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..logging_util import get_logger
from .loop import AgentEvalResult, TaskOutcome
from .solver import Task
from .stats import TokenStats
from .verifier import EvalVerdict, Verifier, coerce_verdict, verdict_from_exception

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .environment import Environment

_log = get_logger("agent_eval.react")


# --------------------------------------------------------------------------
# Action parsing
# --------------------------------------------------------------------------
@dataclass
class Command:
    """The agent wants to run a shell command."""

    cmd: str


@dataclass
class Finish:
    """The agent is done and submits a final answer."""

    answer: str


Action = Command | Finish

_CODE_BLOCK = re.compile(
    r"```(?:bash|sh|shell|console)?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FINAL = re.compile(r"(?:^|\n)\s*final\s*answer\s*:\s*(?P<ans>.*)", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> Action:
    """Extract the next action from a model turn (command takes priority)."""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        # Use the last fenced block; strip a leading `$ ` prompt if present.
        cmd = blocks[-1].strip()
        cmd = "\n".join(line.removeprefix("$ ")
                        for line in cmd.splitlines()).strip()
        if cmd:
            return Command(cmd=cmd)
    final = _FINAL.search(text)
    if final:
        return Finish(answer=final.group("ans").strip())
    return Finish(answer=text.strip())


DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent solving a task in a Linux shell. "
    "Think step by step. To run a command, reply with a single fenced code block:\n"
    "```bash\n<command>\n```\n"
    "You will then receive its output as an Observation. "
    "When the task is complete, reply with a line:\nFinal Answer: <your answer>\n"
    "Do not include both a command and a final answer in the same reply."
)


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------
@dataclass
class ReActStep:
    output: str                     # raw model turn
    action: Action
    observation: str | None = None  # None for a Finish step


@dataclass
class Trajectory:
    task_id: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    solved: bool = False
    reward: float = 0.0
    hit_turn_limit: bool = False
    verdict: EvalVerdict | None = None
    stats: TokenStats = field(default_factory=TokenStats)

    @property
    def turns(self) -> int:
        return len(self.steps)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
@dataclass
class ToolUseSolver:
    """A ReAct agent driving a model against a per-task :class:`Environment`."""

    client: Any                                    # ModelClient / ServingProxy
    env_factory: Callable[[Task], Environment]
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 10
    params: dict[str, Any] = field(default_factory=dict)
    observation_template: str = "Observation:\n{obs}"
    verifier: Verifier | None = None

    def _initial_user(self, task: Task, obs_text: str) -> str:
        parts = [task.prompt]
        if obs_text:
            parts.append(self.observation_template.format(obs=obs_text))
        return "\n\n".join(parts)

    async def _terminal_verdict(self, task: Task, traj: Trajectory,
                                env: Environment, last: float) -> EvalVerdict:
        """Grade the finished trajectory.

        Uses the injected :class:`Verifier` when present (structured, may carry
        sub-metrics + a failure category); otherwise falls back to the legacy
        ``is_solved`` / last-step-reward scalar path, wrapped as a scored verdict.
        A verifier that raises is turned into a non-scored (infra/parse) verdict
        so a grading crash never masquerades as the model being wrong.
        """
        if self.verifier is not None:
            try:
                return await self.verifier.verify(task, traj, env)
            except Exception as exc:  # noqa: BLE001 - grading crash != model wrong
                _log.warning("verifier failed for %s: %s", task.id, exc)
                return verdict_from_exception(exc)
        is_solved = getattr(env, "is_solved", None)
        if callable(is_solved):
            solved = bool(await is_solved())
            return EvalVerdict.from_reward(1.0 if solved else 0.0, solved=solved)
        return EvalVerdict.from_reward(last, solved=last > 0.0)

    async def run(self, task: Task) -> Trajectory:
        env = self.env_factory(task)
        traj = Trajectory(task_id=task.id)
        last_reward = 0.0
        try:
            first = await env.reset()
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._initial_user(task, first.text)},
            ]
            for _ in range(self.max_turns):
                output = await self._chat(messages, traj)
                messages.append({"role": "assistant", "content": output})
                action = parse_action(output)
                if isinstance(action, Finish):
                    traj.steps.append(ReActStep(output, action))
                    traj.final_answer = action.answer
                    self._settle(traj, await self._terminal_verdict(
                        task, traj, env, last_reward))
                    break
                obs = await env.step(action.cmd)
                last_reward = max(last_reward, obs.reward)
                traj.steps.append(ReActStep(output, action, obs.text))
                messages.append(
                    {"role": "user",
                     "content": self.observation_template.format(obs=obs.text)}
                )
                if obs.done:
                    self._settle(traj, await self._terminal_verdict(
                        task, traj, env, last_reward))
                    break
            else:
                traj.hit_turn_limit = True
                self._settle(traj, await self._terminal_verdict(
                    task, traj, env, last_reward))
        finally:
            await env.close()
        _log.info("task %s: turns=%d solved=%s reward=%.3f cat=%s tok=%d/%d%s",
                  task.id, traj.turns, traj.solved, traj.reward,
                  traj.verdict.category if traj.verdict else "-",
                  traj.stats.n_input_tokens, traj.stats.n_output_tokens,
                  " (turn-limit)" if traj.hit_turn_limit else "")
        return traj

    @staticmethod
    def _settle(traj: Trajectory, verdict: EvalVerdict) -> None:
        traj.verdict = verdict
        traj.solved = verdict.solved
        traj.reward = verdict.reward

    async def _chat(self, messages: list[dict[str, str]], traj: Trajectory) -> str:
        """One model turn, folding any reported ``usage`` into ``traj.stats``.

        Falls back to the plain :meth:`ModelClient.chat` for clients (or fakes)
        that predate ``chat_with_usage``, so token accounting is best-effort and
        never a hard dependency.
        """
        with_usage = getattr(self.client, "chat_with_usage", None)
        if callable(with_usage):
            output, usage = await with_usage(messages, **self.params)
            traj.stats.add_usage(usage)
            return output
        return await self.client.chat(messages, **self.params)

    async def solve(self, task: Task) -> str:
        """:class:`Solver` adapter -- returns the final answer text."""
        return (await self.run(task)).final_answer


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
@runtime_checkable
class TrajectoryAgent(Protocol):
    """Anything that turns a task into a graded :class:`Trajectory`.

    Both :class:`ToolUseSolver` and the pluggable scaffolds satisfy this, so the
    aggregator below is agent-infra agnostic.
    """

    async def run(self, task: Task) -> Trajectory: ...


async def run_tool_use_eval(
    name: str,
    tasks: list[Task],
    agent: TrajectoryAgent,
    *,
    concurrency: int = 4,
    score_fn: Callable[[Task, Trajectory], float] | None = None,
) -> AgentEvalResult:
    """Run any trajectory agent over ``tasks`` and aggregate rewards (bounded pool)."""
    sem = asyncio.Semaphore(max(1, concurrency))
    outcomes: list[TaskOutcome | None] = [None] * len(tasks)

    async def _one(idx: int, task: Task) -> None:
        async with sem:
            try:
                traj = await agent.run(task)
            except Exception as exc:  # noqa: BLE001 - isolate per-task failures
                _log.warning("task %s failed: %s", task.id, exc)
                outcomes[idx] = TaskOutcome(
                    task.id, "", 0.0, error=str(exc),
                    verdict=verdict_from_exception(exc))
                return
            try:
                if score_fn is None:
                    score, verdict = traj.reward, traj.verdict
                else:
                    score = float(score_fn(task, traj))
                    verdict = coerce_verdict(score)
            except Exception as exc:  # noqa: BLE001 - scoring error is per-task
                outcomes[idx] = TaskOutcome(task.id, traj.final_answer, 0.0,
                                            error=f"score: {exc}",
                                            verdict=verdict_from_exception(exc),
                                            stats=traj.stats, turns=traj.turns)
                return
            outcomes[idx] = TaskOutcome(task.id, traj.final_answer, score,
                                        verdict=verdict,
                                        stats=traj.stats, turns=traj.turns)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(tasks)))
    result = AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])
    _log.info("tool-use eval %s: %d tasks, mean=%.4f, errors=%d",
              name, result.num_tasks, result.mean_score(),
              sum(1 for o in result.outcomes if o.error))
    return result

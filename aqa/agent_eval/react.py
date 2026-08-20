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
from typing import TYPE_CHECKING, Any

from ..logging_util import get_logger
from .loop import AgentEvalResult, TaskOutcome
from .solver import Task

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

    def _initial_user(self, task: Task, obs_text: str) -> str:
        parts = [task.prompt]
        if obs_text:
            parts.append(self.observation_template.format(obs=obs_text))
        return "\n\n".join(parts)

    async def _terminal_reward(self, env: Environment, last: float) -> tuple[bool, float]:
        is_solved = getattr(env, "is_solved", None)
        if callable(is_solved):
            solved = bool(await is_solved())
            return solved, (1.0 if solved else 0.0)
        return last > 0.0, last

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
                output = await self.client.chat(messages, **self.params)
                messages.append({"role": "assistant", "content": output})
                action = parse_action(output)
                if isinstance(action, Finish):
                    traj.steps.append(ReActStep(output, action))
                    traj.final_answer = action.answer
                    traj.solved, traj.reward = await self._terminal_reward(env, last_reward)
                    break
                obs = await env.step(action.cmd)
                last_reward = max(last_reward, obs.reward)
                traj.steps.append(ReActStep(output, action, obs.text))
                messages.append(
                    {"role": "user",
                     "content": self.observation_template.format(obs=obs.text)}
                )
                if obs.done:
                    traj.solved, traj.reward = await self._terminal_reward(env, last_reward)
                    break
            else:
                traj.hit_turn_limit = True
                traj.solved, traj.reward = await self._terminal_reward(env, last_reward)
        finally:
            await env.close()
        _log.info("task %s: turns=%d solved=%s reward=%.3f%s",
                  task.id, traj.turns, traj.solved, traj.reward,
                  " (turn-limit)" if traj.hit_turn_limit else "")
        return traj

    async def solve(self, task: Task) -> str:
        """:class:`Solver` adapter -- returns the final answer text."""
        return (await self.run(task)).final_answer


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
async def run_tool_use_eval(
    name: str,
    tasks: list[Task],
    agent: ToolUseSolver,
    *,
    concurrency: int = 4,
    score_fn: Callable[[Task, Trajectory], float] | None = None,
) -> AgentEvalResult:
    """Run the ReAct agent over ``tasks`` and aggregate rewards (bounded pool)."""
    sem = asyncio.Semaphore(max(1, concurrency))
    outcomes: list[TaskOutcome | None] = [None] * len(tasks)

    async def _one(idx: int, task: Task) -> None:
        async with sem:
            try:
                traj = await agent.run(task)
            except Exception as exc:  # noqa: BLE001 - isolate per-task failures
                _log.warning("task %s failed: %s", task.id, exc)
                outcomes[idx] = TaskOutcome(task.id, "", 0.0, error=str(exc))
                return
            try:
                score = traj.reward if score_fn is None else float(score_fn(task, traj))
            except Exception as exc:  # noqa: BLE001 - scoring error is per-task
                outcomes[idx] = TaskOutcome(task.id, traj.final_answer, 0.0,
                                            error=f"score: {exc}")
                return
            outcomes[idx] = TaskOutcome(task.id, traj.final_answer, score)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(tasks)))
    result = AgentEvalResult(name=name, outcomes=[o for o in outcomes if o is not None])
    _log.info("tool-use eval %s: %d tasks, mean=%.4f, errors=%d",
              name, result.num_tasks, result.mean_score(),
              sum(1 for o in result.outcomes if o.error))
    return result

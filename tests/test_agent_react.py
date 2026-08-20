"""Tests for E3: ReAct tool-use solver, sandbox environment, serving proxy."""

from __future__ import annotations

import asyncio

import pytest

from aqa.agent_eval import (
    Command,
    Finish,
    SandboxEnvironment,
    ServingProxy,
    Task,
    ToolUseSolver,
    parse_action,
    run_tool_use_eval,
)
from aqa.agent_eval.environment import Observation, _truncate
from aqa.logging_util import AqaError
from aqa.sandbox import LocalSandbox


# --------------------------------------------------------------------------
# action parsing
# --------------------------------------------------------------------------
def test_parse_bash_block():
    act = parse_action("thinking\n```bash\nls -la\n```")
    assert isinstance(act, Command)
    assert act.cmd == "ls -la"


def test_parse_generic_and_sh_blocks():
    assert parse_action("```\ncat x\n```").cmd == "cat x"
    assert parse_action("```sh\npwd\n```").cmd == "pwd"


def test_parse_last_block_wins_and_strips_prompt():
    act = parse_action("```bash\nfirst\n```\nthen\n```bash\n$ second\n```")
    assert isinstance(act, Command)
    assert act.cmd == "second"


def test_parse_final_answer():
    act = parse_action("Some reasoning.\nFinal Answer: the result is 42")
    assert isinstance(act, Finish)
    assert act.answer == "the result is 42"


def test_parse_plain_text_is_finish():
    act = parse_action("just an answer")
    assert isinstance(act, Finish)
    assert act.answer == "just an answer"


def test_parse_command_takes_priority_over_final():
    # a code block present -> still acting, even if 'final answer' words appear
    act = parse_action("Final answer maybe\n```bash\necho hi\n```")
    assert isinstance(act, Command)


# --------------------------------------------------------------------------
# truncation helper
# --------------------------------------------------------------------------
def test_truncate_middle():
    assert _truncate("short", 100) == "short"
    out = _truncate("x" * 100, 20)
    assert "truncated" in out
    assert len(out) < 100


# --------------------------------------------------------------------------
# ServingProxy: turn counter + cache + budget
# --------------------------------------------------------------------------
class RecordingClient:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.chat_calls = 0
        self.complete_calls = 0

    async def chat(self, messages, **params):
        self.chat_calls += 1
        return f"{self.reply}-{self.chat_calls}"

    async def complete(self, prompt, **params):
        self.complete_calls += 1
        return f"c{self.complete_calls}"


def test_proxy_counts_turns():
    client = RecordingClient()
    proxy = ServingProxy(client, cache=False)
    asyncio.run(proxy.chat([{"role": "user", "content": "a"}]))
    asyncio.run(proxy.chat([{"role": "user", "content": "b"}]))
    assert proxy.stats.turns == 2
    assert client.chat_calls == 2


def test_proxy_caches_identical_requests():
    client = RecordingClient()
    proxy = ServingProxy(client, cache=True)
    msg = [{"role": "user", "content": "same"}]
    r1 = asyncio.run(proxy.chat(msg))
    r2 = asyncio.run(proxy.chat(msg))
    assert r1 == r2                       # cached -> identical
    assert client.chat_calls == 1         # endpoint hit once
    assert proxy.stats.cache_hits == 1
    assert proxy.stats.turns == 1
    assert proxy.stats.calls == 2


def test_proxy_cache_distinguishes_params():
    client = RecordingClient()
    proxy = ServingProxy(client, cache=True)
    msg = [{"role": "user", "content": "x"}]
    asyncio.run(proxy.chat(msg, temperature=0))
    asyncio.run(proxy.chat(msg, temperature=1))
    assert client.chat_calls == 2         # different params -> distinct keys


def test_proxy_turn_budget_enforced():
    proxy = ServingProxy(RecordingClient(), cache=False, max_turns=1)
    asyncio.run(proxy.chat([{"role": "user", "content": "a"}]))
    with pytest.raises(AqaError, match="turn budget"):
        asyncio.run(proxy.chat([{"role": "user", "content": "b"}]))


# --------------------------------------------------------------------------
# SandboxEnvironment (real LocalSandbox)
# --------------------------------------------------------------------------
def test_sandbox_environment_step_and_verify():
    async def go():
        env = SandboxEnvironment(
            sandbox=LocalSandbox(),
            files={"seed.txt": "hi"},
            verify_command="test -f out.txt",
        )
        await env.reset()
        assert (await env.step("cat seed.txt")).text == "hi"
        assert not await env.is_solved()
        await env.step("echo done > out.txt")
        assert await env.is_solved()
        await env.close()

    asyncio.run(go())


def test_sandbox_environment_verify_callable():
    async def go():
        env = SandboxEnvironment(
            sandbox=LocalSandbox(),
            verify=lambda sb: sb.read_file("f") == "1" if False else False,
        )
        await env.reset()
        assert not await env.is_solved()
        await env.close()

    asyncio.run(go())


# --------------------------------------------------------------------------
# ToolUseSolver end-to-end (fake client + real sandbox)
# --------------------------------------------------------------------------
class ScriptedClient:
    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    async def chat(self, messages, **params):
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def _touch_env(task):
    return SandboxEnvironment(sandbox=LocalSandbox(), verify_command="test -f done.txt")


def test_tool_use_solver_solves_task():
    replies = ["```bash\necho hi > done.txt\n```", "Final Answer: created it"]
    agent = ToolUseSolver(client=ScriptedClient(replies), env_factory=_touch_env, max_turns=5)
    traj = asyncio.run(agent.run(Task(id="t1", prompt="create done.txt")))
    assert traj.solved
    assert traj.reward == 1.0
    assert traj.turns == 2
    assert traj.final_answer == "created it"
    assert not traj.hit_turn_limit


def test_tool_use_solver_unsolved_when_verify_fails():
    replies = ["```bash\necho nope\n```", "Final Answer: gave up"]
    agent = ToolUseSolver(client=ScriptedClient(replies), env_factory=_touch_env, max_turns=5)
    traj = asyncio.run(agent.run(Task(id="t2", prompt="x")))
    assert not traj.solved
    assert traj.reward == 0.0


def test_tool_use_solver_hits_turn_limit():
    # never finishes -> keeps issuing commands until the budget is spent
    agent = ToolUseSolver(
        client=ScriptedClient(["```bash\ntrue\n```"]),
        env_factory=_touch_env,
        max_turns=3,
    )
    traj = asyncio.run(agent.run(Task(id="t3", prompt="loop")))
    assert traj.hit_turn_limit
    assert traj.turns == 3


def test_tool_use_solver_solve_adapter_returns_answer():
    replies = ["Final Answer: hello"]
    agent = ToolUseSolver(client=ScriptedClient(replies), env_factory=_touch_env, max_turns=2)
    answer = asyncio.run(agent.solve(Task(id="t", prompt="q")))
    assert answer == "hello"


# --------------------------------------------------------------------------
# gym-style environment (step returns done/reward)
# --------------------------------------------------------------------------
class GymEnv:
    def __init__(self):
        self.steps = 0

    async def reset(self):
        return Observation(text="start")

    async def step(self, action):
        self.steps += 1
        if action == "win":
            return Observation(text="you win", done=True, reward=1.0)
        return Observation(text="keep going")

    async def close(self):
        pass


def test_tool_use_solver_uses_step_done_reward():
    replies = ["```bash\nwin\n```"]
    agent = ToolUseSolver(client=ScriptedClient(replies), env_factory=lambda t: GymEnv(), max_turns=5)
    traj = asyncio.run(agent.run(Task(id="g", prompt="win the game")))
    assert traj.solved
    assert traj.reward == 1.0


# --------------------------------------------------------------------------
# run_tool_use_eval aggregation
# --------------------------------------------------------------------------
def test_run_tool_use_eval_aggregates_and_preserves_order():
    # even tasks solve, odd tasks fail (based on task id parity via reply)
    def envf(task):
        return SandboxEnvironment(sandbox=LocalSandbox(), verify_command="test -f done.txt")

    class ParityClient:
        async def chat(self, messages, **params):
            # solve only if the initial user prompt says "solve"
            user = messages[1]["content"]
            if "solve" in user:
                return "```bash\ntouch done.txt\n```"
            return "Final Answer: skip"

    solve_tasks = [Task(id=str(i), prompt="solve" if i % 2 == 0 else "skip") for i in range(3)]
    agent = ToolUseSolver(client=ParityClient(), env_factory=envf, max_turns=3)
    result = asyncio.run(run_tool_use_eval("demo", solve_tasks, agent, concurrency=2))
    assert [o.task_id for o in result.outcomes] == ["0", "1", "2"]
    assert result.per_sample() == [1.0, 0.0, 1.0]


def test_run_tool_use_eval_isolates_errors():
    class BoomClient:
        async def chat(self, messages, **params):
            raise RuntimeError("endpoint down")

    tasks = [Task(id="a", prompt="p")]
    agent = ToolUseSolver(client=BoomClient(), env_factory=_touch_env, max_turns=2)
    result = asyncio.run(run_tool_use_eval("demo", tasks, agent))
    assert result.outcomes[0].error is not None
    assert result.mean_score() == 0.0

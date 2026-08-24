"""Tests: Harbor-parity token accounting + pluggable agent scaffolds."""

from __future__ import annotations

import asyncio
import json

import pytest

from aqa.agent_eval import (
    Task,
    ToolUseSolver,
    available_scaffolds,
    register_scaffold,
    resolve_scaffold,
    run_tool_use_eval,
)
from aqa.agent_eval.environment import SandboxEnvironment
from aqa.agent_eval.scaffold import (
    CodexScaffold,
    ScaffoldContext,
    SweAgentScaffold,
    _endpoint_env,
)
from aqa.agent_eval.stats import TokenStats, parse_usage_from_text
from aqa.logging_util import AqaError
from aqa.sandbox import LocalSandbox
from aqa.sandbox.base import ExecResult
from aqa.swebench import SWEBenchInstance, run_swebench
from aqa.swebench.environment import SWEBenchEnvironment, SWEBenchVerifier


# --------------------------------------------------------------------------
# TokenStats
# --------------------------------------------------------------------------
def test_token_stats_add_chat_usage():
    s = TokenStats()
    s.add_usage({"prompt_tokens": 100, "completion_tokens": 20,
                 "prompt_tokens_details": {"cached_tokens": 30}})
    assert s.n_input_tokens == 100
    assert s.n_output_tokens == 20
    assert s.n_cache_tokens == 30
    assert s.total_tokens == 120
    assert s.n_requests == 1


def test_token_stats_add_raw_and_flat_cache():
    s = TokenStats()
    s.add_usage({"input_tokens": 40, "output_tokens": 5, "cached_input_tokens": 10})
    assert (s.n_input_tokens, s.n_output_tokens, s.n_cache_tokens) == (40, 5, 10)


def test_token_stats_empty_usage_is_noop():
    s = TokenStats()
    s.add_usage(None)
    s.add_usage({})
    assert s.is_empty()
    assert s.n_requests == 0


def test_token_stats_merge_and_price():
    a = TokenStats(n_input_tokens=1000, n_cache_tokens=200, n_output_tokens=500,
                   n_requests=2)
    b = TokenStats(n_input_tokens=500, n_output_tokens=100, n_requests=1)
    a.merge(b)
    assert a.n_input_tokens == 1500
    assert a.n_output_tokens == 600
    assert a.n_requests == 3
    a.price(input_per_1k=1.0, output_per_1k=2.0, cached_per_1k=0.5)
    # uncached input = 1500-200=1300 -> 1.3 ; output 600 -> 1.2 ; cache 200 -> 0.1
    assert a.cost_usd == pytest.approx(2.6)


def test_token_stats_to_dict_hides_none_cost():
    s = TokenStats(n_input_tokens=10, n_output_tokens=2)
    d = s.to_dict()
    assert "cost_usd" not in d
    assert d["total_tokens"] == 12


# --------------------------------------------------------------------------
# usage folding through the ReAct loop
# --------------------------------------------------------------------------
class UsageClient:
    """Fake client that reports token usage on every chat turn."""

    base_url = "http://endpoint:8000"
    model = "served-q"
    api_key = "secret"

    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    async def chat_with_usage(self, messages, **params):
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply, {"prompt_tokens": 50, "completion_tokens": 10}


def test_tool_use_solver_folds_token_usage():
    async def go():
        agent = ToolUseSolver(
            client=UsageClient(["Final Answer: done"]),
            env_factory=lambda task: SandboxEnvironment(sandbox=LocalSandbox()),
            max_turns=3,
        )
        return await agent.run(Task(id="t1", prompt="do it"))

    traj = asyncio.run(go())
    assert traj.stats.n_input_tokens == 50
    assert traj.stats.n_output_tokens == 10
    assert traj.stats.n_requests == 1


def test_run_tool_use_eval_surfaces_tokens():
    async def go():
        agent = ToolUseSolver(
            client=UsageClient(["Final Answer: ok"]),
            env_factory=lambda task: SandboxEnvironment(sandbox=LocalSandbox()),
            max_turns=2,
        )
        return await run_tool_use_eval("t", [Task(id="a", prompt="p")], agent)

    result = asyncio.run(go())
    out = result.outcomes[0]
    assert out.stats is not None and out.stats.n_input_tokens == 50
    assert out.turns == 1
    d = result.to_dict()
    assert d["efficiency"]["tokens"]["n_input_tokens"] == 50
    assert d["efficiency"]["total_turns"] == 1
    assert d["outcomes"][0]["tokens"]["n_output_tokens"] == 10
    assert d["outcomes"][0]["turns"] == 1


# --------------------------------------------------------------------------
# scaffold registry
# --------------------------------------------------------------------------
def _ctx(**kw):
    from aqa.agent_eval.client import ModelClient
    client = ModelClient(base_url="http://x:8000", model="m", api_key="k")
    kw.setdefault("client", client)
    kw.setdefault("env_factory", lambda task: SandboxEnvironment(sandbox=LocalSandbox()))
    return ScaffoldContext(**kw)


def test_available_scaffolds_lists_react_first():
    names = available_scaffolds()
    assert names[0] == "aqa-react"
    assert {"codex", "swe-agent"} <= set(names)


def test_resolve_default_scaffold_is_react():
    agent = resolve_scaffold("aqa-react", _ctx(verifier=SWEBenchVerifier(), max_turns=7))
    assert isinstance(agent, ToolUseSolver)
    assert agent.name == "aqa-react"
    assert agent.max_turns == 7


def test_resolve_unknown_scaffold_raises():
    with pytest.raises(AqaError, match="unknown agent scaffold"):
        resolve_scaffold("nope", _ctx())


def test_register_custom_scaffold():
    register_scaffold("dummy", lambda ctx: CodexScaffold(name="dummy", ctx=ctx))
    assert "dummy" in available_scaffolds()
    assert resolve_scaffold("dummy", _ctx()).name == "dummy"


# --------------------------------------------------------------------------
# CLI-agent command + endpoint plumbing
# --------------------------------------------------------------------------
def test_endpoint_env_normalises_v1():
    env = _endpoint_env(_ctx())
    assert env["OPENAI_BASE_URL"] == "http://x:8000/v1"
    assert env["OPENAI_API_KEY"] == "k"
    assert env["OPENAI_MODEL"] == "m"


def test_codex_build_command():
    sc = CodexScaffold(name="codex", ctx=_ctx(extra_args=["--foo"]))
    cmd = sc.build_command(Task(id="i", prompt="fix the bug"))
    assert cmd[:2] == ["codex", "exec"]
    assert "--json" in cmd
    assert "--model" in cmd and "m" in cmd
    assert cmd[-1] == "fix the bug"
    assert "--foo" in cmd


def test_swe_agent_build_command():
    sc = SweAgentScaffold(name="swe-agent", ctx=_ctx())
    cmd = sc.build_command(Task(id="i", prompt="issue text"))
    assert cmd[:2] == ["sweagent", "run"]
    assert "--agent.model.api_base" in cmd
    assert "http://x:8000/v1" in cmd
    assert "issue text" in cmd


def test_codex_parse_usage_last_wins():
    output = "\n".join([
        "some non-json log line",
        json.dumps({"type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 10,
                                                  "output_tokens": 2}}}),
        json.dumps({"type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 120,
                                                  "output_tokens": 30,
                                                  "cached_input_tokens": 40}}}),
    ])
    sc = CodexScaffold(name="codex", ctx=_ctx())
    stats = sc.parse_usage(output)
    assert stats is not None
    assert stats.n_input_tokens == 120
    assert stats.n_output_tokens == 30
    assert stats.n_cache_tokens == 40


def test_parse_jsonl_usage_none_when_absent():
    assert parse_usage_from_text("just logs\nno json") is None


# --------------------------------------------------------------------------
# CLI scaffold end-to-end (fake sandbox, no Docker / no real CLI)
# --------------------------------------------------------------------------
_USAGE_LINE = json.dumps({
    "type": "token_count",
    "info": {"last_token_usage": {"input_tokens": 200, "output_tokens": 50}},
})


class CliFakeSandbox:
    """Scripts the CLI-agent exec (with token usage) + patch/test grading."""

    name = "fake"

    def __init__(self, *, test_log="", test_rc=0):
        self.test_log = test_log
        self.test_rc = test_rc
        self.commands: list[str] = []
        self.envs: list[dict | None] = []
        self.files: dict[str, str] = {}

    def start(self):
        pass

    def close(self):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def read_file(self, path):
        return self.files[path]

    def exec(self, command, *, timeout=None, cwd=None, env=None):
        self.commands.append(command)
        self.envs.append(dict(env) if env else None)
        if command.startswith(("codex", "sweagent")):
            return ExecResult(command, 0, stdout=_USAGE_LINE)
        if "git diff --cached" in command:
            return ExecResult(command, 0, stdout="diff --git a b\n")
        if "pytest" in command or "test_cmd" in command:
            return ExecResult(command, self.test_rc, stdout=self.test_log)
        return ExecResult(command, 0, stdout="")


def _inst(iid="a__a-1", **kw):
    base = {
        "instance_id": iid,
        "repo": "a/a",
        "base_commit": "c0",
        "problem_statement": "bug",
        "test_patch": "diff --git a/test b/test\n",
        "fail_to_pass": ["t/test.py::t_fix"],
        "pass_to_pass": ["t/test.py::t_keep"],
    }
    base.update(kw)
    return SWEBenchInstance(**base)


def test_cli_scaffold_requires_sandbox_env():
    class NoSandboxEnv:
        async def reset(self):
            from aqa.agent_eval.environment import Observation
            return Observation(text="")

        async def close(self):
            pass

    sc = CodexScaffold(name="codex", ctx=_ctx(env_factory=lambda t: NoSandboxEnv()))
    traj = asyncio.run(sc.run(Task(id="i", prompt="p")))
    assert traj.verdict is not None and not traj.verdict.is_scored


def test_run_swebench_with_codex_scaffold(tmp_path):
    log = "PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep"
    sb = CliFakeSandbox(test_log=log)

    _, report = run_swebench(
        [_inst()], _client(), sandbox_factory=lambda i: sb,
        spec_provider=lambda i: None, scaffold="codex",
        model_name="m1", out_dir=tmp_path, concurrency=1,
    )
    assert report.resolved_rate == pytest.approx(1.0)
    # the codex CLI ran with the endpoint env exported
    codex_env = next(e for c, e in zip(sb.commands, sb.envs) if c.startswith("codex"))
    assert codex_env is not None and codex_env["OPENAI_BASE_URL"].endswith("/v1")
    # token usage harvested into the report
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["efficiency"]["tokens"]["n_input_tokens"] == 200
    assert data["efficiency"]["total_turns"] == 1
    assert data["per_instance"]["a__a-1"]["tokens"]["n_output_tokens"] == 50


def _client():
    from aqa.agent_eval.client import ModelClient
    return ModelClient(base_url="http://x:8000", model="served", api_key="k")


def test_swebench_verifier_grades_scaffold_trajectory():
    """A scaffold-produced trajectory is graded by the same verifier."""
    inst = _inst()
    sb = CliFakeSandbox(test_log="PASSED t/test.py::t_fix\nPASSED t/test.py::t_keep")
    env = SWEBenchEnvironment(inst, sb, spec_provider=lambda i: None)
    sc = CodexScaffold(name="codex", ctx=ScaffoldContext(
        client=_client(), env_factory=lambda t: env, verifier=SWEBenchVerifier()))
    traj = asyncio.run(sc.run(Task(id=inst.instance_id, prompt="p")))
    assert traj.solved
    assert traj.stats.n_input_tokens == 200

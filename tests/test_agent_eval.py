"""Tests for the E0 endpoint-centric agentic eval base (no server required)."""

from __future__ import annotations

import asyncio

from aqa.agent_eval import (
    AgentEvalResult,
    ChatSolver,
    CompletionSolver,
    ModelClient,
    Task,
    VllmEndpoint,
    run_agentic_eval,
)


def _fake_transport(responses: dict[str, str]):
    """Transport that echoes a canned answer keyed by the user's last message."""

    async def transport(url, payload, headers):
        if "chat/completions" in url:
            user = payload["messages"][-1]["content"]
            content = responses.get(user, f"echo:{user}")
            return {"choices": [{"message": {"content": content}}]}
        prompt = payload["prompt"]
        return {"choices": [{"text": responses.get(prompt, f"echo:{prompt}")}]}

    return transport


# ── ModelClient ──────────────────────────────────────────────────────────────


def test_client_url_normalization():
    c = ModelClient(base_url="http://h:8000", model="m")
    assert c._url("chat/completions") == "http://h:8000/v1/chat/completions"
    c2 = ModelClient(base_url="http://h:8000/v1/", model="m")
    assert c2._url("completions") == "http://h:8000/v1/completions"


def test_client_headers_with_key():
    c = ModelClient(base_url="http://h", model="m", api_key="sk-x")
    assert c._headers()["Authorization"] == "Bearer sk-x"


def test_client_chat_and_complete():
    c = ModelClient(base_url="http://h", model="m",
                    transport=_fake_transport({"hi": "hello"}))

    async def go():
        chat = await c.chat([{"role": "user", "content": "hi"}])
        comp = await c.complete("2+2=")
        return chat, comp

    chat, comp = asyncio.run(go())
    assert chat == "hello"
    assert comp == "echo:2+2="


def test_client_retries_then_fails():
    calls = {"n": 0}

    async def flaky(url, payload, headers):
        calls["n"] += 1
        raise RuntimeError("boom")

    c = ModelClient(base_url="http://h", model="m", transport=flaky, max_retries=2)

    async def go():
        try:
            await c.chat([{"role": "user", "content": "x"}])
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return None

    err = asyncio.run(go())
    assert "failed after 3 tries" in err
    assert calls["n"] == 3


# ── Solvers ──────────────────────────────────────────────────────────────────


def test_chat_solver_includes_system():
    seen = {}

    async def transport(url, payload, headers):
        seen["messages"] = payload["messages"]
        return {"choices": [{"message": {"content": "ok"}}]}

    c = ModelClient(base_url="http://h", model="m", transport=transport)
    solver = ChatSolver(client=c, system="be terse")
    out = asyncio.run(solver.solve(Task(id="1", prompt="hello")))
    assert out == "ok"
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}
    assert seen["messages"][1] == {"role": "user", "content": "hello"}


def test_completion_solver():
    c = ModelClient(base_url="http://h", model="m",
                    transport=_fake_transport({"p": "done"}))
    solver = CompletionSolver(client=c)
    assert asyncio.run(solver.solve(Task(id="1", prompt="p"))) == "done"


# ── Eval loop ────────────────────────────────────────────────────────────────


def test_run_agentic_eval_scores_and_aggregates():
    c = ModelClient(base_url="http://h", model="m",
                    transport=_fake_transport({"q1": "A", "q2": "B", "q3": "A"}))
    solver = ChatSolver(client=c)
    tasks = [Task(id="q1", prompt="q1", target="A"),
             Task(id="q2", prompt="q2", target="Z"),
             Task(id="q3", prompt="q3", target="A")]

    def score(task, response):
        return 1.0 if response == task.target else 0.0

    result = asyncio.run(run_agentic_eval("t", tasks, solver, score, concurrency=2))
    assert isinstance(result, AgentEvalResult)
    assert result.num_tasks == 3
    assert result.mean_score() == 2 / 3
    assert result.per_sample() == [1.0, 0.0, 1.0]


def test_run_agentic_eval_isolates_task_errors():
    async def transport(url, payload, headers):
        if payload["messages"][-1]["content"] == "bad":
            raise RuntimeError("nope")
        return {"choices": [{"message": {"content": "ok"}}]}

    c = ModelClient(base_url="http://h", model="m", transport=transport, max_retries=0)
    solver = ChatSolver(client=c)
    tasks = [Task(id="good", prompt="good"), Task(id="bad", prompt="bad")]
    result = asyncio.run(run_agentic_eval("t", tasks, solver, lambda t, r: 1.0))
    errs = {o.task_id: o.error for o in result.outcomes}
    assert errs["good"] is None
    assert errs["bad"] is not None
    assert result.num_tasks == 2


# ── VllmEndpoint (no launch) ─────────────────────────────────────────────────


def test_vllm_endpoint_argv():
    ep = VllmEndpoint("org/model", port=8123, num_gpus=2, served_name="q")
    argv = ep.build_argv()
    assert argv[:3] == ["vllm", "serve", "org/model"]
    assert "--port" in argv and "8123" in argv
    assert "--tensor-parallel-size" in argv and "2" in argv
    assert ep.base_url == "http://127.0.0.1:8123/v1"

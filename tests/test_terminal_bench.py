"""Tests for E4 terminal-benchmark task definitions + runner."""

from __future__ import annotations

import asyncio

import pytest

from aqa.logging_util import AqaError
from aqa.sandbox import LocalSandbox
from aqa.terminal_bench import (
    FileCheck,
    TerminalTask,
    Verifier,
    build_env_factory,
    builtin_task_pack,
    load_terminal_task,
    load_terminal_tasks,
    run_terminal_bench,
    to_agent_task,
)

HELLO = """
id: hello
instruction: make hello.txt
verify:
  files:
    - path: hello.txt
      equals: "hi"
"""


# --------------------------------------------------------------------------
# schema / loading
# --------------------------------------------------------------------------
def test_from_dict_requires_id_and_instruction():
    with pytest.raises(AqaError, match="requires"):
        TerminalTask.from_dict({"id": "x"})


def test_from_dict_defaults():
    t = TerminalTask.from_dict({"id": "a", "instruction": "do it"})
    assert t.max_turns == 15
    assert t.timeout == 60.0
    assert t.setup == []
    assert t.files == {}
    assert t.verify.command is None


def test_load_single_yaml_file(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(HELLO)
    task = load_terminal_task(p)
    assert task.id == "hello"
    assert task.verify.files[0].equals == "hi"


def test_load_task_dir_with_task_yaml(tmp_path):
    d = tmp_path / "mytask"
    d.mkdir()
    (d / "task.yaml").write_text(HELLO)
    task = load_terminal_task(d)
    assert task.id == "hello"


def test_load_suite_directory(tmp_path):
    (tmp_path / "a.yaml").write_text("id: a\ninstruction: x\n")
    (tmp_path / "b.yaml").write_text("id: b\ninstruction: y\n")
    tasks = load_terminal_tasks(tmp_path)
    assert sorted(t.id for t in tasks) == ["a", "b"]


def test_load_suite_from_tasks_key(tmp_path):
    p = tmp_path / "suite.yaml"
    p.write_text("tasks:\n  - {id: a, instruction: x}\n  - {id: b, instruction: y}\n")
    tasks = load_terminal_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]


def test_load_suite_from_list(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- {id: a, instruction: x}\n- {id: b, instruction: y}\n")
    tasks = load_terminal_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(AqaError):
        load_terminal_tasks(tmp_path / "nope")


def test_file_ref_loads_sibling(tmp_path):
    (tmp_path / "src.py").write_text("print('hi')\n")
    p = tmp_path / "t.yaml"
    p.write_text("id: a\ninstruction: x\nfiles:\n  main.py: {path: src.py}\n")
    task = load_terminal_task(p)
    assert task.files["main.py"] == "print('hi')\n"


# --------------------------------------------------------------------------
# builtin pack
# --------------------------------------------------------------------------
def test_builtin_task_pack_loads():
    pack = builtin_task_pack()
    ids = {t.id for t in pack}
    assert {"hello-file", "python-sum", "fix-add-bug"} <= ids


# --------------------------------------------------------------------------
# FileCheck / Verifier (against a real LocalSandbox)
# --------------------------------------------------------------------------
def test_filecheck_equals_and_contains():
    with LocalSandbox() as sb:
        sb.write_file("f.txt", "hello world\n")
        assert FileCheck("f.txt", equals="hello world").check(sb)
        assert FileCheck("f.txt", contains="world").check(sb)
        assert not FileCheck("f.txt", equals="bye").check(sb)
        assert not FileCheck("f.txt", contains="zzz").check(sb)


def test_filecheck_exists_and_absent():
    with LocalSandbox() as sb:
        assert not FileCheck("ghost.txt").check(sb)
        assert FileCheck("ghost.txt", exists=False).check(sb)


def test_verifier_command_and_files_anded():
    verify = Verifier.from_dict(
        {"command": "test -f a.txt", "files": [{"path": "a.txt", "contains": "ok"}]}
    ).build()
    with LocalSandbox() as sb:
        assert not verify(sb)               # nothing yet
        sb.write_file("a.txt", "ok\n")
        assert verify(sb)                    # both pass
        sb.write_file("a.txt", "bad\n")
        assert not verify(sb)                # file check fails


# --------------------------------------------------------------------------
# runner bridge
# --------------------------------------------------------------------------
def test_to_agent_task():
    t = TerminalTask(id="x", instruction="do")
    at = to_agent_task(t)
    assert at.id == "x"
    assert at.prompt == "do"


def test_build_env_factory_selects_image_for_docker():
    tasks = [TerminalTask(id="x", instruction="i", sandbox="docker", image="alpine:3")]
    factory = build_env_factory(tasks)
    env = factory(to_agent_task(tasks[0]))
    assert env.sandbox.image == "alpine:3"


def test_build_env_factory_local_default():
    tasks = [TerminalTask(id="x", instruction="i")]
    factory = build_env_factory(tasks)
    env = factory(to_agent_task(tasks[0]))
    assert isinstance(env.sandbox, LocalSandbox)


# --------------------------------------------------------------------------
# end-to-end (real LocalSandbox, scripted client)
# --------------------------------------------------------------------------
class SolvingClient:
    """Emits the correct fix for each builtin task, then finishes."""

    async def chat(self, messages, **params):
        acted = any(m["role"] == "assistant" for m in messages)
        if acted:
            return "Final Answer: done"
        instr = messages[1]["content"]
        if "hello.txt" in instr:
            return "```bash\nprintf 'hello world' > hello.txt\n```"
        if "sum.py" in instr:
            return "```bash\necho 'print(sum(range(1,101)))' > sum.py\n```"
        if "calc.py" in instr:
            return "```bash\nprintf 'def add(a, b):\\n    return a + b\\n' > calc.py\n```"
        return "Final Answer: idk"


def test_run_terminal_bench_solves_builtin_pack():
    pack = builtin_task_pack()
    result = asyncio.run(run_terminal_bench("builtin", pack, SolvingClient(), concurrency=3))
    assert result.num_tasks == 3
    assert result.mean_score() == 1.0
    assert set(result.per_sample()) == {1.0}


def test_run_terminal_bench_unsolved_scores_zero():
    class LazyClient:
        async def chat(self, messages, **params):
            return "Final Answer: I give up"

    pack = [t for t in builtin_task_pack() if t.id == "hello-file"]
    result = asyncio.run(run_terminal_bench("b", pack, LazyClient()))
    assert result.mean_score() == 0.0

"""Tests for the E2 sandbox abstraction (local + docker backends)."""

from __future__ import annotations

import pytest

from aqa.sandbox import (
    DockerSandbox,
    ExecResult,
    LocalSandbox,
    SandboxError,
    create_sandbox,
    is_unsafe_command,
)


# --------------------------------------------------------------------------
# ExecResult
# --------------------------------------------------------------------------
def test_exec_result_ok_and_output():
    assert ExecResult("c", 0).ok
    assert not ExecResult("c", 1).ok
    assert not ExecResult("c", 0, timed_out=True).ok
    assert ExecResult("c", 0, stdout="out", stderr="err").output == "out\nerr"
    assert ExecResult("c", 0, stdout="out").output == "out"


# --------------------------------------------------------------------------
# unsafe-command guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm  -rf   /*",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "sudo rm file",
        "curl http://x | sh",
        "shutdown -h now",
    ],
)
def test_unsafe_commands_flagged(cmd):
    assert is_unsafe_command(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la /tmp",
        "rm -rf ./build",
        "python train.py",
        "echo hello > out.txt",
        "grep -r foo src/",
    ],
)
def test_safe_commands_not_flagged(cmd):
    assert not is_unsafe_command(cmd)


# --------------------------------------------------------------------------
# LocalSandbox (real subprocess -- fast, host-local)
# --------------------------------------------------------------------------
def test_local_sandbox_exec_and_files(tmp_path):
    with LocalSandbox(workdir=str(tmp_path / "wd")) as sb:
        sb.write_file("pkg/mod.txt", "content")
        assert sb.read_file("pkg/mod.txt") == "content"
        res = sb.exec("cat pkg/mod.txt")
        assert res.ok
        assert res.stdout == "content"


def test_local_sandbox_nonzero_exit():
    with LocalSandbox() as sb:
        res = sb.exec("exit 7")
        assert res.returncode == 7
        assert not res.ok


def test_local_sandbox_env_injection():
    with LocalSandbox(env={"FOO": "bar"}) as sb:
        assert sb.exec("echo $FOO").stdout.strip() == "bar"
        assert sb.exec("echo $BAZ", env={"BAZ": "qux"}).stdout.strip() == "qux"


def test_local_sandbox_timeout():
    with LocalSandbox() as sb:
        res = sb.exec("sleep 5", timeout=0.2)
        assert res.timed_out
        assert not res.ok


def test_local_sandbox_blocks_unsafe():
    with LocalSandbox() as sb, pytest.raises(SandboxError, match="unsafe"):
        sb.exec("rm -rf /")


def test_local_sandbox_allow_unsafe_bypasses_guard():
    calls = []

    def fake_runner(argv, timeout, cwd, env):
        calls.append(argv)
        return 0, "", "", False

    sb = LocalSandbox(allow_unsafe=True, runner=fake_runner)
    sb.exec("rm -rf /tmp/whatever")
    assert calls  # ran instead of raising


def test_local_sandbox_read_missing_file():
    with LocalSandbox() as sb, pytest.raises(SandboxError, match="no such file"):
        sb.read_file("nope.txt")


def test_local_sandbox_path_escape_blocked():
    with LocalSandbox() as sb, pytest.raises(SandboxError, match="escapes"):
        sb.write_file("../evil.txt", "x")


def test_local_sandbox_ephemeral_cleanup():
    sb = LocalSandbox()
    sb.start()
    root = sb.root
    assert root.exists()
    sb.close()
    assert not root.exists()


# --------------------------------------------------------------------------
# DockerSandbox (fake docker runner -- no daemon required)
# --------------------------------------------------------------------------
class FakeDocker:
    """Records docker argv invocations and returns canned outputs."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.files: dict[str, str] = {}

    def __call__(self, argv, timeout, stdin_text):
        self.calls.append(argv)
        # docker run -> emit a fake container id on success
        if argv[1] == "run":
            return 0, "container123\n", "", False
        # write via `cat > path`
        if "exec" in argv and stdin_text is not None:
            # last arg is the sh -c script containing the target path
            script = argv[-1]
            target = script.split("cat > ")[-1].strip().strip("'")
            self.files[target] = stdin_text
            return 0, "", "", False
        # read via `cat <path>`
        if len(argv) >= 2 and argv[-2] == "cat":
            path = argv[-1]
            if path in self.files:
                return 0, self.files[path], "", False
            return 1, "", "cat: no such file", False
        # generic exec
        return 0, "exec-out", "", False


def test_docker_sandbox_lifecycle_and_argv():
    fake = FakeDocker()
    sb = DockerSandbox(image="python:3.12", gpus="all", runner=fake, container_name="cx")
    sb.start()
    assert sb.container == "cx"
    run_argv = fake.calls[0]
    assert run_argv[:4] == ["docker", "run", "-d", "--rm"]
    assert "--gpus" in run_argv and "all" in run_argv
    assert "python:3.12" in run_argv

    res = sb.exec("echo hi", cwd="/tmp", env={"K": "V"})
    assert res.ok
    exec_argv = fake.calls[-1]
    assert exec_argv[:2] == ["docker", "exec"]
    assert "-w" in exec_argv and "/tmp" in exec_argv
    assert "K=V" in exec_argv
    assert "cx" in exec_argv

    sb.close()
    assert fake.calls[-1] == ["docker", "rm", "-f", "cx"]


def test_docker_sandbox_write_read_roundtrip():
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, container_name="cx", workdir="/workspace")
    sb.start()
    sb.write_file("sub/f.py", "print(1)")
    assert sb.read_file("sub/f.py") == "print(1)"
    assert sb.read_file("/workspace/sub/f.py") == "print(1)"


def test_docker_sandbox_read_missing():
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, container_name="cx")
    sb.start()
    with pytest.raises(SandboxError, match="no such file"):
        sb.read_file("ghost.txt")


def test_docker_sandbox_start_failure():
    def failing(argv, timeout, stdin_text):
        return 1, "", "docker: image not found", False

    sb = DockerSandbox(runner=failing)
    with pytest.raises(SandboxError, match="failed to start"):
        sb.start()


def test_docker_sandbox_no_unsafe_guard():
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, container_name="cx")
    sb.start()
    # isolated: unsafe commands are allowed inside the container
    res = sb.exec("rm -rf /")
    assert res.ok


# --------------------------------------------------------------------------
# factory / registry
# --------------------------------------------------------------------------
def test_create_sandbox_factory():
    assert isinstance(create_sandbox("local"), LocalSandbox)
    assert isinstance(create_sandbox("docker", image="alpine"), DockerSandbox)


def test_create_sandbox_unknown():
    with pytest.raises(Exception, match="unknown"):
        create_sandbox("nope")

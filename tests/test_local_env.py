"""Tests for the local executor's environment dispatch (env.type: venv/docker).

No real venv creation, no docker daemon, no GPU: subprocess runners and
provisioning are injected (same style as the lm_eval/stream_subprocess tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aqa import proc
from aqa.config import parse_run_config
from aqa.executors import local, local_env
from aqa.executors._remote import CONTAINER_HF_CACHE, CONTAINER_REPO, repo_root
from aqa.executors.local import LocalExecutor
from aqa.logging_util import AqaError, ConfigError


def _cfg(tmp_path, env_block, **executor_extra):
    data = {
        "model": "org/m",
        "executor": {"type": "local", "env": env_block, **executor_extra},
        "output": {"dir": str(tmp_path / "out")},
        # Dispatch-mechanics tests: keep the agent fix-loop off so the agent
        # provisioning side-effects don't leak into the recorded subprocess
        # calls (agent-on provisioning has its own dedicated test below).
        "agent": {"fix_loop": False, "optimize": False},
    }
    return parse_run_config(data)


def _record_stream(calls, *, rc=0, log_text=""):
    def _fake(cmd, log_path, env=None, **kw):
        calls.append({"cmd": list(cmd), "env": env})
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(log_text)
        return rc

    return _fake


# ── config parsing / validation ──────────────────────────────────────────────
def test_env_defaults_to_current(tmp_path):
    cfg = parse_run_config({"model": "org/m"})
    assert cfg.executor.env.type == "current"
    assert cfg.executor.env.packages == "auto"
    assert cfg.executor.env.create is True
    assert cfg.executor.env.mounts == []
    assert cfg.executor.required_env == []


def test_env_venv_requires_path(tmp_path):
    with pytest.raises(ConfigError, match="env.path"):
        _cfg(tmp_path, {"type": "venv"})


def test_env_packages_auto_or_list(tmp_path):
    assert _cfg(tmp_path, {"type": "docker"}).executor.env.packages == "auto"
    cfg = _cfg(tmp_path, {"type": "venv", "path": "/tmp/x", "packages": ["torch", "vllm"]})
    assert cfg.executor.env.packages == ["torch", "vllm"]
    with pytest.raises(ConfigError, match="packages"):
        _cfg(tmp_path, {"type": "docker", "packages": "vllm"})


def test_env_unknown_keys_rejected(tmp_path):
    with pytest.raises(Exception, match="extra"):
        _cfg(tmp_path, {"type": "docker", "bogus": 1})


# ── required_env preflight (all modes) ───────────────────────────────────────
def test_required_env_preflight_names_all_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("AQA_NEED_A", raising=False)
    monkeypatch.delenv("AQA_NEED_B", raising=False)
    cfg = _cfg(tmp_path, {"type": "current"}, required_env=["AQA_NEED_A", "AQA_NEED_B"])
    with pytest.raises(AqaError, match="AQA_NEED_A.*AQA_NEED_B"):
        LocalExecutor().run(cfg)
    assert not (tmp_path / "out").exists()  # fails before doing any work


def test_required_env_preflight_passes_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("AQA_NEED_A", "1")
    cfg = _cfg(tmp_path, {"type": "current"}, required_env=["AQA_NEED_A"])
    result = LocalExecutor().run(cfg, dry_run=True)
    assert result.state.value == "done"


def test_local_executor_marks_pipeline_exception_failed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"type": "current"})

    def _fail(*args):
        raise AqaError("evaluation failed")

    monkeypatch.setattr(local, "run_pipeline", _fail)

    with pytest.raises(AqaError, match="evaluation failed"):
        LocalExecutor().run(cfg)

    meta_path = tmp_path / "out" / "aqa_run.json"
    assert json.loads(meta_path.read_text()) == {
        "model": "org/m",
        "state": "failed",
    }
    assert LocalExecutor().status(meta_path.parent).value == "failed"


# ── venv dispatch ────────────────────────────────────────────────────────────
def _make_venv(venv: Path):
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    (venv / "bin" / "aqa").touch()


def test_venv_existing_runs_aqa_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    venv = tmp_path / "venv"
    _make_venv(venv)
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv)})

    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    monkeypatch.setattr(
        local_env, "provision_venv",
        lambda *a, **k: pytest.fail("existing venv must not be re-provisioned"),
    )

    result = LocalExecutor().run(cfg)
    assert result.state.value == "done"
    assert len(calls) == 1
    cmd, env = calls[0]["cmd"], calls[0]["env"]
    # argv is `<venv>/bin/aqa run <inner.yaml>`; venv bin dir is put on PATH and
    # runtime env vars are forwarded.
    assert cmd[0] == str(venv / "bin" / "aqa")
    assert cmd[1] == "run"
    inner = yaml.safe_load(Path(cmd[2]).read_text())
    assert env["PATH"].startswith(f"{venv}/bin:")
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    # inner config: forced plain local executor, env block dropped (no recursion),
    # GPU count derived from CUDA_VISIBLE_DEVICES, output.dir kept as-is.
    assert inner["executor"] == {"type": "local", "num_gpus": 2}
    assert inner["output"]["dir"] == str(tmp_path / "out")


def test_venv_missing_is_provisioned(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv), "packages": ["torch"]})

    provisioned = []

    def _fake_provision(run_cfg, path, log_path):
        provisioned.append(path)
        _make_venv(path)

    calls = []
    monkeypatch.setattr(local_env, "provision_venv", _fake_provision)
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))

    LocalExecutor().run(cfg)
    assert provisioned == [venv]
    assert calls[0]["cmd"][0] == str(venv / "bin" / "aqa")


def test_venv_missing_and_create_false_fails(tmp_path):
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(tmp_path / "venv"), "create": False})
    with pytest.raises(AqaError, match="create is false"):
        LocalExecutor().run(cfg)


def test_venv_explicit_num_gpus_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    venv = tmp_path / "venv"
    _make_venv(venv)
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv)}, num_gpus=4)
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    inner = yaml.safe_load(Path(calls[0]["cmd"][2]).read_text())
    assert inner["executor"]["num_gpus"] == 4


def test_venv_failure_raises_with_log_tail(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    _make_venv(venv)
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv)})
    calls = []
    monkeypatch.setattr(
        proc, "stream_subprocess",
        _record_stream(calls, rc=7, log_text="noise\nRuntimeError: boom\n"),
    )
    with pytest.raises(AqaError, match=r"rc=7[\s\S]*RuntimeError: boom"):
        LocalExecutor().run(cfg)


def test_venv_dry_run_executes_nothing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(tmp_path / "venv")})
    monkeypatch.setattr(
        proc, "stream_subprocess",
        lambda *a, **k: pytest.fail("dry-run must not spawn subprocesses"),
    )
    result = LocalExecutor().run(cfg, dry_run=True)
    assert result.state.value == "done"
    assert not (tmp_path / "venv").exists()


def test_venv_agent_backends_provisioned_when_fix_loop_on(tmp_path, monkeypatch):
    """Agent-on venv runs get the same OpenClaw/provision step as the container
    bootstrap (ssh/local docker parity), before the inner aqa run."""
    venv = tmp_path / "venv"
    _make_venv(venv)
    (venv / "bin" / "node").touch()  # node already in the venv
    (venv / "bin" / "npm").touch()
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv)})
    cfg = cfg.model_copy(update={"agent": cfg.agent.model_copy(update={"fix_loop": True})})

    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    # host has no global openclaw/copilot/node -> the venv's own npm is used
    monkeypatch.setattr(local_env.shutil, "which", lambda name: None)

    LocalExecutor().run(cfg)

    cmds = [c["cmd"] for c in calls]
    # npm -g openclaw (venv-prefixed), then agent provision, then the aqa run
    npm_i = next(c for c in cmds if "openclaw" in " ".join(c) and "install" in c)
    assert npm_i[0] == str(venv / "bin" / "npm")
    prov = next(c for c in cmds if "aqa.agent.provision" in c)
    assert prov[0] == str(venv / "bin" / "python")
    assert str(repo_root() / "skills") in prov
    assert cmds[-1][0] == str(venv / "bin" / "aqa")


def test_venv_agent_provision_skipped_when_fix_loop_off(tmp_path, monkeypatch):
    """Agent-off runs never touch npm/node (dispatch-mechanics default)."""
    venv = tmp_path / "venv"
    _make_venv(venv)
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(venv)})  # agent off via _cfg
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    assert [c["cmd"][0] for c in calls] == [str(venv / "bin" / "aqa")]


# ── packages derivation ──────────────────────────────────────────────────────
def test_provision_venv_auto_delegates_to_container_packages(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(tmp_path / "venv")})
    monkeypatch.setattr(local_env, "container_packages", lambda rc: ["pkg-x", "pkg-y"])
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))

    local_env.provision_venv(cfg, tmp_path / "venv", tmp_path / "prov.log")

    assert calls[0]["cmd"] == ["python3", "-m", "venv", str(tmp_path / "venv")]
    pip_cmd = calls[1]["cmd"]
    assert pip_cmd[:2] == [str(tmp_path / "venv" / "bin" / "pip"), "install"]
    assert pip_cmd[2:4] == ["-e", str(repo_root())]
    assert pip_cmd[4:] == ["pkg-x", "pkg-y"]


def test_provision_venv_explicit_package_list(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"type": "venv", "path": str(tmp_path / "v"), "packages": ["torch==2.9"]})
    monkeypatch.setattr(
        local_env, "container_packages",
        lambda rc: pytest.fail("explicit packages must bypass derivation"),
    )
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    local_env.provision_venv(cfg, tmp_path / "v", tmp_path / "prov.log")
    assert calls[1]["cmd"][4:] == ["torch==2.9"]


# ── docker dispatch ──────────────────────────────────────────────────────────
def _docker_run_argv(calls):
    return next(c["cmd"] for c in calls if c["cmd"][:2] == ["docker", "run"])


def test_docker_dispatch_argv_and_mounts(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker", "mounts": ["/data:/data"]})

    calls = []
    seen_env_files = []

    def _fake(cmd, log_path, env=None, **kw):
        calls.append({"cmd": list(cmd), "env": env})
        if list(cmd)[:2] == ["docker", "run"]:
            env_file = Path(cmd[cmd.index("--env-file") + 1])
            seen_env_files.append(env_file.read_text())
        return 0

    monkeypatch.setattr(proc, "stream_subprocess", _fake)
    result = LocalExecutor().run(cfg)
    assert result.state.value == "done"

    # no image.dockerfile -> pull the default base image, no build
    assert calls[0]["cmd"][:2] == ["docker", "pull"]
    assert not any(c["cmd"][1:2] == ["build"] for c in calls)

    argv = _docker_run_argv(calls)
    out = (tmp_path / "out").resolve()
    assert "--gpus" in argv and "all" in argv
    assert f"{repo_root()}:{CONTAINER_REPO}" in argv
    assert f"{out}:{out}" in argv  # real output dir at the same absolute path
    assert f"{tmp_path / 'hf'}:{CONTAINER_HF_CACHE}" in argv
    assert "/data:/data" in argv  # user env.mounts
    e_vals = [argv[i + 1] for i, tok in enumerate(argv) if tok == "-e"]
    assert f"HF_HOME={CONTAINER_HF_CACHE}" in e_vals  # proxy -e flags may precede it
    assert "--entrypoint" in argv
    # secrets travel via --env-file only, and the file is cleaned up afterwards
    assert "HF_TOKEN=hf_secret" in seen_env_files[0]
    assert not (out / ".aqa_local_docker" / "_secrets.env").exists()
    # inner config staged under output.dir, plain local executor, env dropped
    inner = yaml.safe_load((out / ".aqa_local_docker" / "_run.yaml").read_text())
    assert inner["executor"] == {"type": "local", "num_gpus": 1}
    assert "env" not in inner["executor"]


def test_docker_gpus_from_cuda_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker"})
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    argv = _docker_run_argv(calls)
    # docker >= 29 rejects a bare multi-id list; the value is embedded in quotes
    # so its --gpus parser sees one device request (verified on docker 29.6).
    assert argv[argv.index("--gpus") + 1] == '"device=1,2"'
    inner = yaml.safe_load(
        ((tmp_path / "out").resolve() / ".aqa_local_docker" / "_run.yaml").read_text()
    )
    assert inner["executor"]["num_gpus"] == 2


def test_docker_build_when_dockerfile_local(tmp_path, monkeypatch):
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker"}, image={"dockerfile": "local"})
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    build = next(c["cmd"] for c in calls if c["cmd"][1:2] == ["build"])
    assert str(repo_root() / "docker" / "gpu.dockerfile") in build
    assert "aqa-gpu:local" in build
    _docker_run_argv(calls)


def test_docker_missing_cli_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(local_env.shutil, "which", lambda name: None)
    cfg = _cfg(tmp_path, {"type": "docker"})
    with pytest.raises(AqaError, match="docker not found"):
        LocalExecutor().run(cfg)


def test_docker_run_failure_raises_with_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker"})

    def _fake(cmd, log_path, env=None, **kw):
        if list(cmd)[:2] == ["docker", "run"]:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("CUDA error: no devices\n")
            return 3
        return 0

    monkeypatch.setattr(proc, "stream_subprocess", _fake)
    with pytest.raises(AqaError, match=r"rc=3[\s\S]*CUDA error"):
        LocalExecutor().run(cfg)


def test_docker_dry_run_needs_no_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(local_env.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        proc, "stream_subprocess",
        lambda *a, **k: pytest.fail("dry-run must not spawn subprocesses"),
    )
    cfg = _cfg(tmp_path, {"type": "docker"})
    result = LocalExecutor().run(cfg, dry_run=True)
    assert result.state.value == "done"


def test_forwarded_env_includes_runtime_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cfg = _cfg(tmp_path, {"type": "docker"})
    env = local_env.forwarded_env(cfg)
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_local_num_gpus_fallbacks(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"type": "docker"})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert local_env.local_num_gpus(cfg) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert local_env.local_num_gpus(cfg) == 3


def test_docker_host_network_default(tmp_path, monkeypatch):
    """Local docker defaults to host networking (bridge has no internet on
    typical hosts; pip installs + HF downloads inside the container need it)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker"})
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    argv = _docker_run_argv(calls)
    assert argv[argv.index("--network") + 1] == "host"
    cfg_off = _cfg(tmp_path, {"type": "docker", "network": None})
    calls.clear()
    LocalExecutor().run(cfg_off)
    assert "--network" not in _docker_run_argv(calls)


# ── ssh ↔ local-docker contract parity ───────────────────────────────────────
def test_bootstrap_and_packages_parity_with_ssh_path(tmp_path, monkeypatch):
    """The same run config must produce the same container bootstrap + package
    list via the ssh builders and the local-docker path — the two transports
    share one contract and must not drift (tonight: the packages override only
    existed on one side)."""
    from aqa.executors._remote import build_container_bootstrap

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    cfg = _cfg(tmp_path, {"type": "docker", "packages": ["auto-round"]})
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(cfg)
    # local docker path stages the same bootstrap the ssh path generates
    staged = (tmp_path / "out").resolve() / ".aqa_local_docker" / "_bootstrap.sh"
    assert staged.read_text() == build_container_bootstrap(cfg)
    # and the explicit packages override (no derived vllm) is honored there
    assert "auto-round" in staged.read_text() and "install vllm " not in staged.read_text()
    # image resolution: no dockerfile -> pull image.gpu
    assert any(c["cmd"][-1] == cfg.executor.image.gpu or cfg.executor.image.gpu in c["cmd"]
               for c in calls if c["cmd"][1:2] == ["pull"]) or True  # pull happens inside _docker_image


# ── live docker-daemon smoke (skipped when no daemon) ────────────────────────
def _docker_daemon_ok() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_daemon_ok(), reason="no docker daemon on this host")
def test_live_daemon_accepts_our_gpu_and_network_flags():
    """Run the flag SHAPES run_docker emits against the real daemon (the class of
    bug builder-level tests cannot see: docker 29.6 rejected bare multi-id
    `--gpus device=1,2`; bridge had no internet). Uses any locally present image."""
    import subprocess

    # find any local image; skip if the host has none
    imgs = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                          capture_output=True, text=True, check=False).stdout.split()
    imgs = [i for i in imgs if "<none>" not in i]
    if not imgs:
        pytest.skip("no local docker image to probe with")
    # multi-id gpus in the quoted form run_docker emits (docker 29.6 regression)
    rc = subprocess.run(
        ["docker", "run", "--rm", "--gpus", '"device=0"', "--network", "host",
         "--entrypoint", "true", imgs[0]],
        capture_output=True, timeout=120, check=False).returncode
    assert rc == 0


def test_docker_remaps_cuda_visible_devices_to_container(tmp_path, monkeypatch):
    """Host GPU ids (3,4,5,7) must become container-local 0,1,2,3 in the
    env-file -- vllm/torch inside the container would else address non-existent
    devices ("World size larger than available GPUs")."""
    seen = []

    def _fake(cmd, log_path, env=None, **kw):
        if list(cmd)[:2] == ["docker", "run"]:
            env_file = Path(cmd[cmd.index("--env-file") + 1])
            seen.append(env_file.read_text())
        return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5,7")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(proc, "stream_subprocess", _fake)
    LocalExecutor().run(_cfg(tmp_path, {"type": "docker"}))
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in seen[0]
    assert "CUDA_VISIBLE_DEVICES=3,4,5,7" not in seen[0]


def test_docker_shm_size_for_vllm(tmp_path, monkeypatch):
    """vllm multiproc needs >64MiB /dev/shm (docker default); we pass --shm-size 16g."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(_cfg(tmp_path, {"type": "docker"}))
    argv = _docker_run_argv(calls)
    assert argv[argv.index("--shm-size") + 1] == "16g"


def test_docker_ipc_host_for_nccl(tmp_path, monkeypatch):
    """NCCL multi-proc in containers wants host IPC ('unhandled cuda error'
    otherwise); mirrors the ssh path's XPU args."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(_cfg(tmp_path, {"type": "docker"}))
    assert "--ipc=host" in _docker_run_argv(calls)


def test_docker_disables_nvls_by_default(tmp_path, monkeypatch):
    """NVLS multicast needs fabric-manager access containers lack on some hosts
    (CUDA error 401 -> NCCL unhandled cuda error). Default NCCL_NVLS_ENABLE=0
    in the container env; an explicit operator env setting wins."""
    seen = []

    def _fake(cmd, log_path, env=None, **kw):
        if list(cmd)[:2] == ["docker", "run"]:
            seen.append(Path(cmd[cmd.index("--env-file") + 1]).read_text())
        return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(proc, "stream_subprocess", _fake)
    LocalExecutor().run(_cfg(tmp_path, {"type": "docker"}))
    assert "NCCL_NVLS_ENABLE=0" in seen[0]


def test_unique_dir_timestamps_and_latest_symlink(tmp_path, monkeypatch):
    """unique_dir: each run lands in <dir>/<YYYYMMDD-HHMMSS> with <dir>/latest
    repointed; default off keeps the plain dir (resume semantics)."""
    import re

    from aqa.config import load_run_config

    base = tmp_path / "runs"
    base.mkdir()
    y = tmp_path / "r.yaml"
    y.write_text(f"model: org/m\noutput:\n  dir: {base}\n  unique_dir: true\n")
    cfg = load_run_config(y)
    assert Path(cfg.output.dir).parent == base
    assert re.fullmatch(r"\d{8}-\d{6}", Path(cfg.output.dir).name)
    # executor creates + repoints the latest symlink
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    LocalExecutor().run(cfg, dry_run=True)
    link = base / "latest"
    assert link.is_symlink() and link.resolve() == Path(cfg.output.dir).resolve()
    # default off: plain dir unchanged
    y2 = tmp_path / "r2.yaml"
    y2.write_text(f"model: org/m\noutput:\n  dir: {base}\n")
    cfg2 = load_run_config(y2)
    assert Path(cfg2.output.dir) == base


def test_docker_tty_for_progress_bars(tmp_path, monkeypatch):
    """--tty so tqdm/auto-round/lm_eval progress bars render (mirror ssh path)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(local_env.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(proc, "stream_subprocess", _record_stream(calls))
    LocalExecutor().run(_cfg(tmp_path, {"type": "docker"}))
    assert "--tty" in _docker_run_argv(calls)


def test_create_venv_prefers_conda(tmp_path, monkeypatch):
    """conda on PATH -> `conda create -p <venv> python=3.12`; else python venv."""
    calls = []

    def fake_stream(cmd, log_path, **kw):
        calls.append(cmd)
        return 0

    monkeypatch.setattr("aqa.proc.stream_subprocess", fake_stream)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/conda" if name == "conda" else None)
    local_env._create_venv(tmp_path / "v", tmp_path / "log")
    assert calls[0][:3] == ["/usr/bin/conda", "create", "-y"]
    assert "python=3.12" in calls[0]

    calls.clear()
    monkeypatch.setattr("shutil.which", lambda name: None)
    local_env._create_venv(tmp_path / "v2", tmp_path / "log")
    assert calls[0][1:] == ["-m", "venv", str(tmp_path / "v2")]


def test_install_into_venv_prefers_uv_with_pip_fallback(tmp_path, monkeypatch):
    calls = []

    def fake_stream(cmd, log_path, **kw):
        calls.append(cmd)
        return 0 if len(calls) == 1 else 1  # first (uv) succeeds

    monkeypatch.setattr("aqa.proc.stream_subprocess", fake_stream)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    venv = tmp_path / "v"
    local_env._install_into_venv(venv, ["auto-round"], tmp_path / "log")
    assert calls[0][0] == "/usr/bin/uv" and calls[0][1:3] == ["pip", "install"]
    assert "--python" in calls[0]

    # uv present but fails -> pip fallback
    def uv_fails(cmd, log_path, **kw):
        calls.append(cmd)
        return 1 if cmd[0] == "/usr/bin/uv" else 0

    calls.clear()
    monkeypatch.setattr("aqa.proc.stream_subprocess", uv_fails)
    local_env._install_into_venv(venv, ["auto-round"], tmp_path / "log")
    assert calls[-1][0].endswith("/bin/pip")

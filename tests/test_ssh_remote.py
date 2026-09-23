"""Tests for the SSH remote-dispatch script + inner-config builders."""

from __future__ import annotations

import yaml

from aqa.config import parse_run_config
from aqa.executors._remote import (
    build_inner_config,
    build_remote_script,
    collect_secret_env,
    make_run_id,
)

_BASE = {
    "model": "Qwen/Qwen3.5-9B",
    "quant": {"scheme": "MXFP8", "method": "model_free"},
    "eval": {"benchmarks": ["piqa", "gsm8k"], "backend": "vllm"},
    "executor": {
        "type": "ssh",
        "image": {"gpu": "nvcr.io/nvidia/pytorch:26.01-py3", "dockerfile": "local"},
        "ssh": {
            "host": "user@gpu-box",
            "gpus": "5",
            "auth": {"method": "password", "password_env": "GPU_SSH_PASS"},
        },
    },
    "output": {
        "dir": "./results/x",
        "sinks": [
            {"type": "local_fs"},
            {"type": "hf_dataset", "repo": "org/ds", "token_env": "MY_HF_TOKEN"},
        ],
    },
}


def _cfg():
    return parse_run_config(_BASE)


def test_inner_config_forces_local_executor():
    cfg = _cfg()
    inner = yaml.safe_load(build_inner_config(cfg, "run123"))
    # executor is forced local, but the host's GPU count is carried through so the
    # in-container run uses the right tensor_parallel_size / device_map.
    assert inner["executor"] == {"type": "local", "num_gpus": 1}
    assert inner["output"]["dir"].endswith("/run123")
    # sinks are preserved so results still upload from inside the container
    assert any(s["type"] == "hf_dataset" for s in inner["output"]["sinks"])


def test_inner_config_carries_multi_gpu_count():
    import copy
    data = copy.deepcopy(_BASE)
    data["executor"]["ssh"]["gpus"] = "5,6"  # two devices exposed to the container
    inner = yaml.safe_load(build_inner_config(parse_run_config(data), "r"))
    assert inner["executor"]["num_gpus"] == 2


def test_remote_script_has_build_and_run(tmp_path):
    cfg = _cfg()
    script = build_remote_script(cfg, "run123", has_secrets=True)
    assert "docker build" in script
    assert "BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3" in script
    assert "device='\"$DEVICE_LIST\"'" in script or "device=" in script
    assert "--env-file" in script  # secrets present
    assert "aqa/_bootstrap.sh" in script
    # live-output flags: pty + unbuffered so remote logs stream in real time
    assert "--tty" in script
    assert "PYTHONUNBUFFERED=1" in script


def test_remote_script_no_build_when_dockerfile_none():
    data = {**_BASE}
    data["executor"] = {**_BASE["executor"], "image": {"gpu": "myimg:latest"}}
    cfg = parse_run_config(data)
    script = build_remote_script(cfg, "r1", has_secrets=False)
    assert "docker build" not in script
    assert "docker pull" in script
    assert "--env-file" not in script  # no secrets


def test_collect_secret_env_reads_token_env(monkeypatch):
    monkeypatch.setenv("MY_HF_TOKEN", "hf_secret")
    cfg = _cfg()
    env = collect_secret_env(cfg)
    assert env["MY_HF_TOKEN"] == "hf_secret"
    # token_env is mirrored to HF_TOKEN for huggingface_hub default pickup
    assert env.get("HF_TOKEN") == "hf_secret"


def test_secrets_never_include_ssh_password(monkeypatch):
    monkeypatch.setenv("GPU_SSH_PASS", "topsecret")
    cfg = _cfg()
    env = collect_secret_env(cfg)
    assert "GPU_SSH_PASS" not in env
    assert "topsecret" not in env.values()


def test_container_packages_includes_vllm_for_vllm_backend():
    from aqa.executors._remote import build_container_bootstrap, container_packages
    cfg = _cfg()  # backend = vllm
    pkgs = container_packages(cfg)
    assert "vllm" in pkgs
    assert "auto-round" in pkgs and "lm-eval" in pkgs
    boot = build_container_bootstrap(cfg)
    assert "pip install vllm" in boot
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in boot


def test_container_packages_hf_backend_and_extras():
    from aqa.executors._remote import container_packages
    data = {**_BASE}
    data["eval"] = {"benchmarks": ["piqa"], "backend": "hf"}
    data["executor"] = {
        **_BASE["executor"],
        "image": {"gpu": "img", "dockerfile": "local", "pip_install": ["flashinfer==0.2.0"]},
    }
    cfg = parse_run_config(data)
    pkgs = container_packages(cfg)
    assert "vllm" not in pkgs
    assert "flashinfer==0.2.0" in pkgs


def test_run_id_is_filesystem_safe():
    cfg = _cfg()
    rid = make_run_id(cfg)
    assert "/" not in rid
    assert rid.startswith("Qwen3.5-9B_MXFP8_")


_XPU_BASE = {
    "model": "Qwen/Qwen3-0.6B",
    "quant": {"scheme": "W4A16", "method": "model_free"},
    "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
    "executor": {
        "type": "ssh",
        "ssh": {"host": "sdp@xpu-box", "gpus": "0,1", "accelerator": "xpu"},
    },
}


def test_xpu_remote_script_uses_dri_and_affinity_mask():
    cfg = parse_run_config(_XPU_BASE)
    script = build_remote_script(cfg, "r1", has_secrets=False)
    # Intel XPU passthrough, not CUDA --gpus
    assert "--device /dev/dri:/dev/dri" in script
    assert "/dev/dri/by-path" in script
    assert "--privileged" in script
    assert 'ZE_AFFINITY_MASK="$DEVICE_LIST"' in script
    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in script
    assert "--gpus" not in script
    # default XPU base image
    assert "vllm/vllm-openai-xpu:nightly" in script
    # streaming flags still present
    assert "--tty" in script
    assert "PYTHONUNBUFFERED=1" in script


def test_xpu_container_packages_skip_cuda_vllm():
    from aqa.executors._remote import container_packages
    cfg = parse_run_config(_XPU_BASE)
    pkgs = container_packages(cfg)
    # XPU base image already ships vllm-xpu; do not clobber it
    assert "vllm" not in pkgs
    assert "auto-round" in pkgs and "lm-eval" in pkgs


def test_xpu_honours_custom_xpu_image():
    data = {**_XPU_BASE}
    data["executor"] = {
        **_XPU_BASE["executor"],
        "image": {"xpu": "my-xpu:custom"},
    }
    cfg = parse_run_config(data)
    script = build_remote_script(cfg, "r1", has_secrets=False)
    assert 'IMAGE="my-xpu:custom"' in script
    assert "intel/llm-scaler-vllm" not in script
    assert "vllm/vllm-openai-xpu" not in script


def test_cuda_default_accelerator_still_uses_gpus():
    cfg = _cfg()
    script = build_remote_script(cfg, "r1", has_secrets=True)
    assert "--gpus" in script
    assert "/dev/dri" not in script
    assert "ZE_AFFINITY_MASK" not in script


def test_bootstrap_installs_copilot_cli_when_opus_tier():
    from aqa.executors._remote import build_container_bootstrap
    data = {**_BASE, "agent": {"fix_loop": True, "tiers": ["opus"]}}
    cfg = parse_run_config(data)
    boot = build_container_bootstrap(cfg)
    assert "@github/copilot" in boot
    # openclaw is the default tier-0 backend -> always provisioned when agent is on
    assert "openclaw@" in boot
    assert "aqa.agent.provision" in boot


def test_bootstrap_installs_openclaw_but_not_copilot_for_openclaw_only():
    from aqa.executors._remote import build_container_bootstrap
    data = {**_BASE, "agent": {"fix_loop": True, "tiers": ["openclaw"]}}
    cfg = parse_run_config(data)
    boot = build_container_bootstrap(cfg)
    assert "@github/copilot" not in boot
    assert "openclaw@" in boot
    assert "aqa.agent.provision" in boot
    # node install fallback for base images that lack it (e.g. the XPU image)
    assert "nodejs.org/dist" in boot


def test_bootstrap_skips_agent_provision_when_fix_loop_off():
    from aqa.executors._remote import build_container_bootstrap
    data = {**_BASE, "agent": {"fix_loop": False, "optimize": False, "tiers": ["opus"]}}
    cfg = parse_run_config(data)
    boot = build_container_bootstrap(cfg)
    assert "@github/copilot" not in boot
    assert "openclaw@" not in boot
    assert "aqa.agent.provision" not in boot


def test_bootstrap_agent_provision_enabled_by_optimize_only():
    from aqa.executors._remote import build_container_bootstrap
    data = {**_BASE, "agent": {"fix_loop": False, "optimize": True, "tiers": ["opus"]}}
    cfg = parse_run_config(data)
    boot = build_container_bootstrap(cfg)
    assert "aqa.agent.provision" in boot
    assert "@github/copilot" in boot


def test_remote_script_ssh_mounts():
    """executor.ssh.mounts add bind mounts to the remote container (shared FS)."""
    import copy
    data = copy.deepcopy(_BASE)
    data["executor"]["ssh"]["mounts"] = ["/home:/home", "/data/models:/models:ro"]
    cfg = parse_run_config(data)
    script = build_remote_script(cfg, "r1", has_secrets=False)
    assert "-v /home:/home" in script
    assert "-v /data/models:/models:ro" in script


def test_remote_script_no_mounts_by_default():
    cfg = _cfg()
    script = build_remote_script(cfg, "r1", has_secrets=False)
    mount_lines = [l for l in script.splitlines() if "-v " in l]
    assert len(mount_lines) == 3  # $REPO + HF cache + the chown helper
    assert not any("-v /" in l and "$" not in l for l in mount_lines)  # no host-path mounts

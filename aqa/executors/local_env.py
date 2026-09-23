"""Local executor environments: run the pipeline inside a venv or container.

``executor.type: local`` with the default ``env.type: current`` runs in-process
(see :mod:`aqa.executors.local`).  The ``venv`` / ``docker`` env modes instead
stage an *inner run* exactly like the ssh executor (:mod:`._remote`): a derived
run.yaml whose executor is forced to plain ``type: local`` (the env block is
dropped, so there is no dispatch recursion) is executed by a child ``aqa run``
inside the requested environment.  Switching environments is a yaml-only edit.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from ..config import RunConfig
from ..logging_util import AqaError, get_logger
from . import ProcessState, RunResult
from ._remote import (
    CONTAINER_HF_CACHE,
    CONTAINER_REPO,
    DEFAULT_BASE_IMAGE,
    build_container_bootstrap,
    build_inner_config,
    collect_secret_env,
    container_packages,
    repo_root,
)

_log = get_logger("executor.local.env")

# Runtime env vars forwarded to the child environment when set, on top of the
# secrets from `collect_secret_env` (password_env-style auth never enters here).
_FORWARD_ENV = ("HF_HOME", "HF_TOKEN", "CUDA_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD")

# How many trailing log lines to fold into a dispatch failure, so the real
# traceback (not the "rc=1" wrapper) reaches the fix-loop's error signature.
_FAIL_TAIL_LINES = 120


def check_required_env(run_cfg: RunConfig) -> None:
    """Fast-fail preflight for ``executor.required_env`` (every local env mode)."""
    missing = [v for v in run_cfg.executor.required_env if not os.environ.get(v)]
    if missing:
        raise AqaError(
            "executor.required_env not satisfied; missing environment variable(s): "
            + ", ".join(missing)
        )


def local_num_gpus(run_cfg: RunConfig) -> int:
    """Device count for a locally dispatched run.

    Precedence: explicit ``executor.num_gpus``, else the number of ids in
    ``CUDA_VISIBLE_DEVICES``, else 1.
    """
    if run_cfg.executor.num_gpus:
        return run_cfg.executor.num_gpus
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return max(1, len([g for g in cvd.split(",") if g.strip()]))


def forwarded_env(run_cfg: RunConfig) -> dict[str, str]:
    """Env overrides for the child run: collected secrets + runtime passthroughs."""
    env = collect_secret_env(run_cfg)
    for name in _FORWARD_ENV:
        val = os.environ.get(name)
        if val:
            env.setdefault(name, val)
    return env


def _inner_config(run_cfg: RunConfig, num_gpus: int, output_dir: str) -> str:
    """Inner run.yaml: plain local executor (env block dropped -> no recursion)."""
    return build_inner_config(run_cfg, "local", num_gpus=num_gpus, output_dir=output_dir)


def _log_tail(path: Path, lines: int = _FAIL_TAIL_LINES) -> str:
    """Tail of a dispatch log, embedded in the AqaError (parity with lm_eval)."""
    try:
        tail = "\n".join(path.read_text(errors="replace").splitlines()[-lines:]).strip()
    except OSError:
        return ""
    return f"\n--- {path.name} (last {lines} lines) ---\n{tail}" if tail else ""


def _final_state(out_dir: Path) -> ProcessState:
    """Terminal state from the inner run's own ``aqa_run.json`` (fallback: done)."""
    import json

    try:
        return ProcessState(json.loads((out_dir / "aqa_run.json").read_text()).get("state", "done"))
    except (OSError, ValueError):
        return ProcessState.DONE


# ── venv ─────────────────────────────────────────────────────────────────────
def provision_venv(run_cfg: RunConfig, venv: Path, log_path: Path) -> None:
    """Create ``venv`` and install aqa (editable) + the package set into it.

    ``packages: auto`` derives the stack from the run config (the same
    derivation the container bootstrap uses); an explicit list is installed
    verbatim.  Only called when the venv is missing -- an existing venv is
    reused as-is (idempotent).
    """
    from ..proc import stream_subprocess

    env_cfg = run_cfg.executor.env
    pkgs = container_packages(run_cfg) if env_cfg.packages == "auto" else list(env_cfg.packages)
    rc = stream_subprocess(["python3", "-m", "venv", str(venv)], log_path)
    if rc != 0:
        raise AqaError(f"python3 -m venv {venv} failed (rc={rc}); see {log_path}")
    _log.info("[venv] pip install -e %s + %s", repo_root(), " ".join(pkgs) or "(no packages)")
    pip = str(venv / "bin" / "pip")
    rc = stream_subprocess([pip, "install", "-e", str(repo_root()), *pkgs], log_path)
    if rc != 0:
        raise AqaError(f"venv provisioning failed (rc={rc}); see {log_path}" + _log_tail(log_path))


def run_venv(run_cfg: RunConfig, out_dir: Path, *, dry_run: bool = False) -> RunResult:
    """Run the pipeline inside a local venv via a staged inner ``aqa run``.

    Same filesystem as the host, so the inner config keeps the user's
    ``output.dir`` as-is; the child run maintains ``aqa_run.json`` itself.
    """
    from ..proc import stream_subprocess

    env_cfg = run_cfg.executor.env
    venv = Path(env_cfg.path).expanduser()
    work = out_dir / ".aqa_local_venv"
    work.mkdir(parents=True, exist_ok=True)
    inner_yaml = work / "_run.yaml"
    inner_yaml.write_text(_inner_config(run_cfg, local_num_gpus(run_cfg), str(out_dir)))
    log_path = out_dir / "logs" / "executor_venv.log"

    argv = [str(venv / "bin" / "aqa"), "run", str(inner_yaml)]
    env = dict(os.environ)
    env.update(forwarded_env(run_cfg))
    env["PATH"] = f"{venv}/bin:{env.get('PATH', '')}"
    _log.info("[CMD] %s", shlex.join(argv))
    if dry_run:
        _log.info("[dry-run] venv=%s\n%s", venv, inner_yaml.read_text())
        return RunResult(output_dir=out_dir, state=ProcessState.DONE)

    if not (venv / "bin" / "python").exists():
        if not env_cfg.create:
            raise AqaError(f"venv not found at {venv} and executor.env.create is false")
        _log.info("[venv] provisioning %s", venv)
        provision_venv(run_cfg, venv, log_path)
    if not (venv / "bin" / "aqa").exists():
        raise AqaError(
            f"{venv} has no `aqa` entrypoint; delete it or let aqa provision it (env.create: true)"
        )
    _venv_agent_provision(run_cfg, venv, log_path)
    rc = stream_subprocess(argv, log_path, env=env)
    if rc != 0:
        raise AqaError(f"venv pipeline failed (rc={rc}); see {log_path}" + _log_tail(log_path))
    return RunResult(output_dir=out_dir, state=_final_state(out_dir))


def _venv_agent_provision(run_cfg: RunConfig, venv: Path, log_path: Path) -> None:
    """Provision the agent-fix backends inside the venv — parity with the
    container bootstrap's agent step (ssh/local docker), so a local venv run
    has the same OpenClaw/Copilot fix-loop available.

    Node.js (when missing on the host) is unpacked into the venv prefix itself
    and npm installs use ``--prefix <venv>`` — no sudo, no touching the host's
    global npm.  ``aqa.agent.provision`` then lays down the backend configs and
    syncs the project skills, exactly as in the container.  Best-effort: a
    failure here warns but never fails the run (the fix-loop just stays off).
    """
    from ..proc import stream_subprocess
    from ._remote import NODE_VERSION, OPENCLAW_VERSION, _agent_enabled, _wants_copilot_cli

    if not _agent_enabled(run_cfg):
        return

    env = dict(os.environ)
    env["PATH"] = f"{venv}/bin:{env.get('PATH', '')}"
    env["npm_config_prefix"] = str(venv)

    if shutil.which("node") is None and not (venv / "bin" / "node").exists():
        _log.info("[venv] install Node.js %s into the venv (agent CLIs)", NODE_VERSION)
        url = f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz"
        rc = stream_subprocess(
            ["bash", "-c", f"curl -fsSL {url} | tar -xJ --strip-components=1 -C {shlex.quote(str(venv))}"],
            log_path, env=env)
        if rc != 0:
            _log.warning("[venv] node install failed; agent fix-loop tiers disabled")
            return

    npm = venv / "bin" / "npm"
    npm_exe = str(npm) if npm.exists() else (shutil.which("npm") or None)
    if npm_exe:
        if not (venv / "bin" / "openclaw").exists() and shutil.which("openclaw") is None:
            rc = stream_subprocess([npm_exe, "install", "-g", f"openclaw@{OPENCLAW_VERSION}"],
                                   log_path, env=env)
            if rc != 0:
                _log.warning("[venv] openclaw install failed; fix-loop tiers may be unavailable")
        if _wants_copilot_cli(run_cfg) and not (venv / "bin" / "copilot").exists() \
                and shutil.which("copilot") is None:
            rc = stream_subprocess([npm_exe, "install", "-g", "@github/copilot"],
                                   log_path, env=env)
            if rc != 0:
                _log.warning("[venv] copilot CLI install failed; copilot tiers disabled")
    else:
        _log.warning("[venv] npm unavailable; agent fix-loop tiers disabled")
        return

    skills = repo_root() / "skills"
    rc = stream_subprocess(
        [str(venv / "bin" / "python"), "-m", "aqa.agent.provision",
         "--minimax-env", "MINIMAX_API_KEY", "--skills-dir", str(skills)],
        log_path, env=env)
    if rc != 0:
        _log.warning("[venv] agent config provisioning failed (fix-loop may lack auth)")


# ── docker ───────────────────────────────────────────────────────────────────
def _dockerfile_path(run_cfg: RunConfig) -> Path | None:
    """Resolve ``executor.image.dockerfile``: local/default -> the shipped gpu one."""
    df = (run_cfg.executor.image.dockerfile or "").strip()
    if not df:
        return None
    if df in ("local", "default"):
        return repo_root() / "docker" / "gpu.dockerfile"
    path = Path(df).expanduser()
    if not path.exists():
        raise AqaError(f"executor.image.dockerfile path not found: {path}")
    return path


def _docker_image(run_cfg: RunConfig, work: Path, log_path: Path, *, dry_run: bool) -> str:
    """Build (or reference) the image for a local docker run; return tag/name."""
    from ..proc import stream_subprocess

    image = run_cfg.executor.image
    base = (image.gpu or DEFAULT_BASE_IMAGE).strip()
    dockerfile = _dockerfile_path(run_cfg)
    if dockerfile is None:
        # No build: run the base image directly (docker run pulls on demand).
        argv = ["docker", "pull", base]
        _log.info("[CMD] %s", shlex.join(argv))
        if not dry_run:
            rc = stream_subprocess(argv, log_path)
            if rc != 0:  # best effort: a local cache may still satisfy `docker run`
                _log.warning("docker pull %s failed (rc=%s); falling back to local cache", base, rc)
        return base
    tag = "aqa-gpu:local"
    argv = ["docker", "build", "--build-arg", f"BASE_IMAGE={base}"]
    for key, val in image.build_args.items():
        argv += ["--build-arg", f"{key}={val}"]
    argv += ["-f", str(dockerfile), "-t", tag, str(work)]
    _log.info("[CMD] %s", shlex.join(argv))
    if not dry_run:
        rc = stream_subprocess(argv, log_path)
        if rc != 0:
            raise AqaError(f"docker build failed (rc={rc}); see {log_path}" + _log_tail(log_path))
    return tag


def run_docker(run_cfg: RunConfig, out_dir: Path, *, dry_run: bool = False) -> RunResult:
    """Run the pipeline inside a locally built/pulled container.

    Same staged inner-run flow as the ssh executor (``build_container_bootstrap``
    pip-installs aqa + the derived stack, then ``exec aqa run _run.yaml``), but
    launched with the local docker CLI.  The repo is bind-mounted at
    ``/workspace/aqa`` and the real output dir at its *same absolute path*, so
    every path the inner run writes lines up with the host.
    """
    from ..proc import stream_subprocess

    if not dry_run and shutil.which("docker") is None:
        raise AqaError(
            "docker not found on PATH; executor.env.type=docker needs the docker CLI and a "
            "local daemon (with the NVIDIA container toolkit for --gpus)"
        )
    out_dir = out_dir.resolve()  # bind-mount target must be an absolute path
    work = out_dir / ".aqa_local_docker"
    work.mkdir(parents=True, exist_ok=True)
    inner_yaml = work / "_run.yaml"
    inner_yaml.write_text(_inner_config(run_cfg, local_num_gpus(run_cfg), str(out_dir)))
    bootstrap = work / "_bootstrap.sh"
    bootstrap.write_text(build_container_bootstrap(run_cfg))
    log_path = out_dir / "logs" / "executor_docker.log"
    image_name = _docker_image(run_cfg, work, log_path, dry_run=dry_run)

    hf_cache = Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if devices and "," in devices:
        # docker >= 29 rejects a bare multi-id list ("cannot set both Count and
        # DeviceIDs"); embedding quotes makes its --gpus parser treat it as one
        # device request (verified on docker 29.6).
        gpus_value = f'"device={devices}"'
    elif devices:
        gpus_value = f"device={devices}"
    else:
        gpus_value = "all"
    secrets = forwarded_env(run_cfg)
    if devices:
        # The container re-indexes the selected GPUs to 0..N-1; forwarding the
        # HOST indices (e.g. CUDA_VISIBLE_DEVICES=3,4,5,7) would point vllm/torch
        # at non-existent devices inside the container ("World size (4) larger
        # than the number of available GPUs (1)"). Remap to container-local ids.
        secrets["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(len([d for d in devices.split(",") if d.strip()]))
        )
    # NVLink SHARP (NVLS) multicast needs fabric-manager access containers don't
    # get on this host (CUDA error 401 on bind). Default NCCL_NVLS_ENABLE=0 inside
    # the container; an explicit operator setting wins.
    secrets.setdefault("NCCL_NVLS_ENABLE", os.environ.get("NCCL_NVLS_ENABLE", "0"))
    secrets_path = work / "_secrets.env"
    # Host networking shares interfaces but NOT env: hosts behind a proxy need the
    # proxy vars forwarded explicitly or in-container pip/HF downloads fail with
    # "Network is unreachable" (observed on this host, 2026-08-26).
    proxy_env: list[str] = []
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        val = os.environ.get(name)
        if val:
            proxy_env += ["-e", f"{name}={val}"]
    argv = [
        "docker", "run", "--rm",
        # tty so progress bars (tqdm/auto-round/lm_eval) render instead of being
        # flattened (mirrors the ssh path's _CUDA_DEVICE_ARGS).
        "--tty",
        "--gpus", gpus_value,
        # NCCL in containers wants host IPC (vllm multi-proc workers); mirrors the
        # ssh path's XPU args. (Observed: NCCL "unhandled cuda error" without it.)
        "--ipc=host",
        # vLLM/multiproc needs large /dev/shm (docker default 64MiB -> "Insufficient
        # space in /dev/shm"); mirror the ssh path's 16g (_CUDA_DEVICE_ARGS).
        "--shm-size", "16g",
        *(["--network", run_cfg.executor.env.network] if run_cfg.executor.env.network else []),
        *proxy_env,
        "--env-file", str(secrets_path),
        "-e", f"HF_HOME={CONTAINER_HF_CACHE}",
        "-v", f"{repo_root()}:{CONTAINER_REPO}",
        "-v", f"{out_dir}:{out_dir}",
        "-v", f"{hf_cache}:{CONTAINER_HF_CACHE}",
        # The bootstrap expects both staged files at fixed repo paths.
        "-v", f"{inner_yaml}:{CONTAINER_REPO}/_run.yaml",
        "-v", f"{bootstrap}:{CONTAINER_REPO}/_bootstrap.sh",
    ]
    for mount in run_cfg.executor.env.mounts:
        argv += ["-v", mount]
    argv += ["--entrypoint", "bash", image_name, f"{CONTAINER_REPO}/_bootstrap.sh"]
    _log.info("[CMD] %s", shlex.join(argv))
    if dry_run:
        _log.info("[dry-run] secrets=%s image=%s", sorted(secrets), image_name)
        return RunResult(output_dir=out_dir, state=ProcessState.DONE)

    hf_cache.mkdir(parents=True, exist_ok=True)  # keep it user-owned (docker would root-own it)
    secrets_path.write_text("".join(f"{k}={v}\n" for k, v in secrets.items()))
    os.chmod(secrets_path, 0o600)
    try:
        rc = stream_subprocess(argv, log_path)
    finally:
        secrets_path.unlink(missing_ok=True)
    if rc != 0:
        raise AqaError(f"docker pipeline failed (rc={rc}); see {log_path}" + _log_tail(log_path))
    return RunResult(output_dir=out_dir, state=_final_state(out_dir))

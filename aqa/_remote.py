"""Helpers for the SSH executor: build the inner (in-container) config, collect
secret env vars, and generate the remote dispatch bash script.

The remote flow is intentionally self-contained -- it ships *this* package (not
an external repo), builds a docker image, and runs ``aqa run`` inside the
container with ``executor.type=local``.  There is no dependency on lb_eval.
"""

from __future__ import annotations

import os
import shlex
import time
import uuid
from pathlib import Path

import yaml

from ..config import RunConfig
from ..logging_util import get_logger

_log = get_logger("executor.ssh.remote")

DEFAULT_BASE_IMAGE = "nvcr.io/nvidia/pytorch:26.01-py3"
DEFAULT_XPU_BASE_IMAGE = "vllm/vllm-openai-xpu:nightly"

# Agent-fix backend versions installed inside the container (parity with
# local_dispatch). Node is required by both the OpenClaw and Copilot CLIs.
NODE_VERSION = "v22.20.0"
OPENCLAW_VERSION = "2026.6.8"

# Docker device/runtime args per accelerator family.  CUDA isolates GPUs with
# `--gpus '"device=<ids>"'`; Intel XPU passes /dev/dri through and selects cards
# with a Level-Zero affinity mask (re-indexed 0..N-1 inside the container).
_CUDA_DEVICE_ARGS = (
    '    --gpus \'"device=\'"$DEVICE_LIST"\'"\' \\\n'
    '    --shm-size=16g \\\n'
    '    --tty \\\n'
    '    -e PYTHONUNBUFFERED=1 \\\n'
)
_XPU_DEVICE_ARGS = (
    '    --device /dev/dri:/dev/dri \\\n'
    '    -v /dev/dri/by-path:/dev/dri/by-path \\\n'
    '    --ipc=host --privileged \\\n'
    '    --shm-size=16g \\\n'
    '    --tty \\\n'
    '    -e PYTHONUNBUFFERED=1 \\\n'
    '    -e ZE_AFFINITY_MASK="$DEVICE_LIST" \\\n'
    '    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\\n'
)

# Container paths (repo bind-mounted here; results land under RESULTS and are kept
# on the remote host inside the per-run work dir).
CONTAINER_REPO = "/workspace/aqa"
CONTAINER_RESULTS = "/workspace/aqa/results"
CONTAINER_HF_CACHE = "/root/.cache/huggingface"

# Env vars forwarded into the container when present in the local environment.
# password_env (ssh auth) is deliberately NOT here -- it never enters the container.
_SECRET_ALLOWLIST = (
    "HF_TOKEN",
    "HF_TOKENS",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "MINIMAX_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_TOKEN",
)


def make_run_id(run_cfg: RunConfig) -> str:
    """Unique, filesystem-safe per-run id: ``<model>_<scheme>_<ts>_<rand>``."""
    scheme = run_cfg.quant.scheme if run_cfg.quant else "eval"
    base = f"{run_cfg.model.split('/')[-1]}_{scheme}"
    base = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in base)
    return f"{base}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def repo_root() -> Path:
    """Locate the installed source tree root (the dir holding ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "aqa").is_dir():
            return parent
    # Fallback: two levels up from aqa/executors/ -> repo root.
    return here.parents[2]


def build_inner_config(
    run_cfg: RunConfig,
    run_id: str,
    *,
    num_gpus: int | None = None,
    output_dir: str | None = None,
) -> str:
    """Serialize a derived run config for the in-container ``local`` run.

    Strips the ssh/proxy executor wiring (that is the *host's* job) and repoints
    output at the mounted results dir.  Secrets are never embedded -- they arrive
    via the docker ``--env-file``.  Local venv/docker env dispatch passes an
    explicit ``num_gpus`` / ``output_dir`` (there is no ssh block to derive from).
    """
    data = run_cfg.model_dump(mode="json", exclude_none=True)
    # The container is launched with `--gpus device=<ids>`; the in-container run
    # sees executor.type=local (no ssh block), so it would otherwise default to a
    # single GPU. Carry the host's device count through so the remote pipeline uses
    # the right vllm tensor_parallel_size and quantize --device_map.
    if num_gpus is None:
        ssh_gpus = run_cfg.executor.ssh.gpus if run_cfg.executor.ssh else ""
        num_gpus = max(1, len([g for g in str(ssh_gpus).split(",") if g.strip()]))
    data["executor"] = {"type": "local", "num_gpus": num_gpus}
    data.setdefault("output", {})
    data["output"]["dir"] = output_dir or f"{CONTAINER_RESULTS}/{run_id}"
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def collect_secret_env(run_cfg: RunConfig) -> dict[str, str]:
    """Gather env vars to forward into the container (secrets + sink tokens).

    Sources: the standard allowlist (when set in the environment) plus any
    ``*_env`` keys referenced by output sinks (e.g. ``token_env``).
    """
    env: dict[str, str] = {}
    for name in _SECRET_ALLOWLIST:
        val = os.environ.get(name)
        if val:
            env[name] = val
    for sink in run_cfg.output.sinks:
        for key, value in sink.model_dump().items():
            if key.endswith("_env") and isinstance(value, str) and value:
                resolved = os.environ.get(value)
                if resolved:
                    env[value] = resolved
                    # Also expose as HF_TOKEN so huggingface_hub picks it up by default.
                    if "token" in key.lower():
                        env.setdefault("HF_TOKEN", resolved)
    return env


def _proxy_pairs(run_cfg: RunConfig) -> list[tuple[str, str]]:
    proxy = run_cfg.executor.proxy
    pairs: list[tuple[str, str]] = []
    for var, val in (
        ("HTTP_PROXY", proxy.http_proxy),
        ("HTTPS_PROXY", proxy.https_proxy),
        ("NO_PROXY", proxy.no_proxy),
    ):
        if val:
            pairs.append((var, val))
            pairs.append((var.lower(), val))
    return pairs


def build_remote_script(
    run_cfg: RunConfig,
    run_id: str,
    *,
    has_secrets: bool,
    remote_root: str = "~/aqa-runs",
) -> str:
    """Bash executed on the remote host: extract -> docker build -> docker run aqa.

    All per-run state lives under an isolated ``$ROOT/<run_id>`` directory so
    concurrent dispatches to the same machine never collide.  Only the HF cache
    and the docker image tag are shared.
    """
    gpus = run_cfg.executor.ssh.gpus if run_cfg.executor.ssh else "0"
    device_list = ",".join(g.strip() for g in str(gpus).split(",") if g.strip())
    accelerator = _accelerator(run_cfg)
    image = run_cfg.executor.image
    dockerfile = (image.dockerfile or "").strip()
    if accelerator == "xpu":
        base_image = (image.xpu or DEFAULT_XPU_BASE_IMAGE).strip()
        device_args = _XPU_DEVICE_ARGS
    else:
        base_image = (image.gpu or DEFAULT_BASE_IMAGE).strip()
        device_args = _CUDA_DEVICE_ARGS

    proxy_pairs = _proxy_pairs(run_cfg)
    proxy_shell = "".join(f'export {k}={shlex.quote(v)}\n' for k, v in proxy_pairs)
    build_proxy = "".join(f'    --build-arg {k}={shlex.quote(v)} \\\n' for k, v in proxy_pairs)
    run_proxy = "".join(f'    -e {k}={shlex.quote(v)} \\\n' for k, v in proxy_pairs)
    env_file_arg = '    --env-file "$WORK/_secrets.env" \\\n' if has_secrets else ""
    mounts = (run_cfg.executor.ssh.mounts if run_cfg.executor.ssh else []) or []
    mounts_args = "".join(f'    -v {shlex.quote(m)} \\\n' for m in mounts)

    tag = f"aqa-gpu:{'local' if dockerfile else 'base'}"

    if not dockerfile:
        # No build: run the base image directly.
        build_section = (
            'IMAGE="' + base_image + '"\n'
            'echo "== [2/4] docker pull base image (no build) =="\n'
            '( set -x; docker pull "$IMAGE" ) || echo "[dispatch] pull failed; using local cache"\n'
        )
    else:
        # Build our shipped Dockerfile (or a custom one) from a clean context dir.
        build_section = (
            f'IMAGE="{tag}"\n'
            'mkdir -p "$WORK/_dockerctx"\n'
            'cp "$WORK/_gpu.dockerfile" "$WORK/_dockerctx/Dockerfile"\n'
            'echo "== [2/4] docker build (BASE_IMAGE=' + base_image + ') =="\n'
            'build_ok=0\n'
            'for i in 1 2 3; do\n'
            '    if ( set -x; docker build \\\n'
            f'{build_proxy}'
            f'        --build-arg BASE_IMAGE={shlex.quote(base_image)} \\\n'
            '        -f "$WORK/_dockerctx/Dockerfile" -t "$IMAGE" "$WORK/_dockerctx" ); then\n'
            '        build_ok=1; break\n'
            '    fi\n'
            '    echo "[dispatch] docker build attempt $i failed; retrying in 15s..." >&2; sleep 15\n'
            'done\n'
            '[ "$build_ok" = 1 ] || { echo "[dispatch] ERROR: docker build failed" >&2; exit 1; }\n'
        )

    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT=$(eval echo "{remote_root}")
RUN_ID="{run_id}"
WORK="$ROOT/$RUN_ID"
REPO="$WORK/repo"
HF_CACHE_ABS=$(eval echo "~/.cache/huggingface")
DEVICE_LIST="{device_list}"
CONTAINER="aqa-{run_id}"
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"

{proxy_shell}mkdir -p "$WORK" "$REPO" "$HF_CACHE_ABS"

echo "== [1/4] extract staged repo into $REPO =="
tar -xzf "$WORK/_repo.tgz" -C "$REPO"

{build_section}
echo "== [3/4] stage run config + container bootstrap =="
cp "$WORK/_run.yaml" "$REPO/_run.yaml"
cp "$WORK/_bootstrap.sh" "$REPO/_bootstrap.sh"
chmod +x "$REPO/_bootstrap.sh"

echo "== [4/4] docker run: pip install aqa + aqa run (GPUs device=$DEVICE_LIST) =="
trap 'rm -f "$WORK/_secrets.env"' EXIT
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
rc=0
( set -x
docker run --rm --name "$CONTAINER" \\
{device_args}{env_file_arg}{run_proxy}{mounts_args}    -v "$REPO":{CONTAINER_REPO} \\
    -v "$HF_CACHE_ABS":{CONTAINER_HF_CACHE} \\
    --entrypoint bash \\
    "$IMAGE" \\
    {CONTAINER_REPO}/_bootstrap.sh
) || rc=$?

# Container runs as root over the bind mount; restore ownership for the SSH user.
docker run --rm -v "$WORK":/work --entrypoint sh ubuntu:24.04 \\
    -c "chown -R $HOST_UID:$HOST_GID /work" >/dev/null 2>&1 || true

echo "[dispatch] run $RUN_ID finished rc=$rc; results under $REPO/results/$RUN_ID"
exit $rc
"""


def _accelerator(run_cfg: RunConfig) -> str:
    """Accelerator family of the dispatch target ("cuda" or "xpu")."""
    return run_cfg.executor.ssh.accelerator if run_cfg.executor.ssh else "cuda"


def container_packages(run_cfg: RunConfig) -> list[str]:
    """Compute the pip package list installed inside the container.

    Base stack = auto-round + lm-eval; vllm is added when the eval backend needs
    it -- *except* on XPU, where the base image already ships a matching
    vllm-xpu build (installing CUDA ``vllm`` would clobber it).  User extras from
    ``executor.image.pip_install`` are appended (and can pin versions).
    """
    pkgs: list[str] = []
    if run_cfg.eval.backend == "vllm_omni":
        # Diffusion text-to-image stack: engine + pipeline utils + CLIP scorer deps.
        pkgs += [
            "vllm-omni", "diffusers", "datasets", "transformers",
            "sentencepiece", "accelerate", "pillow",
        ]
    elif run_cfg.eval.backend in ("vllm", "vllm_xpu") and _accelerator(run_cfg) != "xpu":
        pkgs.append("vllm")
    pkgs += ["auto-round", "lm-eval"]
    for extra in run_cfg.executor.image.pip_install:
        if extra and extra not in pkgs:
            pkgs.append(extra)
    return pkgs


def _agent_enabled(run_cfg: RunConfig) -> bool:
    agent = run_cfg.agent
    return bool(agent and (agent.fix_loop or agent.optimize))


def _wants_copilot_cli(run_cfg: RunConfig) -> bool:
    """True when the agent stack may invoke the ``copilot`` CLI on the remote host.

    The fix-loop / optimize agent only needs the Copilot CLI when at least one
    configured tier resolves to the ``copilot`` backend (opus/sonnet/minimax/…).
    """
    if not _agent_enabled(run_cfg):
        return False
    from ..agent.tiers import resolve_tiers

    return any(t.backend == "copilot" for t in resolve_tiers(run_cfg.agent.tiers))


def _agent_provision_snippet(run_cfg: RunConfig) -> str:
    """Install + provision the agent-fix backends (OpenClaw + optional Copilot).

    Parity with ``local_dispatch/remote_exec.py``: OpenClaw is the default tier-0
    backend and is always installed; the Copilot CLI is added only when a
    copilot-backed tier is configured.  Node.js is baked into the ``local`` GPU
    image but absent from a plain base image (e.g. the vLLM XPU image), so it is
    installed at runtime when missing.  Config for both backends is laid down by
    ``aqa.agent.provision`` (syncs ``MINIMAX_API_KEY``, shares the project skills).
    """
    if not _agent_enabled(run_cfg):
        return ""
    copilot_install = ""
    if _wants_copilot_cli(run_cfg):
        copilot_install = (
            '    command -v copilot >/dev/null 2>&1 \\\n'
            "        || npm install -g @github/copilot >/dev/null 2>&1 \\\n"
            '        || echo "[bootstrap] WARN: copilot CLI install failed; '
            'copilot tiers disabled" >&2\n'
        )
    return f"""
echo "== container: provision agent-fix backends =="
if ! command -v node >/dev/null 2>&1; then
    echo "== container: install Node.js {NODE_VERSION} (agent CLIs) =="
    curl -fsSL https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz \\
        | tar -xJ --strip-components=1 -C /usr/local 2>/dev/null \\
        || echo "[bootstrap] WARN: node install failed; agent fix-loop tiers disabled" >&2
fi
if command -v npm >/dev/null 2>&1; then
    command -v openclaw >/dev/null 2>&1 \\
        || npm install -g openclaw@{OPENCLAW_VERSION} >/dev/null 2>&1 \\
        || echo "[bootstrap] WARN: openclaw install failed" >&2
{copilot_install}    python -m aqa.agent.provision --minimax-env MINIMAX_API_KEY --skills-dir {CONTAINER_REPO}/skills \\
        || echo "[bootstrap] WARN: agent config provisioning failed" >&2
else
    echo "[bootstrap] WARN: node/npm unavailable; agent fix-loop tiers will be skipped." >&2
fi
"""


def build_container_bootstrap(run_cfg: RunConfig) -> str:
    """Bash run inside the container: install aqa + stack, then ``aqa run`` local.

    PEP-668 ("externally managed") is handled with ``--break-system-packages`` so
    installs land in the NGC base env instead of failing under uv --system.  The
    stack install is non-fatal (the agent fix-loop is meant to reconcile env
    drift), but the aqa install itself must succeed.  Versions are printed for
    debuggability.  When the agent fix-loop is enabled, the OpenClaw (+ optional
    Copilot) backends are installed and provisioned so the fix-loop can run on the
    remote host.
    """
    env_pkgs = run_cfg.executor.env.packages
    # An explicit executor.env.packages list overrides the derived set — required
    # when the base image already ships a custom stack (e.g. a vendor vllm build
    # that must NOT be clobbered by the PyPI `vllm` the derivation would add).
    pkg_list = container_packages(run_cfg) if env_pkgs == "auto" else list(env_pkgs)
    pkgs = " ".join(shlex.quote(p) for p in pkg_list)
    agent_step = _agent_provision_snippet(run_cfg)
    return f"""#!/usr/bin/env bash
set -uo pipefail
cd {CONTAINER_REPO}

export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1

echo "== container: install aqa (editable) =="
if ! pip install -e . ; then
    echo "[bootstrap] FATAL: failed to install aqa" >&2
    exit 1
fi

echo "== container: install quant/eval stack: {pkgs} =="
pip install {pkgs} \\
    || echo "[bootstrap] WARN: stack install had errors; continuing (agent-fix may repair env)"
{agent_step}
echo "== container: environment =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())" || true
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"

echo "== container: aqa run (executor.type=local) =="
exec aqa run {CONTAINER_REPO}/_run.yaml
"""


# Backwards-compatible constant (HF-only stack); prefer build_container_bootstrap().
CONTAINER_BOOTSTRAP = f"""#!/usr/bin/env bash
set -uo pipefail
cd {CONTAINER_REPO}
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
pip install -e . || exit 1
pip install auto-round lm-eval || true
exec aqa run {CONTAINER_REPO}/_run.yaml
"""

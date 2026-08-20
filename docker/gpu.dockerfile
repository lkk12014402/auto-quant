# Self-contained GPU image for `aqa` remote dispatch (executor.type=ssh, image.dockerfile=local).
#
# Built + run by aqa/executors/ssh.py when `executor.image.dockerfile: local`.
# It sits on top of an NVIDIA NGC PyTorch base image so the CUDA toolkit (nvcc 13.x)
# is on PATH and flashinfer / vLLM JIT kernels compile for modern GPUs (sm_90a/H200)
# instead of hitting the CUDA-12.0 `FlagHeads` failure of an apt `nvidia-cuda-toolkit`.
#
# The image installs only OS tooling + `uv`; the actual quant/eval stack
# (auto-round, lm-eval, vllm, aqa itself) is installed at container runtime from the
# staged repo, so the image stays stable while the framework iterates.
#
# Override the base at build time:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3 ...
# or in run.yaml:  executor.image.gpu: nvcr.io/nvidia/pytorch:25.10-py3
#
# NGC images are large (~20 GB) and need a recent NVIDIA driver on the host; verify
# `docker run --gpus all <base> nvidia-smi` on the target box first.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_MININTERVAL=120

# uv (fast Python package manager) used by the container bootstrap to install the stack.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# CLI tools the pipeline expects (NGC ships build-essential/git/curl already).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git jq unzip curl zip ca-certificates wget numactl time xz-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Node.js v22 + the OpenClaw agent-fix backend (parity with local_dispatch's
# agent image). OpenClaw is the default tier-0 fix backend; the Copilot CLI is
# added at container runtime only when a copilot-backed tier (opus/sonnet/minimax)
# is configured. Runtime provisioning (`python -m aqa.agent.provision`) lays the
# ~/.openclaw + ~/.copilot config down and syncs MINIMAX_API_KEY.
RUN curl -fsSL https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
        | tar -xJ --strip-components=1 -C /usr/local
RUN npm install -g openclaw@2026.6.8 && npm cache clean --force

WORKDIR /workspace

# No ENTRYPOINT: ssh.py runs the container with `--entrypoint bash` + a staged
# bootstrap that pip-installs the repo and runs `aqa run` with executor.type=local.

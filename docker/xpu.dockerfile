# Self-contained Intel XPU image for `aqa` remote dispatch
# (executor.type=ssh, executor.ssh.accelerator=xpu, image.dockerfile=local).
#
# Built + run by aqa/executors/ssh.py when the run targets an Intel GPU (B60 /
# Arc / Data Center GPU Max). It sits on top of the official vLLM XPU image so
# the torch-xpu + vLLM-XPU stack (and the Level-Zero / oneAPI runtime) is already
# present -- we deliberately do NOT reinstall torch or vllm here (a CUDA wheel
# would clobber the XPU build and disable device enumeration). Only OS tooling,
# uv, and the OpenClaw agent-fix backend are added, mirroring docker/gpu.dockerfile.
#
# Override the base at build time:
#   docker build --build-arg BASE_IMAGE=vllm/vllm-openai-xpu:nightly ...
# or in run.yaml:  executor.image.xpu: vllm/vllm-openai-xpu:nightly
#
# Verify the target host first: `docker run --device /dev/dri --privileged <base> \
#   python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"`.
ARG BASE_IMAGE=vllm/vllm-openai-xpu:nightly
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_MININTERVAL=120 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# uv (fast Python package manager) used by the container bootstrap to install the
# quant/eval stack (auto-round + lm-eval) on top of the image's vLLM-XPU.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# CLI tools the pipeline expects.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git jq unzip curl zip ca-certificates wget numactl time xz-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Node.js v22 + the OpenClaw agent-fix backend (parity with docker/gpu.dockerfile).
# OpenClaw is the default tier-0 fix backend; the Copilot CLI is added at container
# runtime only when a copilot-backed tier (opus/sonnet/minimax) is configured.
RUN curl -fsSL https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
        | tar -xJ --strip-components=1 -C /usr/local
RUN npm install -g openclaw@2026.6.8 && npm cache clean --force

WORKDIR /workspace

# No ENTRYPOINT: ssh.py runs the container with `--entrypoint bash` + a staged
# bootstrap that pip-installs the repo and runs `aqa run` with executor.type=local.

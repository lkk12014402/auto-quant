#!/usr/bin/env python3
"""vllm inference smoke test for a quantized (or any) local checkpoint.

Loads the model with tensor parallelism and generates a few deterministic
completions, so a broken export (bad config, missing/misnamed weights) fails
here in minutes instead of deep inside an eval run.  Written after the
Qwen3.8-Flash-Next day-0 work (see docs/day0_quantization_recipe.md).

Usage:
  python scripts/smoke_vllm.py /path/to/checkpoint            # TP4 default
  NUM_GPUS=2 python scripts/smoke_vllm.py /path/to/checkpoint
  KV_DTYPE=fp8 LANGUAGE_MODEL_ONLY=1 python scripts/smoke_vllm.py <ckpt>
    ``KV_DTYPE`` matters when the checkpoint itself carries a ``kv_cache_scheme``
    (auto-round ``--static_kv_dtype fp8``): the engine only consumes those scales
    on the matching cache dtype, and a runtime that rejects the scheme fails here
    instead of mid-eval.  ``LANGUAGE_MODEL_ONLY`` skips the vision tower of a VL
    checkpoint (qwen4_exp ships one).
"""

from __future__ import annotations

import os
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
EXTRA_KWARGS: dict = {}
if os.environ.get("KV_DTYPE"):
    EXTRA_KWARGS["kv_cache_dtype"] = os.environ["KV_DTYPE"]
if os.environ.get("LANGUAGE_MODEL_ONLY"):
    EXTRA_KWARGS["language_model_only"] = True

PROMPTS = [
    "The capital of France is",
    "1+1=2. 2+2=4. 4+4=",
    "def quicksort(arr):",
]


def main() -> None:
    if not MODEL:
        sys.exit("usage: smoke_vllm.py /path/to/checkpoint  (NUM_GPUS env, default 4)")
    llm = LLM(model=MODEL, tensor_parallel_size=NUM_GPUS, max_model_len=8192,
              dtype="bfloat16", gpu_memory_utilization=0.9,
              enable_prefix_caching=False, trust_remote_code=True,
              **EXTRA_KWARGS)
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(PROMPTS, sp)
    for prompt, out in zip(PROMPTS, outs):
        print(f"\n### {prompt!r}\n{out.outputs[0].text!r}")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()

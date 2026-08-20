"""vLLM evaluation backend (CUDA)."""

from __future__ import annotations

from typing import Any

from . import merge_model_args


class VllmBackend:
    name = "vllm"
    lm_eval_model_flag = "vllm"

    def build_model_args(
        self,
        model: str,
        *,
        num_gpus: int = 1,
        gen_max_toks: int = 2048,
        generative: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        args = [
            f"pretrained={model}",
            f"tensor_parallel_size={num_gpus}",
            "max_model_len=8192",
            "gpu_memory_utilization=0.9",
            "dtype=bfloat16",
            "trust_remote_code=True",
            "add_bos_token=True",
            "enable_prefix_caching=False",
            f"max_gen_toks={gen_max_toks}",
        ]
        return merge_model_args(args, extra_args)

    def env_overrides(self) -> dict[str, str]:
        return {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


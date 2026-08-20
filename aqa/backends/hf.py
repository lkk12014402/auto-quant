"""HuggingFace `transformers` evaluation backend."""

from __future__ import annotations

from typing import Any

from . import merge_model_args


class HfBackend:
    name = "hf"
    lm_eval_model_flag = "hf"

    def build_model_args(
        self,
        model: str,
        *,
        num_gpus: int = 1,
        gen_max_toks: int = 2048,
        generative: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        args = [f"pretrained={model}", "dtype=bfloat16", "trust_remote_code=True"]
        if num_gpus > 1:
            args.append("parallelize=True")
        return merge_model_args(args, extra_args)

    def env_overrides(self) -> dict[str, str]:
        return {}


"""lm-eval client backend for an already running OpenAI-compatible completions server."""

from __future__ import annotations

from typing import Any

from . import merge_model_args


class LocalCompletionsBackend:
    name = "local_completions"
    lm_eval_model_flag = "local-completions"

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
            f"model={model}",
            f"tokenizer={model}",
            "tokenizer_backend=huggingface",
            "base_url=http://127.0.0.1:8000/v1/completions",
            "tokenized_requests=False",
            "num_concurrent=1",
            "max_retries=50",
            "timeout=500",
            "max_length=131072",
            f"max_gen_toks={gen_max_toks}",
        ]
        return merge_model_args(args, extra_args)

    def env_overrides(self) -> dict[str, str]:
        return {}

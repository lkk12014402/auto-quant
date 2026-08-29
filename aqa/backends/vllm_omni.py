"""vllm-omni evaluation backend (diffusion text-to-image, CUDA).

Unlike hf/vllm this backend is not driven through lm_eval: the ``diffusion``
harness talks to vllm-omni's ``Omni`` API directly, so the lm_eval-facing
surface of the :class:`Backend` protocol is a no-op here.
"""

from __future__ import annotations

from typing import Any


class VllmOmniBackend:
    name = "vllm_omni"
    lm_eval_model_flag = "vllm_omni"  # placeholder; the diffusion harness never reads it

    def build_model_args(
        self,
        model: str,
        *,
        num_gpus: int = 1,
        gen_max_toks: int = 2048,
        generative: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        # The diffusion harness loads the model via ``Omni(model=...)`` and does
        # not consume lm_eval ``--model_args``; nothing to render.
        return ""

    def env_overrides(self) -> dict[str, str]:
        return {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

"""Inference backends: encapsulate how a model is loaded for evaluation.

A backend maps to lm-eval's ``--model`` flag and builds the ``--model_args``
string, faithful to the historic ``local_dispatch`` ``build_model_args`` logic.
Adding a new infra (e.g. vllm_xpu) is a new class + entry-point registration.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


def _fmt(value: Any) -> str:
    """Render a value the way lm_eval's --model_args expects (bools => True/False)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def merge_model_args(base: list[str], extra: dict[str, Any] | None) -> str:
    """Merge ``key=value`` defaults with user ``extra`` (extra overrides), preserving order.

    ``base`` is a list of ``"key=value"`` strings; ``extra`` is a dict whose keys
    replace matching base keys and whose new keys are appended.  Used so users can
    override or extend backend model_args from ``eval.lm_eval.model_args`` (e.g.
    ``enable_thinking=False``, ``reasoning_parser=qwen3``).
    """
    pairs: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for item in base:
        key, _, val = item.partition("=")
        seen[key] = len(pairs)
        pairs.append((key, val))
    for key, value in (extra or {}).items():
        rendered = _fmt(value)
        if key in seen:
            pairs[seen[key]] = (key, rendered)
        else:
            seen[key] = len(pairs)
            pairs.append((key, rendered))
    return ",".join(f"{k}={v}" for k, v in pairs)


@runtime_checkable
class Backend(Protocol):
    name: str
    lm_eval_model_flag: str

    def build_model_args(
        self,
        model: str,
        *,
        num_gpus: int = 1,
        gen_max_toks: int = 2048,
        generative: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        ...

    def env_overrides(self) -> dict[str, str]:
        """Extra environment variables to export before invoking lm_eval."""
        ...


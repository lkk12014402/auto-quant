"""Harness adapters: one contract, many evaluation frameworks.

* ``native``  -- our in-repo kernel (migrated from NeMo Evaluator, Apache-2.0).
* ``lm_eval`` -- EleutherAI lm-evaluation-harness (optional dep).
* ``diffusion`` -- vllm-omni text-to-image generation + CLIP score.
* ``evalscope`` -- optional subprocess plugin package ``aqa-harness-evalscope``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..registry import (
    BenchmarkRecipe,
    BenchmarkResult,
)


@runtime_checkable
class Harness(Protocol):
    name: str

    def available(self) -> bool:
        """Return True if this harness can run (e.g. its CLI/lib is installed)."""
        ...

    def run(
        self,
        recipe: BenchmarkRecipe,
        model: str,
        backend,
        out_dir: Path,
    ) -> BenchmarkResult:
        """Evaluate ``model`` on ``recipe`` and return a normalised result."""
        ...

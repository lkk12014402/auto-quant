"""Model providers: how the model-under-eval is produced/served.

This is the eval-centric core of the framework.  Evaluation is the primary
concern; *where the model comes from* is a pluggable :class:`ModelProvider`:

  * ``pretrained`` -- evaluate a HF repo id / local path as-is (default).
  * ``quantize``   -- run autoquant (optionally agent-optimize + fix-loop) to
    produce a quantized checkpoint, then evaluate it.  Quantization is thus a
    *plugin*, not the framework's reason for being.
  * ``endpoint``   -- an already-served OpenAI-compatible endpoint (driven
    directly by the agentic evals; no local load).

A provider returns a :class:`ResolvedModel` telling the pipeline what to feed the
eval harnesses: a local ``model`` path/id, and/or a served ``endpoint``.

Mirrors NeMo Evaluator's separation of the *served target* from the *benchmark*:
the harness/benchmark side stays identical regardless of how the target was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ..config import RunConfig


@dataclass
class ResolvedModel:
    """The evaluation target produced by a provider.

    ``model`` is a HF id / local path for harnesses that load weights (lm_eval,
    native); ``endpoint`` + ``served_model`` describe an OpenAI-compatible server
    for the agentic path.  ``kind`` is ``"path"`` or ``"endpoint"``.
    """

    model: str
    kind: str = "path"
    endpoint: str | None = None
    served_model: str | None = None
    api_key_env: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderContext:
    """Everything a provider needs from the pipeline (kept injectable for tests).

    ``quantize_fn`` runs the (fix-loop-wrapped) quantization into a target dir; the
    pipeline binds it so the heavy self-heal wiring stays in one place while the
    quantize provider stays a thin, unit-testable orchestrator.
    """

    out_dir: Path
    num_gpus: int = 1
    quantize_fn: Callable[[Path], None] | None = None


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def provide(self, run_cfg: RunConfig, ctx: ProviderContext) -> ResolvedModel:
        """Produce the evaluation target for ``run_cfg``."""
        ...

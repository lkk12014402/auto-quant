"""Builtin model providers: pretrained / quantize / endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging_util import AqaError, get_logger
from ..quantize import resolve_export_dir
from .base import ProviderContext, ResolvedModel

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

_log = get_logger("providers")


class PretrainedProvider:
    """Evaluate the model exactly as given (HF repo id or local path)."""

    name = "pretrained"

    def provide(self, run_cfg: RunConfig, ctx: ProviderContext) -> ResolvedModel:
        return ResolvedModel(model=run_cfg.model, kind="path")


class QuantizeProvider:
    """Quantize the model (optionally agent-optimize + fix-loop), then evaluate it.

    The actual quantize call — including the agent fix-loop — is injected via
    ``ctx.quantize_fn`` by the pipeline, so this provider stays a thin orchestrator
    that is trivially unit-testable with a fake.
    """

    name = "quantize"

    def provide(self, run_cfg: RunConfig, ctx: ProviderContext) -> ResolvedModel:
        if ctx.quantize_fn is None:
            raise AqaError("quantize provider requires a quantize_fn (pipeline wiring)")
        quant_dir = ctx.out_dir / "quantized"
        quant = run_cfg.provider.quant
        _log.info("== quantize (%s / %s) ==", quant.scheme, quant.method)
        ctx.quantize_fn(quant_dir)
        model_dir = resolve_export_dir(quant_dir)
        return ResolvedModel(model=str(model_dir), kind="path",
                             metadata={"scheme": quant.scheme, "method": quant.method})


class EndpointProvider:
    """Evaluate an already-served OpenAI-compatible endpoint (agentic path)."""

    name = "endpoint"

    def provide(self, run_cfg: RunConfig, ctx: ProviderContext) -> ResolvedModel:
        p = run_cfg.provider
        if not p.base_url:
            raise AqaError("endpoint provider requires `provider.base_url`")
        return ResolvedModel(
            model=p.served_model or run_cfg.model,
            kind="endpoint",
            endpoint=p.base_url,
            served_model=p.served_model or run_cfg.model,
            api_key_env=p.api_key_env,
        )

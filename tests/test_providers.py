"""Tests for the eval-centric ModelProvider flip + backward-compat quant shim."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqa.config import parse_run_config
from aqa.logging_util import AqaError, ConfigError
from aqa.providers.base import ProviderContext, ResolvedModel
from aqa.providers.builtin import (
    EndpointProvider,
    PretrainedProvider,
    QuantizeProvider,
)
from aqa.registry import PROVIDERS


# ── backward-compat: legacy top-level `quant:` still works ────────────────────
def test_legacy_top_level_quant_folds_into_provider():
    cfg = parse_run_config({"model": "org/m", "quant": {"scheme": "MXFP8", "method": "model_free"}})
    assert cfg.provider.type == "quantize"
    assert cfg.provider.quant.scheme == "MXFP8"
    # `.quant` back-compat accessor keeps the old pipeline semantics.
    assert cfg.quant.scheme == "MXFP8"


def test_new_provider_quantize_form():
    cfg = parse_run_config({"model": "org/m", "provider": {"type": "quantize",
                                                           "quant": {"scheme": "NVFP4"}}})
    assert cfg.provider.type == "quantize"
    assert cfg.quant.scheme == "NVFP4"


def test_default_is_pretrained_eval_only():
    cfg = parse_run_config({"model": "org/m"})
    assert cfg.provider.type == "pretrained"
    assert cfg.quant is None  # eval-only; pipeline quant gate stays a no-op


def test_endpoint_provider_config():
    cfg = parse_run_config({"model": "org/m",
                            "provider": {"type": "endpoint", "base_url": "http://h:8000/v1",
                                         "served_model": "org/m", "api_key_env": "MY_KEY"}})
    assert cfg.provider.type == "endpoint"
    assert cfg.quant is None
    assert cfg.provider.base_url == "http://h:8000/v1"


def test_conflicting_quant_and_nonquantize_provider_raises():
    with pytest.raises(ConfigError, match="provider.type"):
        parse_run_config({"model": "org/m", "quant": {"scheme": "MXFP8"},
                          "provider": {"type": "endpoint", "base_url": "http://h/v1"}})


def test_explicit_null_quant_is_eval_only():
    cfg = parse_run_config({"model": "org/m", "quant": None})
    assert cfg.provider.type == "pretrained"
    assert cfg.quant is None


def test_quantize_provider_defaults_quant_block():
    cfg = parse_run_config({"model": "org/m", "provider": {"type": "quantize"}})
    assert cfg.quant is not None
    assert cfg.quant.scheme == "MXFP4"  # QuantConfig default


# ── provider registry + implementations ──────────────────────────────────────
def test_providers_registered():
    assert {"pretrained", "quantize", "endpoint"}.issubset(set(PROVIDERS.names()))


def test_pretrained_provider_returns_model_as_is():
    cfg = parse_run_config({"model": "org/m"})
    resolved = PretrainedProvider().provide(cfg, ProviderContext(out_dir=Path("/tmp/o")))
    assert resolved == ResolvedModel(model="org/m", kind="path")


def test_quantize_provider_runs_injected_fn_and_returns_dir(tmp_path):
    cfg = parse_run_config({"model": "org/m", "quant": {"scheme": "MXFP8"}})
    called = {}

    def fake_quantize(quant_dir: Path) -> None:
        called["dir"] = quant_dir

    ctx = ProviderContext(out_dir=tmp_path, quantize_fn=fake_quantize)
    resolved = QuantizeProvider().provide(cfg, ctx)
    assert called["dir"] == tmp_path / "quantized"
    assert resolved.model == str(tmp_path / "quantized")
    assert resolved.metadata["scheme"] == "MXFP8"


def test_quantize_provider_without_fn_raises():
    cfg = parse_run_config({"model": "org/m", "quant": {"scheme": "MXFP8"}})
    with pytest.raises(AqaError, match="quantize_fn"):
        QuantizeProvider().provide(cfg, ProviderContext(out_dir=Path("/tmp/o")))


def test_endpoint_provider_resolves_endpoint():
    cfg = parse_run_config({"model": "org/m",
                            "provider": {"type": "endpoint", "base_url": "http://h:8000/v1"}})
    resolved = EndpointProvider().provide(cfg, ProviderContext(out_dir=Path("/tmp/o")))
    assert resolved.kind == "endpoint"
    assert resolved.endpoint == "http://h:8000/v1"
    assert resolved.served_model == "org/m"


def test_endpoint_provider_requires_base_url():
    cfg = parse_run_config({"model": "org/m", "provider": {"type": "endpoint"}})
    with pytest.raises(AqaError, match="base_url"):
        EndpointProvider().provide(cfg, ProviderContext(out_dir=Path("/tmp/o")))

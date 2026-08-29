"""Seed phase hooks: quantize/eval sanity validators.

* ``quant_sanity`` (after_quantize): regression hook for the 0-packed-layer
  incident -- an export that *claims* auto-round quantization
  (``quantization_config`` in a config.json) but whose safetensors contain no
  packed ``qweight`` tensor is a fake-quantized checkpoint; fail the phase
  loudly instead of evaluating bf16 weights under a quantized name.
* ``eval_sanity`` (after_benchmark): the leaderboard failure heuristic -- a
  benchmark scoring exactly 0 almost always means eval breakage; fail the phase
  so the fix-loop investigates (the error text feeds its signature).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..logging_util import AqaError, get_logger

_log = get_logger("hooks.sanity")

_AUTOROUND_METHODS = {"auto-round", "auto_round"}
# LLM exports carry config.json at the root; diffusion pipeline exports carry
# the quantized component (transformer/unet) in a subdir.
_COMPONENT_DIRS = ("", "transformer", "unet")


def _claiming_dirs(output_dir: Path, quant_cfg) -> list[Path]:
    """Dirs whose config.json declares an auto-round quantization."""
    fmt = getattr(quant_cfg, "export_format", "") if quant_cfg is not None else ""
    claiming: list[Path] = []
    for comp in _COMPONENT_DIRS:
        d = output_dir / comp if comp else output_dir
        cfg_path = d / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            qcfg = json.loads(cfg_path.read_text()).get("quantization_config")
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(qcfg, dict):
            continue
        method = str(qcfg.get("quant_method", ""))
        if method in _AUTOROUND_METHODS or fmt == "auto_round":
            claiming.append(d)
    return claiming


def _has_packed_weights(d: Path) -> bool | None:
    """True when any safetensors shard in ``d`` holds a packed qweight tensor.

    ``None`` when the check cannot be performed (safetensors missing or an
    unreadable shard) -- the hook then passes rather than guessing.  Header-only
    reads; no tensor data is loaded.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        _log.warning("quant_sanity: safetensors unavailable; check skipped")
        return None
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        return False
    for shard in shards:
        try:
            with safe_open(str(shard), framework="numpy") as fh:
                keys = fh.keys()
            if any("qweight" in key for key in keys):
                return True
        except Exception as exc:  # noqa: BLE001 - corrupt/partial shard
            _log.warning("quant_sanity: could not read %s: %s", shard, exc)
            return None
    return False


class QuantSanityHook:
    """after_quantize validator: claimed-but-unpacked exports fail the phase."""

    name = "quant_sanity"

    def after_quantize(self, run_cfg, quant_cfg, output_dir) -> None:
        out = Path(output_dir)
        claiming = _claiming_dirs(out, quant_cfg)
        if not claiming:
            return  # no quantization_config (e.g. a "fake" format): nothing to check
        packed = [_has_packed_weights(d) for d in claiming]
        if any(p is True for p in packed) or any(p is None for p in packed):
            return  # packed weights exist, or the check could not be performed
        raise AqaError(
            f"quantization produced 0 packed layers in {out}: the export declares "
            "auto-round (quantization_config) but no safetensors shard contains a "
            "qweight tensor -- failing the phase instead of shipping a "
            "fake-quantized checkpoint"
        )


class EvalSanityHook:
    """after_benchmark validator: an exactly-zero score means eval breakage."""

    name = "eval_sanity"

    def after_benchmark(self, run_cfg, recipe, result_or_exception, out_dir) -> None:
        if isinstance(result_or_exception, Exception):
            return  # harness failures surface through the runner's own error path
        result = result_or_exception
        name = getattr(result, "name", None) or getattr(recipe, "name", recipe)
        if getattr(result, "score", None) == 0:
            raise AqaError(
                f"benchmark {name} scored 0 (acc=0 => fail; all-zero scores usually "
                "mean eval breakage, not a bad model)"
            )

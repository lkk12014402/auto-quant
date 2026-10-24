"""Quantization executor: build and run the auto-round command.

Ports the deterministic quantization path from ``local_dispatch`` into a typed
builder.  When ``agent.optimize`` is on, the agent-produced command replaces the
one built here (see aqa/agent/optimize.py); otherwise this deterministic builder
is used.
"""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from ..config import QuantConfig
from ..logging_util import AqaError, get_logger

_log = get_logger("quantize")


def build_autoround_command(
    model: str, cfg: QuantConfig, output_dir: Path, *, num_gpus: int = 1
) -> list[str]:
    """Construct the ``auto-round`` CLI command for a quantization run.

    ``num_gpus`` drives device placement: with more than one visible GPU the
    model is sharded across cards via ``--device_map auto`` (unless the user set
    ``quant.device_map`` explicitly or already passed ``--device_map`` in
    ``extra_args``).
    """
    cmd = ["auto-round", "--model_name", model, "--scheme", cfg.scheme]
    if cfg.method == "model_free":
        cmd.append("--model_free")
    if cfg.algorithm:
        cmd += ["--algorithm", cfg.algorithm]
    if cfg.enable_alg_ext:
        cmd.append("--enable_alg_ext")
    if cfg.hadamard_type:
        cmd += ["--rotation-hadamard-type", cfg.hadamard_type]
    if cfg.opt_rtn is not None:
        cmd.append("--enable_opt_rtn" if cfg.opt_rtn else "--disable_opt_rtn")
    if cfg.iters is not None:
        cmd += ["--iters", str(cfg.iters)]
    if cfg.nsamples is not None:
        cmd += ["--nsamples", str(cfg.nsamples)]
    if cfg.quant_lm_head:
        cmd.append("--quant_lm_head")
    if cfg.static_kv_fp8:
        cmd += ["--static_kv_dtype", "fp8"]
    if cfg.static_attention_fp8:
        cmd += ["--static_attention_dtype", "fp8"]
    if cfg.ignore_layers:
        cmd += ["--ignore_layers", cfg.ignore_layers]
    if cfg.layer_config:
        # Per-layer mixed-precision overrides (e.g. MoE experts -> MXFP4 while the
        # rest stays MXFP8). auto-round's CLI parser prefers strict JSON, so emit
        # compact JSON rather than the unquoted `{k:{...}}` shorthand.
        cmd += ["--layer_config", json.dumps(cfg.layer_config, separators=(",", ":"))]
    device_map = resolve_device_map(cfg, num_gpus)
    if device_map and "--device_map" not in cfg.extra_args:
        cmd += ["--device_map", device_map]
    cmd += ["--format", cfg.export_format, "--output_dir", str(output_dir)]
    cmd += list(cfg.extra_args)
    return cmd


def resolve_export_dir(quant_dir: Path) -> Path:
    """Return the directory that actually holds the exported model.

    auto-round's model-load (tuning) path nests the export one level down at
    ``<output_dir>/<model>-<scheme>/``, while the model_free path writes flat
    into ``output_dir``.  When ``quant_dir`` has no config.json of its own and
    exactly one child directory does, resolve to that child.
    """
    quant_dir = Path(quant_dir)
    if (quant_dir / "config.json").exists() or not quant_dir.is_dir():
        return quant_dir
    candidates = sorted(
        d for d in quant_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
    )
    if len(candidates) == 1:
        _log.info("quantized export nested at %s; using it as the model dir",
                  candidates[0].name)
        return candidates[0]
    return quant_dir


def run_quantization(
    model: str, cfg: QuantConfig, output_dir: Path, *, num_gpus: int = 1
) -> Path:
    """Run auto-round quantization and return the output model directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_autoround_command(model, cfg, output_dir, num_gpus=num_gpus)
    return _run_autoround(cmd, output_dir)


def run_quant_command(cmd: str | list[str], output_dir: Path) -> Path:
    """Run a pre-built (e.g. agent-optimized) auto-round command; validate the tool exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    return _run_autoround(argv, output_dir)


def run_quant_script(script: str, output_dir: Path) -> Path:
    """Run a reproduced manifest's quantization script verbatim under ``bash``.

    Unlike :func:`run_quant_command` (one argv, no shell), this supports the
    INC recipe's multi-line shell scripts, environment prefixes and line
    continuations.  Callers (``aqa reproduce``) only pass a script the user
    explicitly asked to replay.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "reproduce_quant.sh"
    script_path.write_text("set -euo pipefail\n" + script.rstrip() + "\n",
                           encoding="utf-8")
    needs_autoround = ("auto-round" in script) or ("auto_round" in script)
    return _run_autoround(["bash", str(script_path)], output_dir,
                          require_autoround=needs_autoround)


def resolve_device_map(cfg: QuantConfig, num_gpus: int) -> str | None:
    """The effective ``--device_map`` value: explicit config, else auto on >1 GPU."""
    return cfg.device_map or ("auto" if num_gpus > 1 else None)


def ensure_device_map(cmd: str, cfg: QuantConfig, num_gpus: int) -> str:
    """Append ``--device_map <value>`` to an agent-built command when missing.

    Keeps the agent-optimize path multi-GPU aware without overriding a device_map
    the agent (or the user via ``quant.device_map``) already chose.
    """
    device_map = resolve_device_map(cfg, num_gpus)
    if not device_map or "--device_map" in shlex.split(cmd):
        return cmd
    return f"{cmd} --device_map {shlex.quote(device_map)}"


def _run_autoround(cmd: list[str], output_dir: Path,
                   *, require_autoround: bool = True) -> Path:
    _log.info("[CMD] %s", " ".join(shlex.quote(c) for c in cmd))
    if require_autoround and shutil.which("auto-round") is None:
        raise AqaError("auto-round is not installed; `pip install auto-round`")
    log_path = output_dir / "quantize.log"
    from ..proc import stream_subprocess

    rc = stream_subprocess(cmd, log_path)
    if rc != 0:
        raise AqaError(f"quantization failed (rc={rc}); see {log_path}")
    return output_dir

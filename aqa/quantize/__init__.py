"""Quantization executor: build and run the auto-round command.

Ports the deterministic quantization path from ``local_dispatch`` into a typed
builder.  When ``agent.optimize`` is on, the agent-produced command replaces the
one built here (see aqa/agent/optimize.py); otherwise this deterministic builder
is used.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from ..config import QuantConfig
from ..logging_util import AqaError, get_logger

_log = get_logger("quantize")


def build_autoround_command(
    model: str, cfg: QuantConfig, output_dir: Path
) -> list[str]:
    """Construct the ``auto-round`` CLI command for a quantization run."""
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
    if cfg.ignore_layers:
        cmd += ["--ignore_layers", cfg.ignore_layers]
    cmd += ["--format", cfg.export_format, "--output_dir", str(output_dir)]
    cmd += list(cfg.extra_args)
    return cmd


def run_quantization(model: str, cfg: QuantConfig, output_dir: Path) -> Path:
    """Run auto-round quantization and return the output model directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_autoround_command(model, cfg, output_dir)
    return _run_autoround(cmd, output_dir)


def run_quant_command(cmd: str | list[str], output_dir: Path) -> Path:
    """Run a pre-built (e.g. agent-optimized) auto-round command; validate the tool exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    return _run_autoround(argv, output_dir)


def _run_autoround(cmd: list[str], output_dir: Path) -> Path:
    _log.info("[CMD] %s", " ".join(shlex.quote(c) for c in cmd))
    if shutil.which("auto-round") is None:
        raise AqaError("auto-round is not installed; `pip install auto-round`")
    log_path = output_dir / "quantize.log"
    from ..proc import stream_subprocess

    rc = stream_subprocess(cmd, log_path)
    if rc != 0:
        raise AqaError(f"quantization failed (rc={rc}); see {log_path}")
    return output_dir

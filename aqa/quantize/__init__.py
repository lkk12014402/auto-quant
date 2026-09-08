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
from ..hooks import run_hook_chain
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


def run_quantization(
    model: str, cfg: QuantConfig, output_dir: Path, *, num_gpus: int = 1,
    hooks: tuple | None = None,
) -> Path:
    """Run auto-round quantization and return the output model directory.

    ``hooks`` is a resolved phase-hook chain (``aqa.hooks.resolve_hooks``); the
    ``after_quantize`` hooks run after post-export repair, and a validator's
    ``AqaError`` fails the phase (e.g. a claimed-but-unpacked export).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_autoround_command(model, cfg, output_dir, num_gpus=num_gpus)
    out = _run_autoround(cmd, output_dir)
    _post_export(model, output_dir)
    run_hook_chain("after_quantize", hooks, quant_cfg=cfg, output_dir=output_dir)
    return out


def run_quant_command(cmd: str | list[str], output_dir: Path, *,
                      hooks: tuple | None = None) -> Path:
    """Run a pre-built (e.g. agent-optimized) auto-round command; validate the tool exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    out = _run_autoround(argv, output_dir)
    _post_export(_model_name_of(argv), output_dir)
    run_hook_chain("after_quantize", hooks, quant_cfg=None, output_dir=output_dir)
    return out


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


# ── diffusion post-export repair ──────────────────────────────────────────────
# auto-round saves the non-quantized pipeline components (text_encoder, vae, ...)
# itself, and can mangle them: Z-Image's Qwen3 text encoder is re-saved with the
# inner `model.` prefix stripped from every weight, which breaks vllm-omni's
# loader ("no module or parameter named 'text_encoder.embed_tokens'").  These
# components are never quantized, so the safe repair is to restore the pristine
# directory from the source model whenever the exported one does not match.

_QUANTIZED_COMPONENTS = ("transformer", "unet")


def _safetensors_keys(path: Path) -> set[str] | None:
    """Union of tensor keys across all shards (header-only read). None if n/a."""
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - quant stack always has safetensors
        _log.warning("safetensors unavailable; skipping export component verification")
        return None
    keys: set[str] = set()
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return set() if (path / "config.json").exists() else None
    for shard in shards:
        try:
            with safe_open(str(shard), framework="numpy") as fh:
                keys.update(fh.keys())
        except Exception as exc:  # noqa: BLE001 - corrupt/partial shard
            _log.warning("could not read %s: %s", shard, exc)
            return None
    return keys


def _component_matches(src: Path, dst: Path) -> bool:
    if not dst.is_dir():
        return False
    src_keys, dst_keys = _safetensors_keys(src), _safetensors_keys(dst)
    if src_keys is None:
        return True  # nothing verifiable on the source side
    if dst_keys is None:
        return not src_keys  # export unreadable/empty: mismatch iff source has weights
    return src_keys == dst_keys


def repair_diffusion_export(model: str, output_dir: Path) -> list[str]:
    """Restore mangled non-quantized pipeline components from the source model.

    Only applies to diffusion pipeline exports (source dir with model_index.json);
    the quantized component (transformer/unet) is never touched.  Returns the list
    of repaired component names (empty = export verified clean / not applicable).
    """
    src = Path(model)
    if not src.is_dir() or not (src / "model_index.json").exists():
        return []
    repaired: list[str] = []
    for comp in sorted(src.iterdir()):
        if not comp.is_dir() or comp.name in _QUANTIZED_COMPONENTS:
            continue
        if _safetensors_keys(comp) is None:
            continue  # not a weight-bearing component (tokenizer/, scheduler/, ...)
        out_comp = output_dir / comp.name
        if _component_matches(comp, out_comp):
            continue
        _log.warning("exported component %r mismatch; restoring pristine copy from source", comp.name)
        shutil.rmtree(out_comp, ignore_errors=True)
        shutil.copytree(comp, out_comp)
        repaired.append(comp.name)
    if repaired:
        report = {"repaired_components": repaired, "source": str(src)}
        (output_dir / "aqa_export_repair.json").write_text(json.dumps(report, indent=2))
    return repaired


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


def restore_config_structure(model: str, output_dir: Path) -> bool:
    """Restore the exported config.json's structural fields from the source model.

    auto-round's model-load (tuning) path re-serializes config.json through the
    *installed* transformers' ``save_pretrained``, which normalizes fields to
    that version's schema — e.g. with transformers 5.16.1 a Qwen4-Exp
    checkpoint's ``layer_types`` entry "full_attention" comes back as
    "qwen_sparse_attention", which vllm's config validation then rejects.
    Repair by taking the source config verbatim and grafting on only the
    export's ``quantization_config``.  The pre-repair file is kept as
    ``config.json.autoround.bak``.  Returns True when a repair was applied.
    """
    src = Path(model)
    # Repo-id sources have no local config to restore from; diffusion pipelines
    # (model_index.json) keep their component-level repair instead.
    if not src.is_dir() or (src / "model_index.json").exists():
        return False
    export_dir = resolve_export_dir(output_dir)
    src_cfg_path, out_cfg_path = src / "config.json", export_dir / "config.json"
    if not src_cfg_path.exists() or not out_cfg_path.exists():
        return False
    src_cfg = json.loads(src_cfg_path.read_text())
    out_cfg = json.loads(out_cfg_path.read_text())
    quant_cfg = out_cfg.get("quantization_config")
    if quant_cfg is None:
        return False  # export carries no quant metadata; leave it untouched
    repaired = dict(src_cfg)
    repaired["quantization_config"] = quant_cfg
    if repaired == out_cfg:
        return False
    backup = out_cfg_path.with_name("config.json.autoround.bak")
    backup.write_text(json.dumps(out_cfg, indent=2, ensure_ascii=False))
    out_cfg_path.write_text(json.dumps(repaired, indent=2, ensure_ascii=False))
    _log.warning(
        "export config.json structure restored from source (auto-round re-serialization "
        "drift, e.g. layer_types renames); pre-repair copy kept as %s", backup.name)
    return True


def repair_fused_name_gaps(output_dir: Path) -> list[str]:
    """Fill fused-name gaps in the export's ``quantization_config.extra_config``.

    auto-round writes ``extra_config`` (the skip list) with the *unfused* module
    names from the checkpoint (``mlp.gate_proj`` / ``mlp.up_proj``).  vllm builds
    dense/shared-expert MLPs as a *fused* ``gate_up_proj`` (MergedColumnParallelLinear),
    and its INC quant-scope check matches names exactly — so the fused module
    misses the skip and gets built quantized while the checkpoint holds plain
    bf16 weights (KeyError ``...gate_up_proj.weight`` at load).  For every
    ``X.gate_proj`` entry with a sibling ``X.up_proj``, add ``X.gate_up_proj``
    with the same config.  Returns the list of fused names added.
    """
    export_dir = resolve_export_dir(output_dir)
    cfg_path = export_dir / "config.json"
    if not cfg_path.exists():
        return []
    cfg = json.loads(cfg_path.read_text())
    extra = (cfg.get("quantization_config") or {}).get("extra_config")
    if not isinstance(extra, dict):
        return []
    added: list[str] = []
    for key in list(extra.keys()):
        if not key.endswith(".gate_proj"):
            continue
        base = key[: -len(".gate_proj")]
        fused = f"{base}.gate_up_proj"
        if f"{base}.up_proj" in extra and fused not in extra:
            extra[fused] = dict(extra[key])
            added.append(fused)
    if added:
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        _log.info("post-export: added %d fused gate_up_proj entries to extra_config", len(added))
    return added


def _post_export(model: str | None, output_dir: Path) -> None:
    if not model:
        return
    try:
        repaired = repair_diffusion_export(model, output_dir)
        if repaired:
            _log.info("post-export repair restored components: %s", repaired)
    except Exception as exc:  # noqa: BLE001 - repair is best-effort, never blocks
        _log.warning("post-export repair failed (ignored): %s", exc)
    try:
        restore_config_structure(model, output_dir)
    except Exception as exc:  # noqa: BLE001 - repair is best-effort, never blocks
        _log.warning("post-export config restore failed (ignored): %s", exc)
    try:
        repair_fused_name_gaps(output_dir)
    except Exception as exc:  # noqa: BLE001 - repair is best-effort, never blocks
        _log.warning("post-export fused-name repair failed (ignored): %s", exc)


def _model_name_of(cmd: str | list[str]) -> str | None:
    toks = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    for flag in ("--model_name", "--model"):
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks):
                return toks[i + 1]
    return None

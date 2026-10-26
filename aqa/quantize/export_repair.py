"""Deterministic post-export repairs for exported checkpoints.

auto-round writes the quantization config in *checkpoint* names.  Some runtimes
address the same modules under a different namespace (GLM-5.3's vLLM model maps
``model.language_model.*`` to ``language_model.model.*`` and fuses
``gate_proj``/``up_proj`` into ``gate_up_proj``).  Rewriting that config by hand
inside a run directory is not reproducible; these helpers make the transform a
pipeline step with a backup and an audit record.

The diffusion helpers repair non-quantized pipeline components that auto-round
re-saves with mangled names; they never touch a quantized component.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..logging_util import AqaError, get_logger

_log = get_logger("quantize.export_repair")

_QUANT_MARKERS = ("qweight", "weight_packed", "weight_scale", "weight_global_scale")
_GLM_PREFIXES = (
    ("model.language_model.", "language_model.model."),
    ("model.visual.", "visual."),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AqaError(f"cannot read {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _resolve_export(quant_dir: Path) -> Path:
    quant_dir = Path(quant_dir)
    if (quant_dir / "config.json").is_file() or not quant_dir.is_dir():
        return quant_dir
    candidates = sorted(
        d for d in quant_dir.iterdir()
        if d.is_dir() and (d / "config.json").is_file())
    return candidates[0] if len(candidates) == 1 else quant_dir


def _shard_keys(component: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - safetensors is a hard runtime dep
        return set()
    keys: set[str] = set()
    for shard in sorted(component.glob("*.safetensors")):
        try:
            with safe_open(str(shard), framework="numpy") as fh:
                keys.update(fh.keys())
        except Exception:  # noqa: BLE001 - partial/corrupt shard means "unknown"
            return set()
    return keys


def _is_quantized_component(component: Path) -> bool:
    return any(any(marker in key for marker in _QUANT_MARKERS)
               for key in _shard_keys(component))


def repair_diffusion_export(source_dir: str | Path,
                            out_dir: str | Path) -> list[str]:
    """Restore mangled non-quantized pipeline components from ``source_dir``.

    Components that contain packed/scaled weights (the quantized transformer) are
    never restored.  Returns the repaired component names and writes
    ``aqa_export_repair.json`` when anything changed.
    """
    source = Path(source_dir)
    out = Path(out_dir)
    if not (source / "model_index.json").is_file():
        return []
    repaired: list[str] = []
    for component in sorted(source.iterdir()):
        if not component.is_dir():
            continue
        if not any(component.glob("*.safetensors")):
            continue
        target = out / component.name
        if target.is_dir() and _is_quantized_component(target):
            continue
        source_keys = _shard_keys(component)
        if not source_keys:
            continue
        if target.is_dir() and _shard_keys(target) == source_keys:
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(component, target)
        repaired.append(component.name)
        _log.info("export repair: restored %s from %s", component.name, source)
    if repaired:
        _write_json(out / "aqa_export_repair.json",
                    {"repaired_components": repaired})
    return repaired


def _nested_quant_config(out: Path) -> Path | None:
    direct = out / "config.json"
    if direct.is_file() and isinstance(
            _read_json(direct).get("quantization_config"), dict):
        return direct
    for path in sorted(out.glob("*/config.json")):
        if isinstance(_read_json(path).get("quantization_config"), dict):
            return path
    return None


def restore_config_structure(source_dir: str | Path,
                             out_dir: str | Path) -> bool:
    """Restore source ``config.json`` structure while preserving quantization.

    auto-round can rename architecture fields during re-serialization (e.g.
    ``text_config.layer_types``).  Only structural drift is repaired; a config
    that differs solely in ``quantization_config`` is left untouched.  The
    pre-repair file is kept as ``config.json.autoround.bak``.
    """
    source = Path(source_dir)
    out = Path(out_dir)
    if (source / "model_index.json").is_file():
        return False
    src_path = source / "config.json"
    if not src_path.is_file():
        return False
    target = _nested_quant_config(out)
    if target is None:
        return False
    src_cfg = _read_json(src_path)
    out_cfg = _read_json(target)
    src_struct = {k: v for k, v in src_cfg.items() if k != "quantization_config"}
    out_struct = {k: v for k, v in out_cfg.items() if k != "quantization_config"}
    if src_struct == out_struct:
        return False
    backup = target.with_name(target.name + ".autoround.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
    merged = dict(src_struct)
    merged["quantization_config"] = out_cfg["quantization_config"]
    _write_json(target, merged)
    _log.info("export repair: restored config structure in %s", target)
    return True


def repair_fused_name_gaps(quant_dir: str | Path) -> list[str]:
    """Add ``gate_up_proj`` entries when ``gate_proj``+``up_proj`` both exist.

    Only touches ``quantization_config.extra_config``; used for runtimes that
    look up the fused module name while auto-round recorded the split names.
    """
    out = _resolve_export(Path(quant_dir))
    path = out / "config.json"
    if not path.is_file():
        return []
    cfg = _read_json(path)
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return []
    extra = qcfg.get("extra_config")
    if not isinstance(extra, dict):
        return []
    added: list[str] = []
    for key in list(extra):
        if not key.endswith(".gate_proj"):
            continue
        fused = key[:-len(".gate_proj")] + ".gate_up_proj"
        up = key[:-len(".gate_proj")] + ".up_proj"
        if up in extra and fused not in extra:
            extra[fused] = extra[key]
            added.append(fused)
    if added:
        _write_json(path, cfg)
    return added


def _glm_runtime_name(name: str) -> str:
    for source, target in _GLM_PREFIXES:
        if name.startswith(source):
            return target + name[len(source):]
    if name == "lm_head":
        return "language_model.lm_head"
    return name


def repair_glm5_vllm(quant_dir: str | Path) -> dict[str, Any]:
    """Rewrite a GLM-5.3 auto-round config into vLLM's module namespace.

    vLLM's GLM model class has no ``packed_modules_mapping``, so a generic
    ``targets=["Linear"]`` + ``ignore`` config cannot match the fused
    ``gate_up_proj`` experts.  The reproducible form is an explicit target list
    derived from the exported index, expressed in vLLM module names.
    """
    out = _resolve_export(Path(quant_dir))
    cfg_path = out / "config.json"
    if not cfg_path.is_file():
        raise AqaError(f"export repair: {cfg_path} is missing")
    index_path = out / "model.safetensors.index.json"
    if not index_path.is_file():
        raise AqaError(f"export repair: {index_path} is missing")
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict):
        raise AqaError(f"export repair: {index_path} has no weight_map")
    packed = sorted({str(k)[:-len(".weight_packed")] for k in weight_map
                     if str(k).endswith(".weight_packed")})
    if not packed:
        raise AqaError("export repair: no weight_packed tensors found")
    targets = sorted({_glm_runtime_name(name) for name in packed})

    cfg = _read_json(cfg_path)
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        raise AqaError(f"export repair: {cfg_path} has no quantization_config")
    groups = qcfg.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise AqaError("export repair: quantization_config has no config_groups")
    group_id = next(
        (gid for gid, group in groups.items()
         if isinstance(group, dict)
         and isinstance(group.get("weights"), dict)
         and group["weights"].get("num_bits") == 4),
        next(iter(groups)))
    group = groups[group_id]
    if not isinstance(group, dict):
        raise AqaError(f"export repair: config group {group_id} is not an object")
    group["targets"] = targets
    qcfg["config_groups"] = {group_id: group}
    qcfg["ignore"] = []
    qcfg["format"] = "mxfp4-pack-quantized"

    backup = out / ".pre_export_repair_config.json"
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
    _write_json(cfg_path, cfg)
    _write_json(out / "quantization_config.json", qcfg)
    command = (f"python -m aqa.quantize.export_repair {out} "
               "--kind glm5_vllm")
    _log.info("export repair: wrote %d vLLM-namespace targets to %s",
              len(targets), cfg_path)
    return {"kind": "glm5_vllm", "targets": len(targets),
            "backup": str(backup), "command": command}


_REPAIRS = {"glm5_vllm": repair_glm5_vllm}


def repair_export(quant_dir: str | Path, kind: str) -> dict[str, Any]:
    """Apply a named post-export repair (``glm5_vllm``)."""
    if kind not in _REPAIRS:
        raise AqaError(f"unknown export repair kind {kind!r}; "
                       f"choose from {sorted(_REPAIRS)}")
    return _REPAIRS[kind](quant_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Repair an auto-round export "
                                                 "config for a runtime namespace.")
    parser.add_argument("quant_dir", help="exported model directory")
    parser.add_argument("--kind", required=True, choices=sorted(_REPAIRS))
    args = parser.parse_args(argv)
    info = repair_export(args.quant_dir, args.kind)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

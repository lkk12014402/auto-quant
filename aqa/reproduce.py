"""``aqa reproduce`` — replay a recipe variant and verify it (RFC 0002 §5).

The module is the **programmatic entry point** for external agents as well as the
implementation behind the CLI::

    from aqa.reproduce import reproduce_manifest, plan_reproduction

    plan = plan_reproduction("recipe.yaml", variant="mxfp4-v1")   # dry-run
    report = reproduce_manifest("recipe.yaml", variant="mxfp4-v1",
                                out_dir="./results/reproduce/qwen-mxfp4")

A reproduce run reuses the normal phase machine (QUANT→EVAL→CLOSE→VERIFY):
the manifest is adapted to a :class:`~aqa.config.RunConfig`, the frozen
quantization script runs verbatim, and a post-run VERIFY phase writes
``reproduction_report.json`` with the three checks from the RFC:

1. **input equivalence** — model source, quantization command, package versions;
2. **quantization structure** — per-module scheme vs the manifest's
   ``quantization_details``;
3. **evaluation numbers** — per-task scores vs the manifest's result table,
   with absolute + relative tolerances.

A reproduced run never mutates the Recipe KB, never uploads weights, and never
self-heals: it is a read-only verification of what the manifest claims.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ActionGuardConfig,
    AgentConfig,
    EvalConfig,
    ExecutorConfig,
    LmEvalOptions,
    OutputConfig,
    PolicyGateConfig,
    ProviderConfig,
    QuantConfig,
    RecipeKBConfig,
    ReproduceConfig,
    ReproduceTolerance,
    RunConfig,
)
from .logging_util import AqaError, get_logger
from .session._io import atomic_write_json

_log = get_logger("reproduce")

REPORT_FILE = "reproduction_report.json"
SCHEMA_VERSION = 1

_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")
_HF_URL_RE = re.compile(r"^https://huggingface\.co/([^/\s]+/[^/\s]+)/?$")
_SCHEME_RE = re.compile(r"(MXFP\d+|W\d+A\d+|NVFP\d+|FP\d+|INT\d+)", re.IGNORECASE)


# ───────────────────────── manifest / variant ─────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load + minimally validate an INC model-recipe manifest."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise AqaError(f"cannot read manifest {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise AqaError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AqaError(f"manifest {path} is not a mapping")
    if str(data.get("schema_version")) != "1.0":
        raise AqaError(f"manifest {path}: unsupported schema_version "
                       f"{data.get('schema_version')!r} (expected '1.0')")
    if not isinstance(data.get("model"), dict) or not data["model"].get("source"):
        raise AqaError(f"manifest {path}: model.source is required")
    if not isinstance(data.get("variants"), list) or not data["variants"]:
        raise AqaError(f"manifest {path}: variants[] is required")
    return data


def resolve_variant(manifest: dict[str, Any],
                    variant_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return ``(id, variant)``; auto-selects when the manifest has one variant."""
    variants = [v for v in manifest.get("variants", []) if isinstance(v, dict)]
    if not variants:
        raise AqaError("manifest has no variants")
    if variant_id:
        for variant in variants:
            if variant.get("id") == variant_id:
                return variant_id, variant
        ids = ", ".join(str(v.get("id")) for v in variants)
        raise AqaError(f"variant {variant_id!r} not found (available: {ids})")
    if len(variants) == 1:
        return str(variants[0].get("id")), variants[0]
    ids = ", ".join(str(v.get("id")) for v in variants)
    raise AqaError(f"manifest has multiple variants; pass --variant ({ids})")


def _operations(manifest: dict[str, Any], variant: dict[str, Any]) -> dict[str, dict]:
    ids = variant.get("recipe", {}).get("operations") or []
    ops = manifest.get("operations") or {}
    return {str(op_id): ops[op_id] for op_id in ids if op_id in ops}


def _setting_map(operation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for setting in operation.get("settings") or []:
        if isinstance(setting, dict) and setting.get("key"):
            out[str(setting["key"])] = setting.get("value")
    return out


def _first_linear_op(manifest: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    for op in _operations(manifest, variant).values():
        label = str(op.get("label", "")).lower()
        if "linear" in label or "embedding" in label:
            return op
    return {}


def _scheme_from_settings(settings: dict[str, Any]) -> str | None:
    data_type = settings.get("data_type")
    if isinstance(data_type, str) and data_type:
        return data_type
    text = str(settings.get("scheme") or "")
    match = _SCHEME_RE.search(text)
    return match.group(1) if match else None


def _exclude_layers(settings: dict[str, Any]) -> str | None:
    value = settings.get("exclude_layers")
    if isinstance(value, list):
        return ",".join(str(v) for v in value if str(v).strip()) or None
    if isinstance(value, str):
        return value or None
    return None


# ───────────────────────── script parsing ─────────────────────────


def _tokens(script: str) -> list[str]:
    normalized = re.sub(r"\\\s*\n\s*", " ", script or "")
    try:
        return shlex.split(normalized)
    except ValueError:
        return normalized.split()


def _opt(tokens: list[str], name: str) -> str | None:
    for i, token in enumerate(tokens):
        if token == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _all_opts(tokens: list[str], name: str) -> list[str]:
    out: list[str] = []
    for i, token in enumerate(tokens):
        if token == name and i + 1 < len(tokens):
            out.append(tokens[i + 1])
        elif token.startswith(name + "="):
            out.append(token.split("=", 1)[1])
    return out


def _flag(tokens: list[str], name: str) -> bool:
    return any(t == name or t.startswith(name + "=") for t in tokens)


def _to_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_kv(raw: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in (raw or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key:
            out[key] = _to_number(value.strip())
    return out


def _tasks(manifest: dict[str, Any], variant: dict[str, Any],
           tokens: list[str]) -> list[str]:
    raws = _all_opts(tokens, "--tasks") or _all_opts(tokens, "--task")
    if raws:
        out: list[str] = []
        for raw in raws:
            for task in raw.split(","):
                task = task.strip()
                if task and task not in out:
                    out.append(task)
        return out
    # Fall back to the result table's metric columns (task field).
    out: list[str] = []
    for table in manifest.get("result_tables") or []:
        for column in table.get("columns") or []:
            if column.get("kind") == "metric" and column.get("task"):
                task = str(column["task"])
                if task not in out:
                    out.append(task)
    return out


def _parse_eval_script(script: str, manifest: dict[str, Any],
                       variant: dict[str, Any]) -> EvalConfig:
    tokens = _tokens(script)
    tasks = _tasks(manifest, variant, tokens)
    if not tasks:
        raise AqaError("evaluation script has no --tasks and the manifest has no "
                       "metric result columns")
    backend = _opt(tokens, "--model") or ""
    if backend.startswith("vllm"):
        backend = "vllm"
    elif backend == "hf":
        backend = "hf"
    else:
        # INC's unified script (`bash run_lm_eval.sh`) hides the backend; infer
        # it from the script or the declared evaluation environment.
        env_text = json.dumps(
            (manifest.get("environments") or {}).get("evaluation") or {}).lower()
        backend = "vllm" if ("vllm" in script.lower() or "vllm" in env_text) else "hf"
    model_args = _parse_kv(_opt(tokens, "--model_args"))
    for key in ("pretrained", "tokenizer", "tensor_parallel_size",
                "max_model_len", "gpu_memory_utilization", "enable_prefix_caching"):
        model_args.pop(key, None)
    gen_kwargs = _parse_kv(_opt(tokens, "--gen_kwargs"))
    metadata: dict[str, Any] = {}
    raw_metadata = _opt(tokens, "--metadata")
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except json.JSONDecodeError:
            pass
    limit = _opt(tokens, "--limit")
    seed = _opt(tokens, "--seed")
    batch_size = _opt(tokens, "--batch_size") or "auto"
    off = str(model_args.get("enable_thinking", "")).lower() in ("false", "0", "no")
    # No explicit flag in the recipe: let the eval layer auto-detect the model's
    # thinking switch. Forcing "on" made reproduce plans claim a mode the
    # original run never used.
    thinking = "off" if off else "auto"
    off_args = {"enable_thinking": False}
    if model_args.get("reasoning_parser"):
        off_args["reasoning_parser"] = model_args["reasoning_parser"]
    return EvalConfig(
        benchmarks=tasks,
        backend=backend,
        harness="lm_eval",
        batch_size=batch_size,
        baseline=False,
        thinking=thinking,
        thinking_off_model_args=off_args,
        limit=int(limit) if limit and str(limit).isdigit() else None,
        lm_eval=LmEvalOptions(
            model_args=model_args,
            gen_kwargs=gen_kwargs,
            metadata=metadata,
            seed=int(seed) if seed and str(seed).lstrip("-").isdigit() else 42,
            apply_chat_template=True if _flag(tokens, "--apply_chat_template") else None,
            fewshot_as_multiturn=(
                True if _flag(tokens, "--fewshot_as_multiturn") else None),
        ),
    )


def _parse_quant_script(script: str, manifest: dict[str, Any],
                        variant: dict[str, Any]) -> QuantConfig:
    tokens = _tokens(script)
    op = _first_linear_op(manifest, variant)
    settings = _setting_map(op)
    scheme = _opt(tokens, "--scheme") or _scheme_from_settings(settings) or "MXFP4"
    iters_raw = _opt(tokens, "--iters")
    nsamples_raw = _opt(tokens, "--nsamples")
    method = "model_free" if _flag(tokens, "--model_free") else "autoround"
    layer_config: dict[str, Any] | None = None
    raw_layer_config = _opt(tokens, "--layer_config")
    if raw_layer_config:
        try:
            parsed = json.loads(raw_layer_config)
            if isinstance(parsed, dict):
                layer_config = parsed
        except json.JSONDecodeError:
            layer_config = None
    ignore = _opt(tokens, "--ignore_layers") or _exclude_layers(settings)
    opt_rtn: bool | None = None
    if _flag(tokens, "--enable_opt_rtn"):
        opt_rtn = True
    elif _flag(tokens, "--disable_opt_rtn"):
        opt_rtn = False
    return QuantConfig(
        scheme=scheme,
        method=method,
        algorithm=_opt(tokens, "--algorithm"),
        iters=int(iters_raw) if iters_raw and str(iters_raw).isdigit() else 0,
        nsamples=int(nsamples_raw) if nsamples_raw and str(nsamples_raw).isdigit() else None,
        export_format=_opt(tokens, "--format") or "llm_compressor",
        ignore_layers=ignore,
        layer_config=layer_config,
        quant_lm_head=_flag(tokens, "--quant_lm_head"),
        opt_rtn=opt_rtn,
        enable_alg_ext=_flag(tokens, "--enable_alg_ext"),
        static_kv_fp8=(_opt(tokens, "--static_kv_dtype") or "").lower() == "fp8",
        static_attention_fp8=(
            _opt(tokens, "--static_attention_dtype") or "").lower() == "fp8",
        device_map=_opt(tokens, "--device_map"),
        command_override=script,
    )


# ───────────────────────── variable binding ─────────────────────────


def _double_quote(value: str) -> str:
    return (value.replace("\\", "\\\\").replace('"', '\\"')
            .replace("`", "\\`").replace("$", "\\$"))


def _substitute(script: str, mapping: dict[str, str], *, where: str) -> str:
    text = script or ""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name not in mapping:
            if default is not None:
                return default
            return match.group(0)
        value = str(mapping[name])
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if before == '"' and after == '"':
            return _double_quote(value)
        if before == "'" and after == "'":
            return value.replace("'", "'\\''")
        return shlex.quote(value)

    text = _VAR_RE.sub(repl, text)
    unresolved = sorted({m.group(1) for m in _VAR_RE.finditer(text)})
    if unresolved:
        raise AqaError(f"{where}: undeclared variable(s) {unresolved}")
    return text


_OUTPUT_DIR_RE = re.compile(
    r"(--output_dir(?:=|\s+))(\"[^\"]*\"|'[^']*'|\S+)")


def _force_output_dir(script: str, target: Path) -> str:
    """Point the reproduced quantization at this run's ``quantized/`` dir.

    New INC scripts declare ``${OUTPUT_DIR}``; older aqa-emitted scripts may
    carry a literal path.  Rewrite either form so the replay never writes into
    the original run's directory.
    """
    quoted = shlex.quote(str(target))
    text, count = _OUTPUT_DIR_RE.subn(lambda m: m.group(1) + quoted, script)
    if count == 0 and re.search(r"\bauto[-_]round\b", text):
        text = text.rstrip() + " --output_dir " + quoted
    return text


def _bindings(block: dict[str, Any], manifest: dict[str, Any], *,
              out_dir: Path, mode: str, checkpoint: str | None) -> dict[str, str]:
    source = str(manifest["model"]["source"])
    quant_dir = out_dir / "quantized"
    mapping: dict[str, str] = {}
    for name, spec in (block.get("variables") or {}).items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("kind")
        origin = spec.get("from")
        if kind == "input_model" or origin == "model.source":
            mapping[name] = source
        elif kind == "output_dir":
            mapping[name] = str(quant_dir)
        elif kind == "checkpoint":
            if mode == "eval-only" and checkpoint:
                mapping[name] = checkpoint
            elif origin == "quantization.output_dir" or mode == "full":
                mapping[name] = str(quant_dir)
            else:
                mapping[name] = str(spec.get("default") or checkpoint or source)
        elif kind == "parameter":
            mapping[name] = str(spec.get("default") or "")
    return mapping


def _eval_only_checkpoint(manifest: dict[str, Any], variant: dict[str, Any],
                          evaluation: dict[str, Any]) -> str | None:
    hf_url = variant.get("hf_url")
    if hf_url:
        match = _HF_URL_RE.match(str(hf_url))
        if match:
            return match.group(1)
    spec = (evaluation.get("variables") or {}).get("MODEL_PATH")
    if isinstance(spec, dict):
        default = spec.get("default")
        if default and default not in (".", "./"):
            return str(default)
    return None


def _reproduction_model(manifest: dict[str, Any], variant: dict[str, Any],
                        evaluation: dict[str, Any], *, mode: str,
                        checkpoint: str | None) -> str:
    if mode == "full":
        return str(manifest["model"]["source"])
    if checkpoint:
        return checkpoint
    ckpt = _eval_only_checkpoint(manifest, variant, evaluation)
    if not ckpt:
        raise AqaError(
            "eval-only reproduce needs a checkpoint: pass --checkpoint or use a "
            "variant with hf_url / MODEL_PATH.default")
    return ckpt


# ───────────────────────── RunConfig adapter ─────────────────────────


def default_out_dir(manifest_path: str | Path, variant_id: str) -> Path:
    stem = Path(manifest_path).stem
    return Path.cwd() / "reproduce" / f"{stem}-{variant_id}"


def build_reproduction_config(
    manifest: dict[str, Any],
    variant: dict[str, Any],
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    mode: str = "full",
    checkpoint: str | None = None,
    abs_tol: float = 0.01,
    rel_tol: float = 0.02,
    on_input_mismatch: str = "warn",
) -> RunConfig:
    """Adapt one manifest variant to the normal aqa pipeline config."""
    if mode not in ("full", "eval-only"):
        raise AqaError("mode must be 'full' or 'eval-only'")
    variant_id = str(variant.get("id"))
    evaluation = variant.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("script"):
        raise AqaError(f"variant {variant_id!r} has no evaluation command")
    out_dir = Path(out_dir).resolve()
    model = _reproduction_model(
        manifest, variant, evaluation, mode=mode, checkpoint=checkpoint)
    eval_script = _substitute(
        str(evaluation.get("script")), _bindings(
            evaluation, manifest, out_dir=out_dir, mode=mode,
            checkpoint=(model if mode == "eval-only" else checkpoint)),
        where=f"variant {variant_id} evaluation")
    eval_cfg = _parse_eval_script(eval_script, manifest, variant)

    quant: QuantConfig | None = None
    provider = ProviderConfig(type="pretrained", upload=None)
    if mode == "full":
        quantization = variant.get("quantization")
        if not isinstance(quantization, dict) or not quantization.get("script"):
            raise AqaError(f"variant {variant_id!r} has no quantization command; "
                           "use --mode eval-only for an incomplete variant")
        quant_script = _substitute(
            str(quantization.get("script")), _bindings(
                quantization, manifest, out_dir=out_dir, mode="full",
                checkpoint=checkpoint),
            where=f"variant {variant_id} quantization")
        quant_script = _force_output_dir(quant_script, out_dir / "quantized")
        quant = _parse_quant_script(quant_script, manifest, variant)
        provider = ProviderConfig(type="quantize", quant=quant, upload=None)

    agent = AgentConfig(fix_loop=False, optimize=False, critic=False).model_copy(
        update={
            "recipe_kb": RecipeKBConfig(enabled=False, pin=False),
            "policy_gate": PolicyGateConfig(enabled=False),
            "action_guard": ActionGuardConfig(enabled=False),
        })
    return RunConfig(
        model=model,
        provider=provider,
        eval=eval_cfg,
        gate=None,
        agent=agent,
        executor=ExecutorConfig(type="local"),
        output=OutputConfig(dir=str(out_dir), sinks=[], unique_dir=False),
        reproduce=ReproduceConfig(
            manifest=str(Path(manifest_path).resolve()), variant=variant_id,
            mode=mode, tolerance=ReproduceTolerance(abs=abs_tol, rel=rel_tol),
            on_input_mismatch=on_input_mismatch),
    )


def expected_results(manifest: dict[str, Any], variant_id: str) -> dict[str, float]:
    """Task→score expected by the manifest's result table for this variant."""
    out: dict[str, float] = {}
    for table in manifest.get("result_tables") or []:
        for row in table.get("rows") or []:
            if row.get("kind") != "variant" or row.get("variant") != variant_id:
                continue
            values = row.get("values") or {}
            for column in table.get("columns") or []:
                if column.get("kind") != "metric" or not column.get("task"):
                    continue
                value = values.get(column.get("id"))
                if isinstance(value, (int, float)):
                    out[str(column["task"])] = float(value)
    return out


def plan_reproduction(
    manifest_path: str | Path,
    *,
    variant: str | None = None,
    mode: str = "full",
    out_dir: str | Path | None = None,
    checkpoint: str | None = None,
    abs_tol: float = 0.01,
    rel_tol: float = 0.02,
    on_input_mismatch: str = "warn",
) -> dict[str, Any]:
    """Resolve a manifest into an executable plan (no GPU / no file writes)."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    variant_id, variant_data = resolve_variant(manifest, variant)
    target = Path(out_dir or default_out_dir(manifest_path, variant_id)).resolve()
    cfg = build_reproduction_config(
        manifest, variant_data, manifest_path=manifest_path, out_dir=target,
        mode=mode, checkpoint=checkpoint, abs_tol=abs_tol, rel_tol=rel_tol,
        on_input_mismatch=on_input_mismatch)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "variant": variant_id,
        "mode": mode,
        "out_dir": str(target),
        "model": cfg.model,
        "quantization": {
            "environment": "quantization",
            "script": cfg.quant.command_override if cfg.quant else None,
        } if mode == "full" else None,
        "evaluation": {
            "backend": cfg.eval.backend,
            "benchmarks": list(cfg.eval.benchmarks),
            "thinking": cfg.eval.thinking,
            "model_args": dict(cfg.eval.lm_eval.model_args),
            "batch_size": cfg.eval.batch_size,
        },
        "expected_results": expected_results(manifest, variant_id),
        "tolerance": {"abs": abs_tol, "rel": rel_tol},
        "on_input_mismatch": on_input_mismatch,
    }


# ───────────────────────── verification ─────────────────────────


def _normalized_command(command: str) -> str:
    return " ".join(_tokens(command))


def _pkg_name(name: str) -> str:
    return str(name).lower().replace("_", "-")


def _package_diffs(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    expected_pkgs = {}
    for env in expected.values():
        if isinstance(env, dict):
            for pkg in env.get("packages") or []:
                if "==" in str(pkg):
                    name, version = str(pkg).split("==", 1)
                    expected_pkgs[_pkg_name(name)] = version
    actual_pkgs = {_pkg_name(k): str(v) for k, v in (actual or {}).items()}
    for name, version in sorted(expected_pkgs.items()):
        got = actual_pkgs.get(name)
        if got is None:
            diffs.append(f"package {name}=={version} not recorded in env_lock")
        elif got != version:
            diffs.append(f"package {name}: expected {version}, got {got}")
    return diffs


def _compare_quantization(expected: dict[str, Any],
                          actual: dict[str, Any]) -> dict[str, Any]:
    exp_groups = {str(g.get("id")): g for g in expected.get("module_groups") or []}
    act_groups = {str(g.get("id")): g for g in actual.get("module_groups") or []}
    mismatches: list[dict[str, Any]] = []
    for group_id in sorted(set(exp_groups) | set(act_groups)):
        exp = exp_groups.get(group_id)
        act = act_groups.get(group_id)
        if exp is None:
            mismatches.append({"id": group_id, "expected": None,
                               "got": act.get("scheme")})
        elif act is None:
            mismatches.append({"id": group_id, "expected": exp.get("scheme"),
                               "got": None})
        elif exp.get("scheme") != act.get("scheme"):
            mismatches.append({"id": group_id, "expected": exp.get("scheme"),
                               "got": act.get("scheme")})
    return {"status": "pass" if not mismatches else "fail",
            "group_mismatches": mismatches}


def _actual_quant_details(run_cfg: RunConfig, out_dir: Path) -> dict[str, Any]:
    from .quantize import resolve_export_dir
    from .report.quant_details import compute_quantization_details

    quant = run_cfg.quant
    if quant is None:
        return {}
    quant_dir = resolve_export_dir(out_dir / "quantized")
    model_cfg = _read_json(quant_dir / "config.json")
    return compute_quantization_details(
        model_cfg=model_cfg, declared_scheme=quant.scheme, quant_dir=quant_dir,
        model_source=run_cfg.model, declared_ignore=quant.ignore_layers,
        quant_lm_head=quant.quant_lm_head, layer_config=quant.layer_config,
        kv_scheme="FP8" if quant.static_kv_fp8 else None,
        attn_scheme="FP8" if quant.static_attention_fp8 else None)


def _evaluation_check(expected: dict[str, float], actual: dict[str, float],
                      *, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    if not expected:
        return {"status": "inconclusive",
                "note": "manifest has no variant result row to compare",
                "tolerance": {"abs": abs_tol, "rel": rel_tol},
                "per_benchmark": []}
    if not actual:
        return {"status": "fail", "note": "eval/accuracy.json has no results",
                "tolerance": {"abs": abs_tol, "rel": rel_tol},
                "per_benchmark": []}
    rows: list[dict[str, Any]] = []
    for task, exp in sorted(expected.items()):
        got = actual.get(task)
        if got is None:
            rows.append({"id": task, "expected": exp, "got": None, "within": False})
            continue
        delta = abs(got - exp)
        rel = delta / abs(exp) if exp else None
        within = delta <= abs_tol or (rel is not None and rel <= rel_tol)
        rows.append({"id": task, "expected": exp, "got": got,
                     "abs_diff": round(delta, 6),
                     "rel_diff": round(rel, 6) if rel is not None else None,
                     "within": within})
    return {"status": "pass" if all(r["within"] for r in rows) else "fail",
            "tolerance": {"abs": abs_tol, "rel": rel_tol},
            "per_benchmark": rows}


def verify_reproduction(
    manifest: dict[str, Any],
    variant: dict[str, Any],
    *,
    out_dir: str | Path,
    mode: str,
    run_cfg: RunConfig,
    abs_tol: float = 0.01,
    rel_tol: float = 0.02,
    on_input_mismatch: str = "warn",
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the three RFC checks against a completed reproduce run."""
    out_dir = Path(out_dir)
    variant_id = str(variant.get("id"))
    run_manifest = _read_json(out_dir / "run_manifest.json")
    env_lock = _read_json(out_dir / "env_lock.json")

    diffs: list[str] = []
    if mode == "eval-only":
        expected_ckpt = _eval_only_checkpoint(
            manifest, variant, variant.get("evaluation") or {})
        comparable = bool(expected_ckpt) and (
            re.fullmatch(r"[^/\s]+/[^/\s]+", str(expected_ckpt)) is not None
            or Path(str(expected_ckpt)).is_absolute())
        if comparable and expected_ckpt != run_cfg.model:
            diffs.append(f"checkpoint: expected {expected_ckpt}, got {run_cfg.model}")
    else:
        expected_source = str(manifest["model"]["source"])
        if expected_source != run_cfg.model:
            diffs.append(f"model.source: expected {expected_source}, got {run_cfg.model}")
        actual_cmd = str(_read_json(out_dir / ".quant_command.json").get("command") or "")
        expected_cmd = run_cfg.quant.command_override if run_cfg.quant else None
        if expected_cmd and _normalized_command(expected_cmd) != _normalized_command(actual_cmd):
            diffs.append("quantization command differs from the manifest script")
    diffs.extend(_package_diffs(
        manifest.get("environments") or {}, env_lock.get("packages") or {}))
    input_status = "match" if not diffs else "mismatch"

    if mode == "eval-only":
        quant_status = {"status": "skipped", "note": "eval-only mode"}
    else:
        expected_details = (manifest.get("quantization_details") or {}).get(variant_id)
        if not isinstance(expected_details, dict) or not expected_details.get("module_groups"):
            quant_status = {"status": "skipped",
                            "note": "manifest has no quantization_details for this variant"}
        else:
            quant_status = _compare_quantization(
                expected_details, _actual_quant_details(run_cfg, out_dir))

    from .report import read_eval_result

    try:
        actual_result = read_eval_result(out_dir / "eval")
        actual_scores = {n: b.score for n, b in actual_result.benchmarks.items()}
    except Exception:  # noqa: BLE001 - missing eval is a failed check, not a crash
        actual_scores = {}
    eval_status = _evaluation_check(
        expected_results(manifest, variant_id), actual_scores,
        abs_tol=abs_tol, rel_tol=rel_tol)
    if input_status == "mismatch" and on_input_mismatch == "warn" \
            and eval_status["status"] == "pass":
        eval_status = dict(eval_status)
        eval_status["status"] = "inconclusive"
        eval_status["note"] = "input/environment mismatch; numbers are not comparable"

    statuses = [eval_status.get("status")]
    if mode == "full":
        statuses.append(quant_status.get("status"))
    if input_status == "mismatch" and on_input_mismatch == "fail":
        statuses.append("fail")
    if "fail" in statuses:
        verdict = "not_reproduced"
    elif input_status == "mismatch" or any(
            s in ("skipped", "inconclusive") for s in statuses):
        verdict = "inconclusive"
    else:
        verdict = "reproduced"

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path or ""),
        "variant": variant_id,
        "mode": mode,
        "run_id": run_manifest.get("run_id"),
        "output_dir": str(out_dir.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_equivalence": {"status": input_status, "diffs": diffs},
        "quantization_reproduction": quant_status,
        "evaluation_reproduction": eval_status,
        "verdict": verdict,
    }


def verify_and_write(run_cfg: RunConfig, out_dir: str | Path,
                     result: Any = None, baseline: Any = None) -> dict[str, Any]:
    """Pipeline hook: verify the current run and write ``reproduction_report.json``."""
    cfg = run_cfg.reproduce
    if cfg is None:
        return {}
    out_dir = Path(out_dir)
    try:
        manifest = load_manifest(cfg.manifest)
        variant_id, variant = resolve_variant(manifest, cfg.variant)
        report = verify_reproduction(
            manifest, variant, out_dir=out_dir, mode=cfg.mode, run_cfg=run_cfg,
            abs_tol=cfg.tolerance.abs, rel_tol=cfg.tolerance.rel,
            on_input_mismatch=cfg.on_input_mismatch, manifest_path=cfg.manifest)
    except Exception as exc:  # noqa: BLE001 - verify must never hide the run
        report = {
            "schema_version": SCHEMA_VERSION, "manifest": cfg.manifest,
            "variant": cfg.variant, "mode": cfg.mode,
            "output_dir": str(out_dir.resolve()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_equivalence": {"status": "mismatch", "diffs": [str(exc)]},
            "quantization_reproduction": {"status": "skipped"},
            "evaluation_reproduction": {"status": "inconclusive"},
            "verdict": "not_reproduced",
            "error": f"{type(exc).__name__}: {exc}",
        }
    atomic_write_json(out_dir / REPORT_FILE, report)
    return report


# ───────────────────────── public entry point ─────────────────────────


def reproduce_manifest(
    manifest_path: str | Path,
    *,
    variant: str | None = None,
    mode: str = "full",
    out_dir: str | Path | None = None,
    checkpoint: str | None = None,
    abs_tol: float = 0.01,
    rel_tol: float = 0.02,
    on_input_mismatch: str = "warn",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Reproduce one recipe variant and return the verification report.

    This is the external-agent entry point: it never raises for a reproducible
    mismatch (the report's ``verdict`` carries the outcome) and accepts a
    ``dry_run`` that returns the resolved plan without touching the GPU.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    variant_id, variant_data = resolve_variant(manifest, variant)
    target = Path(out_dir or default_out_dir(manifest_path, variant_id)).resolve()

    if dry_run:
        return plan_reproduction(
            manifest_path, variant=variant_id, mode=mode, out_dir=target,
            checkpoint=checkpoint, abs_tol=abs_tol, rel_tol=rel_tol,
            on_input_mismatch=on_input_mismatch)

    if target.exists() and any(target.iterdir()):
        if not force:
            raise AqaError(
                f"reproduce output dir is not empty: {target} "
                "(pass force=True / --force to use a timestamped sibling)")
        target = target.with_name(
            f"{target.name}-{time.strftime('%Y%m%d-%H%M%S')}")

    cfg = build_reproduction_config(
        manifest, variant_data, manifest_path=manifest_path, out_dir=target,
        mode=mode, checkpoint=checkpoint, abs_tol=abs_tol, rel_tol=rel_tol,
        on_input_mismatch=on_input_mismatch)

    from .pipeline import run_pipeline

    target.mkdir(parents=True, exist_ok=True)
    # Config snapshot: a programmatic reproduce run has no owner-authored
    # run.yaml, so write the resolved config for artifact checks/reproduction.
    try:
        (target / "run.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=False,
                           allow_unicode=True, width=100), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - snapshot is best-effort
        _log.warning("run.yaml snapshot failed: %s", exc)
    meta_path = target / "aqa_run.json"
    meta_path.write_text(json.dumps({"model": cfg.model, "state": "running"}))
    try:
        run_pipeline(cfg, target, resume=False)
        meta_path.write_text(json.dumps({"model": cfg.model, "state": "done"}))
    except Exception as exc:  # noqa: BLE001 - report the failure, don't hide it
        meta_path.write_text(json.dumps({"model": cfg.model, "state": "failed"}))
        report = {
            "schema_version": SCHEMA_VERSION, "manifest": str(manifest_path),
            "variant": variant_id, "mode": mode, "output_dir": str(target),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_equivalence": {"status": "mismatch", "diffs": []},
            "quantization_reproduction": {"status": "skipped"},
            "evaluation_reproduction": {"status": "inconclusive"},
            "verdict": "not_reproduced",
            "error": f"{type(exc).__name__}: {exc}",
        }
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / REPORT_FILE, report)
        return report

    report = _read_json(target / REPORT_FILE)
    if not report:
        report = verify_and_write(cfg, target)
    return report

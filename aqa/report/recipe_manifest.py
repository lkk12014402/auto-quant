"""``recipe.yaml`` emitter — the INC-Recipes-compatible model manifest (RFC 0002).

The CLOSE phase already writes aqa's internal reproducibility manifest
(``manifest.json``).  This module projects a *quantize* run onto the
INC-Quantization-Recipes model-recipe format (schema ``1.0``, protocol 1.0) so a
finished aqa run drops a ``recipe.yaml`` that the recipe site can ingest
directly, plus two aqa-native extensions that the strict INC schema does not
model:

* ``quantization_details`` — the per-module precision/coverage map
  (:mod:`aqa.report.quant_details`).
* ``provenance`` — the aqa run id / gate verdict that produced the manifest.

Both extensions are *top-level and strippable*: :func:`to_inc_strict` removes
them so the remaining core validates against the INC schema
(``additionalProperties: false`` everywhere).  One aqa run == one *variant*;
merging many variants into a model catalogue is a separate concern.

The emitter follows the current INC template (``docs/model-id-template.yaml``):

* commands are templates with declared ``MODEL`` / ``OUTPUT_DIR`` /
  ``MODEL_PATH`` variables — no absolute local paths;
* only schema fields are emitted (no ``status``, ``working_directory``,
  ``tasks`` or per-column ``metric``);
* result columns name the evaluated task; rows carry raw fractions and a
  reported ratio;
* a variant with un-fingerprinted numbers or a non-GO gate lists ``issues``
  instead of pretending to be a complete, verified recipe.

Everything is deterministic (no GPU / no network) and best-effort: a failure is
logged and never aborts the run.  Inputs are the structured artifacts already
written earlier in the pipeline (``env_lock.json``, ``quant_plan.json``,
``eval_report.json``) plus the live ``RunConfig`` / ``EvalResult``.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..logging_util import get_logger
from ..session._io import now_iso
from .names import canonical_artifact_name, _lowest_bit_scheme
from .owner_actions import load_owner_actions
from .quant_details import (
    _scheme_weight_bits,
    _slug,
    compute_quantization_details,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig
    from ..eval.registry import EvalResult

_log = get_logger("report.recipe_manifest")

RECIPE_FILE = "recipe.yaml"

# Which tracked packages describe each environment (INC splits quant vs eval).
_QUANT_PKGS = {"torch", "transformers", "auto-round", "auto_round", "accelerate",
               "datasets", "huggingface-hub", "compressed-tensors", "autoquant-agent"}
_EVAL_PKGS = {"torch", "transformers", "vllm", "lm-eval", "lm_eval", "datasets",
              "huggingface-hub"}
# Display-name canonicalisation (underscore aliases collapse onto hyphen form).
_PKG_CANON = {"auto_round": "auto-round", "lm_eval": "lm-eval"}

_REPO_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


# ───────────────────────── small helpers ─────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _model_basename(model: str) -> str:
    return model.rstrip("/").split("/")[-1] or model


def _is_repo_id(value: str | None) -> bool:
    return bool(value and _REPO_ID_RE.fullmatch(str(value).strip().rstrip("/")))


def _model_source(run_cfg: "RunConfig") -> str:
    """The original model identity: prefer the HF repo id over a local path."""
    return str(run_cfg.output.report_model or run_cfg.model)


def _element_format(scheme: str) -> str | None:
    s = scheme.upper()
    if "MXFP8" in s or s in {"FP8", "W8A8"}:
        return "E4M3"
    if "MXFP4" in s:
        return "E2M1"
    if "NVFP4" in s or "FP4" in s:
        return "E2M1"
    return None


def _activation_scheme(scheme: str) -> str:
    m = re.match(r"^W(\d+)A(\d+)$", scheme.upper())
    if m:
        return f"Weight {m.group(1)}-bit / Activation {m.group(2)}-bit"
    return f"Weight {scheme} / Activation {scheme}"


def _scheme_settings(scheme: str) -> list[tuple[str, str, Any]]:
    """Canonical INC settings for a scheme (template `operations` keys)."""
    s = scheme.upper()
    if "MXFP4" in s:
        return [
            ("data_type", "Data type", "MXFP4"),
            ("element_format", "Element format", "E2M1"),
            ("scheme", "Scheme", "Weight – MXFP4, Activation – MXFP4"),
            ("weight_scheme", "Weight scheme", "Static, block-wise, RCEIL"),
            ("activation_scheme", "Activation scheme", "Dynamic, block-wise, RCEIL"),
            ("block_size", "Block size", 32),
            ("scale_type", "Scale type", "FP8 E8M0"),
            ("scale_method", "Scale method", "1D block-wise scaling for MXFP4"),
            ("extra_tensor_scaling", "Extra tensor scaling", "None"),
        ]
    if "MXFP8" in s:
        return [
            ("data_type", "Data type", "MXFP8"),
            ("element_format", "Element format", "E4M3"),
            ("scheme", "Scheme", "Weight – MXFP8, Activation – MXFP8"),
            ("weight_scheme", "Weight scheme", "Static, block-wise, RCEIL"),
            ("activation_scheme", "Activation scheme", "Dynamic, block-wise, RCEIL"),
            ("block_size", "Block size", 32),
            ("scale_type", "Scale type", "FP8 E8M0"),
            ("scale_method", "Scale method", "1D block-wise scaling for MXFP8"),
            ("extra_tensor_scaling", "Extra tensor scaling", "None"),
        ]
    if s.startswith("W4A16"):
        return [
            ("data_type", "Data type", "INT4"),
            ("scheme", "Scheme", "Weight 4-bit / Activation 16-bit"),
            ("weight_scheme", "Weight scheme", "Static, group-wise, symmetric"),
            ("group_size", "Group size", 128),
        ]
    if s.startswith("W8A16"):
        return [
            ("data_type", "Data type", "INT8"),
            ("scheme", "Scheme", "Weight 8-bit / Activation 16-bit"),
            ("weight_scheme", "Weight scheme", "Static, group-wise, symmetric"),
            ("group_size", "Group size", 128),
        ]
    if "NVFP4" in s:
        return [
            ("data_type", "Data type", "NVFP4"),
            ("element_format", "Element format", "E2M1"),
            ("scheme", "Scheme", "Weight – NVFP4, Activation – NVFP4"),
            ("weight_scheme", "Weight scheme", "Static, per-group, RNE"),
            ("group_size", "Group size", 16),
            ("scale_type", "Scale type", "FP8 E4M3"),
        ]
    if "FP8" in s:
        return [
            ("data_type", "Data type", "FP8"),
            ("element_format", "Element format", "E4M3"),
            ("scheme", "Scheme", "Static, per-tensor, RNE"),
            ("scale_type", "Scale type", "FP32"),
        ]
    return [
        ("data_type", "Data type", scheme),
        ("scheme", "Scheme", _activation_scheme(scheme)),
    ]


def _environments(env_lock: dict[str, Any]) -> dict[str, Any]:
    """Split the frozen package set into INC ``quantization``/``evaluation`` envs."""
    pkgs: dict[str, str] = env_lock.get("packages") or {}
    acc = env_lock.get("accelerator") or {}
    acc_str = None
    if acc:
        fam = acc.get("device_name") or acc.get("family")
        n = acc.get("num_devices")
        acc_str = f"{fam} x{n}" if fam and n else (str(fam) if fam else None)

    def _build(allow: set[str]) -> dict[str, Any] | None:
        picked: dict[str, str] = {}
        for name, ver in pkgs.items():
            if name not in allow or not ver:
                continue
            canon = _PKG_CANON.get(name, name)
            picked.setdefault(canon, ver)  # first (canonical) wins, keeps unique
        if not picked:
            return None
        entry: dict[str, Any] = {
            "packages": sorted(f"{n}=={v}" for n, v in picked.items())}
        if env_lock.get("python"):
            entry["python"] = env_lock["python"]
        if acc_str:
            entry["accelerator"] = acc_str
        return entry

    envs: dict[str, Any] = {}
    q = _build(_QUANT_PKGS)
    e = _build(_EVAL_PKGS)
    if q:
        envs["quantization"] = q
    if e:
        envs["evaluation"] = e
    # The schema requires the key, not a non-empty map; the INC template uses
    # `environments: {}` when no command/environment is known.
    return envs


# Group families that are never a Linear quantization target (precision-critical
# or non-Linear): embeddings, MoE router, norms, misc.  They are structurally out
# of the Linear operation's scope, so they are not listed in ``exclude_layers``
# (mirrors the INC convention + auto-round's always-keep set).
_NON_QUANTIZABLE_GROUP_FAMILIES = frozenset({"embed", "router", "norms", "other"})


def _exclude_layers(details: dict[str, Any]) -> list[str]:
    """Stable ids of quantization-candidate families kept fully BF16.

    A group is an exclusion iff it is BF16, belongs to a quantizable family (not
    embeddings/router/norms), and is a *whole* family kept BF16 — not the ``-bf16``
    remainder of a family that was mostly quantized (those partial keeps live in
    ``quantization_details.module_groups``, not in the Linear op's exclude list).
    """
    out: list[str] = []
    for g in details.get("module_groups", []):
        gid = g.get("id", "")
        if g.get("scheme") != "BF16" or gid.endswith("-bf16"):
            continue
        family = gid[:-len("-quant")] if gid.endswith("-quant") else gid
        if family in _NON_QUANTIZABLE_GROUP_FAMILIES:
            continue
        out.append(gid)
    return out


def _operations(scheme: str, details: dict[str, Any],
                kv_scheme: str | None, attn_scheme: str | None,
                ) -> tuple[dict[str, Any], list[str], list[str]]:
    """Synthesise operations from the actual per-module dispositions.

    A mixed run (e.g. MXFP8 body + MXFP4 experts) naturally yields more than one
    Linear operation; each gets its own settings block.  Returns
    ``(operations, ordered_ids, labels)``.
    """
    exclude = _exclude_layers(details)
    specs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for group in details.get("module_groups", []):
        op_id = group.get("operation")
        if not op_id or op_id in seen:
            continue
        seen.add(op_id)
        op_scheme = str(group.get("scheme") or scheme)
        label = "Linear (MOE)" if group.get("family") == "moe-experts" else "Linear"
        specs.append((str(op_id), label, op_scheme))
    if not specs:
        specs.append((f"{_slug(scheme)}-linear", "Linear", scheme))

    ops: dict[str, Any] = {}
    order: list[str] = []
    labels: list[str] = []
    used_labels: set[str] = set()
    for op_id, label, op_scheme in specs:
        if label in used_labels:
            label = f"{label} ({op_scheme})"
        used_labels.add(label)
        settings = [
            {"key": key, "label": lab, "value": value}
            for key, lab, value in _scheme_settings(op_scheme)
        ]
        if exclude:
            settings.append({"key": "exclude_layers", "label": "Exclude layers",
                             "value": exclude})
        ops[op_id] = {"label": label, "settings": settings}
        order.append(op_id)
        labels.append(label)
    if kv_scheme:
        ops["fp8-kv-cache"] = {"label": "KV cache", "settings": [
            {"key": "data_type", "label": "Data type", "value": "FP8"},
            {"key": "element_format", "label": "Element format", "value": "E4M3"},
            {"key": "scheme", "label": "Scheme", "value": "Static, Per-tensor, RNE"},
            {"key": "scale_type", "label": "Scale type", "value": "FP32"}]}
        order.append("fp8-kv-cache")
        labels.append("KV cache")
    if attn_scheme:
        ops["fp8-attention"] = {"label": "Attention", "settings": [
            {"key": "data_type", "label": "Data type", "value": "FP8"},
            {"key": "element_format", "label": "Element format", "value": "E4M3"},
            {"key": "scheme", "label": "Scheme", "value": "Static, Per-tensor, RNE"},
            {"key": "scale_type", "label": "Scale type", "value": "FP32"}]}
        order.append("fp8-attention")
        labels.append("Attention")
    return ops, order, labels


def _shell_token(token: str) -> str:
    if "${" in token:
        escaped = token.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        return f'"{escaped}"'
    return shlex.quote(token)


def _format_script(tokens: list[str]) -> str:
    """One flag per line, keeping its value on the same line."""
    if not tokens:
        return ""
    lines = [tokens[0]]
    for token in tokens[1:]:
        if token.startswith("-"):
            lines.append("    " + _shell_token(token))
        elif lines:
            lines[-1] += " " + _shell_token(token)
        else:  # pragma: no cover - the first token is always present
            lines.append(_shell_token(token))
    return " \\\n".join(lines)


_PATH_FLAGS = {"--tokenizer", "--calib_data", "--dataset", "--output_path"}


def _rewrite_quant_command(command: str, *, model: str,
                           output_name: str | None) -> str:
    """Rewrite the executed command into the INC template (no local paths)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--device_map":
            i += 2
            continue
        if token.startswith("--device_map="):
            i += 1
            continue
        if token in _PATH_FLAGS:
            # Drop the flag and its local-path value; the evaluation output path
            # is an aqa-run detail, not part of a reproducible recipe.
            i += 2
            continue
        if token.startswith(tuple(f + "=" for f in _PATH_FLAGS)):
            i += 1
            continue
        if token == "--model_name":
            out.extend(["--model_name", "${MODEL}"])
            i += 2
            continue
        if token in {"--model", "--model_path"}:
            out.append(f"{token}=${{MODEL}}")
            i += 2
            continue
        if token.startswith(("--model_name=", "--model=", "--model_path=")):
            out.append(token.split("=", 1)[0] + "=${MODEL}")
            i += 1
            continue
        if token == "--output_dir":
            out.extend(["--output_dir", "${OUTPUT_DIR}"])
            i += 2
            continue
        if token.startswith("--output_dir="):
            out.append("--output_dir=${OUTPUT_DIR}")
            i += 1
            continue
        if output_name and token.rstrip("/").endswith("/" + output_name):
            out.append("${OUTPUT_DIR}")
            i += 1
            continue
        if token == model or token.rstrip("/") == model.rstrip("/"):
            out.append("${MODEL}")
            i += 1
            continue
        if token.startswith("/") or token.startswith("~/"):
            i += 1
            continue
        out.append(token)
        i += 1
    return _format_script(out)


def _rewrite_eval_script(commands: list[str], *, model_paths: list[str],
                         variable: str) -> str:
    rendered: list[str] = []
    for command in commands:
        text = command
        for path in sorted((p for p in model_paths if p), key=len, reverse=True):
            text = text.replace(path, "${" + variable + "}")
        try:
            tokens = shlex.split(text)
        except ValueError:
            rendered.append(text)
            continue
        out: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"--output_path", "--log_samples"}:
                i += 2 if token == "--output_path" else 1
                continue
            if token.startswith("--output_path="):
                i += 1
                continue
            # lm_eval's --model_args is a comma-separated key=value string.  A
            # run may be replayed after the checkpoint moved, so rewrite the
            # `pretrained`/`tokenizer` values regardless of the recorded path.
            if "," in token or token.startswith(("pretrained=", "tokenizer=")):
                parts = token.split(",")
                parts = [
                    ("pretrained=${" + variable + "}"
                     if part.startswith("pretrained=") else
                     "tokenizer=${" + variable + "}"
                     if part.startswith("tokenizer=") else part)
                    for part in parts
                ]
                token = ",".join(parts)
            out.append(token)
            i += 1
        rendered.append(_format_script(out))
    return "\n".join(rendered)


def _eval_block(commands: list[str], *, model_paths: list[str], variable: str,
                variables: dict[str, Any],
                environment: str = "evaluation") -> dict[str, Any] | None:
    """INC ``evaluation`` command block (``None`` when it would be incomplete)."""
    if not commands:
        return None
    script = _rewrite_eval_script(commands, model_paths=model_paths,
                                  variable=variable)
    used = set(_VAR_RE.findall(script))
    return {
        "environment": environment,
        "shell": "bash",
        "variables": {k: v for k, v in variables.items() if k in used},
        "script": script,
    }


def _result_table(family: str, variant_id: str, result: "EvalResult",
                  baseline: "EvalResult | None",
                  smoke_limit: int | None = None,
                  baseline_label: str = "BF16") -> dict[str, Any] | None:
    """One accuracy table: metric columns + reported average/ratio/bits."""
    benches = result.benchmarks
    if not benches:
        return None
    columns: list[dict[str, Any]] = []
    col_ids: dict[str, str] = {}
    used: set[str] = set()
    for name, _b in benches.items():
        cid = _slug(name)
        while cid in used:
            cid += "-x"
        used.add(cid)
        col_ids[name] = cid
        columns.append({"id": cid, "label": name, "kind": "metric",
                        "task": name, "unit": "fraction"})
    columns.append({"id": "average", "label": "Average (reported)",
                    "kind": "reported_summary", "unit": "fraction"})
    columns.append({"id": "ratio", "label": "Ratio (reported)",
                    "kind": "reported_summary", "unit": "percent"})

    rows: list[dict[str, Any]] = []
    base_mean = baseline.mean_score() if baseline else None
    if baseline and baseline.benchmarks:
        bvals = {col_ids[n]: round(baseline.benchmarks[n].score, 6)
                 for n in benches if n in baseline.benchmarks}
        bvals["average"] = round(baseline.mean_score(), 6)
        bvals["ratio"] = None
        rows.append({"id": f"bf16-{family}", "label": baseline_label,
                     "kind": "baseline", "baseline": "bf16", "values": bvals})
    vmean = result.mean_score()
    vvals = {col_ids[n]: round(b.score, 6) for n, b in benches.items()}
    vvals["average"] = round(vmean, 6)
    vvals["ratio"] = (round(vmean / base_mean * 100, 2)
                      if base_mean else None)
    rows.append({"id": f"{variant_id}-result", "label": variant_id.upper(),
                 "kind": "variant", "variant": variant_id, "values": vvals})
    notes = ("Reported numbers from this aqa run; rows carry no input "
             "fingerprint. Bits exclude scale bits.")
    if smoke_limit:
        notes += (f" Smoke evaluation: --limit {smoke_limit} per task; "
                  "these are not full-set results.")
    return {
        "id": f"{family}-accuracy", "title": "Accuracy",
        "notes": notes,
        "columns": columns, "rows": rows,
    }


def _hf_url(run_cfg: "RunConfig") -> str | None:
    up = getattr(run_cfg.provider, "upload", None)
    if up is None or not up.repo:
        return None
    name = up.repo_name or canonical_artifact_name(run_cfg)
    return f"https://huggingface.co/{up.repo}/{name}"


# ───────────────────────── public API ─────────────────────────


def build_recipe_manifest(
    *,
    run_cfg: "RunConfig",
    result: "EvalResult",
    run_id: str,
    out_dir: str | Path,
    quant_dir: str | Path | None = None,
    baseline: "EvalResult | None" = None,
    gate_verdict: str | None = None,
    gate_reasons: list[str] | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project a quantize run onto the INC recipe manifest (+ aqa extensions).

    Returns ``None`` for non-quantize runs (there is no recipe to emit).
    """
    quant = run_cfg.quant
    if quant is None:
        return None
    out_dir = Path(out_dir)

    env_lock = _read_json(out_dir / "env_lock.json")
    quant_plan = _read_json(out_dir / "quant_plan.json")
    post_export = _read_json(out_dir / ".post_export.json")
    repair_kind = (post_export.get("kind")
                   or getattr(quant, "export_repair", None))

    # Model identity: a local checkpoint path is not a publishable model.source,
    # but it is the only identity we have; record the gap in `issues`.
    source = _model_source(run_cfg)
    model_name = _model_basename(source)
    scheme = quant.scheme
    rtn = quant.method == "model_free" or (quant.iters or 0) == 0
    algorithm = "RTN" if rtn else "AutoRound"
    family = _slug(_lowest_bit_scheme(quant))
    if quant.layer_config:
        family += "-mixed"
    variant_id = f"{family}-v1"

    # Optional KV/attention static quant, parsed from the frozen command.
    command = str(quant_plan.get("command") or "")
    if not command:
        command = str(_read_json(out_dir / ".quant_command.json").get("command") or "")
    kv_scheme = "FP8" if "--static_kv_dtype" in command else None
    attn_scheme = "FP8" if "--static_attention_dtype" in command else None

    # The per-module coverage map (reads the exported quant config when present).
    if model_cfg is None:
        from .phase_artifacts import _read_model_config
        model_cfg = _read_model_config(str(quant_dir) if quant_dir else run_cfg.model)
    details = compute_quantization_details(
        model_cfg=model_cfg or {}, declared_scheme=scheme, quant_dir=quant_dir,
        model_source=run_cfg.model,
        declared_ignore=quant.ignore_layers, quant_lm_head=quant.quant_lm_head,
        layer_config=quant.layer_config,
        kv_scheme=kv_scheme, attn_scheme=attn_scheme)
    source_quantized = bool((model_cfg or {}).get("quantization_config"))
    baseline_label = "Official checkpoint" if source_quantized else "BF16"

    operations, op_order, op_labels = _operations(
        scheme, details, kv_scheme, attn_scheme)

    # Command blocks: template variables instead of local paths.
    output_name = (run_cfg.provider.upload.repo_name
                   if run_cfg.provider.upload is not None
                   and run_cfg.provider.upload.repo_name
                   else canonical_artifact_name(run_cfg))
    quantization_cmd = None
    if command:
        quant_script = _rewrite_quant_command(
            command, model=run_cfg.model, output_name=output_name)
        if repair_kind:
            quant_script = (quant_script.rstrip() + "\n"
                            f"python -m aqa.quantize.export_repair "
                            f"\"${{OUTPUT_DIR}}\" --kind {shlex.quote(str(repair_kind))}")
        quant_vars = {
            "MODEL": {"kind": "input_model", "from": "model.source",
                      "description": "Source model ID or local copy"},
            "OUTPUT_DIR": {"kind": "output_dir", "default": output_name,
                           "description": "Quantized model output directory"},
        }
        used = set(_VAR_RE.findall(quant_script))
        quantization_cmd = {
            "environment": "quantization",
            "shell": "bash",
            "variables": {k: v for k, v in quant_vars.items() if k in used},
            "script": quant_script,
        }

    quant_paths = [str(quant_dir)] if quant_dir else []
    if quant_dir:
        quant_paths.append(str(Path(quant_dir)))
    evaluation_cmd = _eval_block(
        list(result.commands), model_paths=quant_paths, variable="MODEL_PATH",
        variables={"MODEL_PATH": {
            "kind": "checkpoint", "from": "quantization.output_dir",
            "default": output_name,
            "description": "Quantized checkpoint to evaluate"}})

    # Honest status + issues (INC: an incomplete variant must list issues).
    issues: list[str] = []
    if quantization_cmd is None:
        issues.append("Quantize command was not recorded for this run.")
    if evaluation_cmd is None:
        issues.append("Evaluation command was not recorded for this run.")
    if not _is_repo_id(source):
        issues.append("model.source is a local path; replace it with the "
                      "Hugging Face repo id before upstreaming.")
    if run_cfg.eval.limit:
        issues.append(
            f"Evaluation was smoke-limited to {run_cfg.eval.limit} samples/task; "
            "result_tables are smoke, not full-set results.")
    if result.benchmarks:
        issues.append("Results are reported numbers (no input fingerprint recorded).")
    if post_export:
        issues.append(
            "Post-export repair applied: "
            f"{post_export.get('kind', repair_kind)} rewrote the quantization "
            "config for the runtime namespace; the raw auto-round config is "
            f"backed up at {post_export.get('backup', 'the export backup')}.")
    try:
        actions = load_owner_actions(out_dir) or []
    except Exception:  # noqa: BLE001 - malformed owner log must not break recipe
        actions = []
    for action in actions:
        if action.get("kind") in {"patch", "command", "note"} and action.get("text"):
            issues.append(
                f"Owner {action['kind']} during the run: {action['text']}")
    if gate_verdict and gate_verdict != "GO":
        issues.append(f"Gate verdict was {gate_verdict}: " + "; ".join(gate_reasons or []))

    variant: dict[str, Any] = {
        "id": variant_id,
        "family": family,
        "label": "V1 — " + " + ".join(op_labels),
        "recipe": {"approach": "PTQ", "algorithm": algorithm,
                   "operations": op_order},
        "quantization": quantization_cmd,
        "evaluation": evaluation_cmd,
    }
    avg_bits = details.get("effective_bits")
    if isinstance(avg_bits, (int, float)):
        variant["avg_bits"] = avg_bits
    hf_url = _hf_url(run_cfg)
    if hf_url:
        variant["hf_url"] = hf_url
    if issues:
        variant["issues"] = issues

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "model": {"id": _slug(model_name), "name": model_name, "source": source},
        "environments": _environments(env_lock),
        "operations": operations,
        "variants": [variant],
    }

    # Optional baseline + result tables.
    if baseline is not None and baseline.commands and baseline.benchmarks:
        be = _eval_block(
            list(baseline.commands), model_paths=[run_cfg.model],
            variable="MODEL_PATH",
            variables={"MODEL_PATH": {
                "kind": "checkpoint", "default": source,
                "description": "Original model"}})
        if be is not None:
            manifest["baselines"] = [
                {"id": "bf16", "label": baseline_label, "evaluation": be}]
    table = _result_table(family, variant_id, result, baseline,
                          smoke_limit=run_cfg.eval.limit,
                          baseline_label=baseline_label)
    manifest["result_tables"] = [table] if table else []

    # aqa extensions (strippable; see to_inc_strict).
    manifest["quantization_details"] = {variant_id: details}
    manifest["provenance"] = {
        "tool": "aqa", "run_id": run_id, "created_at": now_iso(),
        "gate": None if gate_verdict is None else {
            "verdict": gate_verdict, "reasons": list(gate_reasons or [])},
        "note": ("quantization_details and provenance are aqa extensions; call "
                 "to_inc_strict() / drop them before upstreaming to INC-Recipes."),
    }
    return manifest


def to_inc_strict(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with aqa extensions removed (validates against INC schema)."""
    core = copy.deepcopy(manifest)
    core.pop("quantization_details", None)
    core.pop("provenance", None)
    return core


def inc_schema_errors(manifest: dict[str, Any],
                      schema_path: str | Path) -> list[str]:
    """Validate ``manifest`` against the INC JSON Schema; ``[]`` means valid.

    ``jsonschema`` is an optional dependency: if it is not installed this
    returns a single explanatory error instead of raising.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - optional dependency
        return ["jsonschema is not installed; cannot validate against INC schema"]
    schema = json.loads(Path(schema_path).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in error.path)}: {error.message}"
        for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    ]


class _LiteralDumper(yaml.SafeDumper):
    """YAML dumper that renders multi-line strings (scripts) as ``|`` blocks."""


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> Any:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_presenter)


def _dump_manifest(manifest: dict[str, Any]) -> str:
    return yaml.dump(manifest, Dumper=_LiteralDumper, sort_keys=False,
                     default_flow_style=False, allow_unicode=True, width=100)


def strict_recipe_name(manifest: dict[str, Any]) -> str | None:
    """Filename for the strict INC projection (``<model.id>.yaml``)."""
    model_id = str((manifest.get("model") or {}).get("id") or "").strip()
    if not model_id:
        return None
    safe = re.sub(r"[^a-z0-9-]+", "-", model_id.lower()).strip("-")
    if not safe:
        return None
    # Never overwrite the aqa-extended recipe.yaml.
    return f"{safe}.yaml" if f"{safe}.yaml" != RECIPE_FILE else "recipe.inc.yaml"


def write_strict_recipe(out_dir: str | Path,
                        manifest: dict[str, Any]) -> Path | None:
    """Write the INC-strict projection as ``<out_dir>/<model.id>.yaml``.

    This is the file that can be copied verbatim to
    ``INC-Quantization-Recipes/_data/recipes/<model.id>.yaml`` and validated
    with ``bundle exec ruby tools/recipe_data.rb validate`` — it has no
    ``quantization_details`` / ``provenance`` extensions.
    """
    name = strict_recipe_name(manifest)
    if not name:
        return None
    try:
        path = Path(out_dir) / name
        path.write_text(_dump_manifest(to_inc_strict(manifest)), encoding="utf-8")
        return path
    except Exception as exc:  # noqa: BLE001 - artifact write must never abort a run
        _log.warning("strict recipe write failed: %s", exc)
        return None


def write_recipe_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> Path | None:
    """Serialise the aqa-extended ``recipe.yaml`` + the strict ``<model.id>.yaml``.

    The extended file carries ``quantization_details`` / ``provenance`` for aqa's
    own audit; the strict sidecar is the INC-schema-valid file for upstreaming.
    Both writes are best-effort: a failure logs and never aborts the run.
    """
    try:
        path = Path(out_dir) / RECIPE_FILE
        path.write_text(_dump_manifest(manifest), encoding="utf-8")
        write_strict_recipe(out_dir, manifest)
        return path
    except Exception as exc:  # noqa: BLE001 - artifact write must never abort a run
        _log.warning("recipe.yaml write failed: %s", exc)
        return None

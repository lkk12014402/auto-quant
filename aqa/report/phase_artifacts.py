"""P2 phase-gate artifacts: the machine-readable ``*.json`` siblings.

RFC 0001 §4 fixes a per-phase artifact contract.  Several phases historically
produced only prose (``analysis.md``) or left their intent implicit in the
``run.yaml``.  This module writes the deterministic, machine-readable siblings
so ``aqa inspect`` can validate each phase's product and downstream tooling can
read structured data instead of grepping logs:

* ``env_lock.json``    (ENV)     — frozen software/hardware stack.
* ``analysis.json``    (ANALYZE) — model identity + the planned quant scope + a
  *predicted* per-module disposition table (which family lands at which precision
  and the command token that set it), built by the same engine as the CLOSE-phase
  ``quantization_details`` so plan vs. actual diff cleanly.  Reads real weight
  headers when the model is on disk / in the HF cache, else analytical config
  dims; may fetch the tiny public ``config.json`` but never the weights.
* ``quant_plan.json``  (QUANT)   — the quant recipe + the exact frozen command.
* ``eval_report.json`` (EVAL)    — source vs quant scores + per-benchmark gap.

All writers are deterministic and best-effort (no GPU; ANALYZE may fetch a model's
public ``config.json``/headers but never weights): a failure is logged and never
aborts the run.  Writes are atomic (``os.replace``).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_util import get_logger
from ..session._io import atomic_write_json, now_iso
from ..session.state import SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig
    from ..eval.registry import EvalResult

_log = get_logger("report.phase_artifacts")

ENV_LOCK_FILE = "env_lock.json"
ANALYSIS_FILE = "analysis.json"
QUANT_PLAN_FILE = "quant_plan.json"
EVAL_REPORT_FILE = "eval_report.json"


def _read_model_config(model: str) -> dict[str, Any]:
    """Load a local ``config.json`` (best-effort, no network)."""
    try:
        p = Path(model) / "config.json"
        if p.is_file():
            return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _fetch_model_config(model: str) -> tuple[dict[str, Any], str]:
    """Resolve a model's ``config.json`` (identity only — never weights).

    Tries, in order: a local directory, the HF cache (``local_files_only``), then
    a networked fetch of *just* ``config.json`` from the hub.  ANALYZE is allowed
    to consult config/headers over the network (RFC 0002); this only ever pulls
    the tiny public ``config.json``, not the weights.  Returns ``(cfg, source)``.
    """
    cfg = _read_model_config(model)
    if cfg:
        return cfg, "config.json (local)"
    try:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(model, "config.json", local_files_only=True)
            source = "config.json (hf-cache)"
        except Exception:  # noqa: BLE001 - not cached; fall through to network
            path = hf_hub_download(model, "config.json")
            source = "config.json (hub)"
        return json.loads(Path(path).read_text()), source
    except Exception:  # noqa: BLE001 - offline / gated / hub unavailable
        return {}, "unavailable"


def _moe_topology(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Extract MoE topology from config (tolerating multimodal ``text_config``)."""
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(*keys: str) -> Any:
        for k in keys:
            v = cfg.get(k)
            if v is None:
                v = tc.get(k)
            if v is not None:
                return v
        return None

    ne = g("num_experts", "num_local_experts", "n_routed_experts")
    if not ne:
        return None
    return {
        "num_experts": ne,
        "experts_per_tok": g("num_experts_per_tok", "moe_topk"),
        "moe_intermediate_size": g("moe_intermediate_size"),
        "shared_expert": bool(g("shared_expert_intermediate_size", "n_shared_experts")),
    }


def _identity(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    """Model identity block from config (arch, type, layers, modality, MoE)."""
    multimodal = bool(cfg.get("vision_config") or cfg.get("image_token_id")
                      or cfg.get("vision_start_token_id"))
    return {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": _num_layers(cfg),
        "multimodal": multimodal,
        "moe": _moe_topology(cfg),
        "source": source,
    }


def _num_layers(cfg: dict[str, Any]) -> int | None:
    """Layer count, tolerating multimodal nesting (``text_config``)."""
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        if isinstance(cfg.get(key), int):
            return cfg[key]
    text = cfg.get("text_config")
    if isinstance(text, dict):
        return _num_layers(text)
    return None


def write_env_lock(out_dir: Path, run_cfg: "RunConfig", run_id: str) -> Path | None:
    """ENV phase: freeze the software/hardware stack (reuses manifest probes)."""
    import platform

    from .manifest import _accelerator, _package_versions

    try:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now_iso(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "accelerator": _accelerator(run_cfg),
        }
        path = Path(out_dir) / ENV_LOCK_FILE
        atomic_write_json(path, payload)
        return path
    except Exception as exc:  # noqa: BLE001 - artifact write must never abort a run
        _log.warning("env_lock.json write failed: %s", exc)
        return None


def write_analysis(out_dir: Path, run_cfg: "RunConfig", run_id: str) -> Path | None:
    """ANALYZE phase: model identity + the *predicted* per-module quant plan.

    The headline product is ``disposition`` — a per-module-family table that joins
    the model's real structure with the declared command, resolving each family to
    its applied precision and naming the command token that set it (global
    ``--scheme``, a ``--layer_config`` override such as ``mlp.experts`` → MXFP4, an
    ``--ignore_layers`` match → BF16, or a precision-critical keep).  It is built
    by the *same* function as the CLOSE-phase ``quantization_details``, so the
    predicted plan and the actual exported coverage share a schema and diff
    cleanly.  Everything is deterministic/offline except a best-effort fetch of
    the tiny public ``config.json`` + safetensors headers (never the weights).
    """
    try:
        cfg, cfg_source = _fetch_model_config(run_cfg.model)
        quant = run_cfg.provider.quant if run_cfg.provider.type == "quantize" else None
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now_iso(),
            "model": run_cfg.model,
            "identity": _identity(cfg, cfg_source),
        }

        if quant is None:
            payload["quant_scope"] = None
            payload["note"] = "pretrained/baseline run — no quantization planned"
            path = Path(out_dir) / ANALYSIS_FILE
            atomic_write_json(path, payload)
            return path

        payload["quant_scope"] = {
            "scheme": quant.scheme,
            "method": quant.method,
            "algorithm": quant.algorithm,
            "iters": quant.iters,
            "nsamples": quant.nsamples,
            "export_format": quant.export_format,
            "ignore_layers": quant.ignore_layers,
            "layer_config": quant.layer_config,
            "quant_lm_head": quant.quant_lm_head,
            "static_kv_fp8": quant.static_kv_fp8,
            "static_attention_fp8": quant.static_attention_fp8,
        }

        # The literal auto-round command this plan will run (built by the real
        # builder, so analysis.json and quant_plan.json agree verbatim).
        try:
            from ..quantize import build_autoround_command
            cmd = build_autoround_command(
                run_cfg.model, quant, Path(out_dir) / "quantized", num_gpus=1)
            payload["planned_command"] = " ".join(shlex.quote(c) for c in cmd)
        except Exception as exc:  # noqa: BLE001 - command preview is best-effort
            _log.debug("planned_command preview failed: %s", exc)

        # The predicted per-module disposition — same builder as CLOSE, with no
        # export yet, so it reflects the declared command. Reads real weight
        # headers when the model is on disk / in the HF cache, else analytical
        # dims from config.json.
        from .names import scheme_display
        from .quant_details import compute_quantization_details
        disposition = compute_quantization_details(
            model_cfg=cfg, declared_scheme=quant.scheme, quant_dir=None,
            model_source=run_cfg.model, declared_ignore=quant.ignore_layers,
            quant_lm_head=quant.quant_lm_head, layer_config=quant.layer_config,
            kv_scheme="FP8" if quant.static_kv_fp8 else None,
            attn_scheme="FP8" if quant.static_attention_fp8 else None)
        disposition["stage"] = "predicted"
        payload["scheme_label"] = scheme_display(quant)
        payload["disposition"] = disposition

        notes = [
            "disposition is PREDICTED from the declared command; CLOSE emits the "
            "ACTUAL quantization_details from the exported checkpoint (same schema).",
            "each module_groups[].matched_by names the command token that set the "
            "precision (scheme / layer_config:<key> / ignore:<token> / keep).",
        ]
        if disposition.get("derived_from") != "weight_shapes":
            notes.append("params are analytical (config dims): weights not local "
                         "yet — regenerated exactly once the model is cached.")
        payload["notes"] = notes

        path = Path(out_dir) / ANALYSIS_FILE
        atomic_write_json(path, payload)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("analysis.json write failed: %s", exc)
        return None


def write_quant_plan(out_dir: Path, run_cfg: "RunConfig", run_id: str) -> Path | None:
    """QUANT phase: the recipe fields + the actual frozen command (if recorded)."""
    quant = run_cfg.provider.quant if run_cfg.provider.type == "quantize" else None
    if quant is None:
        return None
    try:
        command, command_source = None, None
        cmd_path = Path(out_dir) / ".quant_command.json"
        if cmd_path.is_file():
            rec = json.loads(cmd_path.read_text())
            command = rec.get("command")
            command_source = rec.get("source")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now_iso(),
            "scheme": quant.scheme,
            "method": quant.method,
            "export_format": quant.export_format,
            "algorithm": quant.algorithm,
            "iters": quant.iters,
            "nsamples": quant.nsamples,
            "ignore_layers": quant.ignore_layers,
            "quant_lm_head": quant.quant_lm_head,
            "device_map": quant.device_map,
            "command": command,
            "command_source": command_source,
        }
        path = Path(out_dir) / QUANT_PLAN_FILE
        atomic_write_json(path, payload)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("quant_plan.json write failed: %s", exc)
        return None


def _scores(result: "EvalResult | None") -> dict[str, float]:
    if result is None:
        return {}
    return {n: b.score for n, b in result.benchmarks.items()}


def write_eval_report(out_dir: Path, run_cfg: "RunConfig", run_id: str,
                      result: "EvalResult", baseline: "EvalResult | None",
                      gate_verdict: str | None,
                      gate_reasons: list[str] | None = None) -> Path | None:
    """EVAL phase: synthesize source(bf16) vs quant scores + per-benchmark gap."""
    try:
        quant_scores = _scores(result)
        source_scores = _scores(baseline)
        gap: dict[str, Any] | None = None
        if baseline is not None:
            per = {n: round(quant_scores[n] - source_scores[n], 6)
                   for n in quant_scores if n in source_scores}
            gap = {
                "mean_score": round(result.mean_score() - baseline.mean_score(), 6),
                "per_benchmark": per,
            }
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now_iso(),
            "model": run_cfg.model,
            "backend": run_cfg.eval.backend,
            "limit": run_cfg.eval.limit,
            "quant": {"mean_score": result.mean_score(),
                      "results": result.to_dict().get("results", {})},
            "source": None if baseline is None else {
                "mean_score": baseline.mean_score(),
                "results": baseline.to_dict().get("results", {})},
            "gap": gap,
            "gate": None if gate_verdict is None else {
                "verdict": gate_verdict, "reasons": list(gate_reasons or [])},
        }
        path = Path(out_dir) / EVAL_REPORT_FILE
        atomic_write_json(path, payload)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("eval_report.json write failed: %s", exc)
        return None

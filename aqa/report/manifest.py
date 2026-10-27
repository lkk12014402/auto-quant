"""Reproducibility manifest: freeze *how* an eval run was produced.

A :class:`RunManifest` is the NeMo-Evaluator-style record that makes a score
reproducible: the model + quant fingerprint, the exact eval configuration
(benchmarks, harness, sampling params, seed, few-shot), an environment
fingerprint (key dependency versions + accelerator), the results (with clean
denominators), and a re-runnable command.  It is written next to the accuracy
json and rides the results dataset, so anyone can diff two manifests to see
*why* two W4A16 numbers differ (seed / vllm version / few-shot) instead of
guessing.

The module is pure + dependency-light (only stdlib + the run config/result it is
handed) so it is unit-tested without a real run.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .owner_actions import load_owner_actions

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig
    from ..eval.registry import EvalResult

SCHEMA_VERSION = "1.0"

# Dependency versions worth pinning for reproducibility (quant + eval stack).
_TRACKED_PACKAGES = (
    "autoquant-agent", "torch", "transformers", "auto-round", "auto_round",
    "lm-eval", "lm_eval", "vllm", "accelerate", "datasets", "huggingface-hub",
)


def _package_versions() -> dict[str, str]:
    """Best-effort ``{package: version}`` for the tracked stack (installed only)."""
    seen: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            seen[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return seen


def _accelerator(run_cfg: RunConfig) -> dict[str, Any]:
    ssh = run_cfg.executor.ssh
    if ssh is not None:
        gpus = [g for g in ssh.gpus.split(",") if g.strip()]
        return {"family": ssh.accelerator, "num_devices": len(gpus) or 1,
                "executor": run_cfg.executor.type, "remote": True}
    # Local executor: probe the accelerators actually visible to this process
    # (honours CUDA_VISIBLE_DEVICES) so the frozen env records the real
    # hardware instead of unconditionally claiming cpu/local.
    family, num_devices, device_name = _local_accelerator()
    acc: dict[str, Any] = {"family": family, "num_devices": num_devices,
                           "executor": run_cfg.executor.type, "remote": False}
    if device_name:
        acc["device_name"] = device_name
    return acc


def _local_accelerator() -> tuple[str, int, str]:
    """Best-effort probe of locally visible accelerators.

    Returns a ``(family, num_devices, device_name)`` tuple. ``family`` is a
    normalised token (``cuda``/``xpu``/``cpu/local``) so downstream consumers
    stay stable, while ``device_name`` carries the human-readable detail.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in practice
        return "cpu/local", 1, ""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover - driver edge cases
                name = ""
            return "cuda", torch.cuda.device_count(), name
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available() and xpu.device_count() > 0:
            try:
                name = torch.xpu.get_device_name(0)
            except Exception:  # pragma: no cover
                name = ""
            return "xpu", xpu.device_count(), name
    except Exception:  # pragma: no cover - any probe failure => cpu fallback
        pass
    return "cpu/local", 1, ""


def _reproduce_command(config_path: str | None) -> str:
    if config_path:
        return f"aqa run {config_path}"
    return "aqa run <config.yaml>"


def record_quant_command(out_dir: Path, command: str, *, source: str) -> None:
    """Persist the *actual* quantize command so the manifest can capture it.

    Written by the pipeline where the command is known (builder output or an
    agent-optimized recipe), then read back in :func:`build_manifest` -- the
    quantize phase runs behind the provider boundary, so a small file-drop is
    the least-coupled way to surface the executed command to the manifest.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _QUANT_COMMAND_FILE).write_text(
            json.dumps({"command": command, "source": source}, ensure_ascii=False),
            encoding="utf-8")
    except OSError:  # pragma: no cover - best-effort provenance
        pass


def _load_quant_command(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / _QUANT_COMMAND_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        return None


_QUANT_COMMAND_FILE = ".quant_command.json"
_POST_EXPORT_FILE = ".post_export.json"


def record_post_export(out_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a deterministic post-export repair record for the manifest/recipe."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _POST_EXPORT_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:  # pragma: no cover - best-effort provenance
        pass


def _load_post_export(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / _POST_EXPORT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_knowledge_sync(out_dir: Path) -> dict[str, Any] | None:
    path = Path(out_dir) / "knowledge_sync.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class RunManifest:
    """Everything needed to reproduce (and audit) one eval run's numbers."""

    schema_version: str
    created_at: str
    run_id: str
    model: dict[str, Any]
    provider: dict[str, Any] | None
    quant: dict[str, Any] | None
    eval: dict[str, Any]
    environment: dict[str, Any]
    results: dict[str, Any]
    gate: dict[str, Any] | None = None
    commands: dict[str, Any] = field(default_factory=dict)
    reproduce: dict[str, Any] = field(default_factory=dict)
    # RFC 0002 §5: present only for an ``aqa reproduce`` run (manifest + variant
    # + mode + tolerances); the layout uses it to require reproduction_report.json.
    reproduction: dict[str, Any] | None = None
    # Owner actions recorded during the run (``aqa log-action`` / owner_actions.jsonl).
    owner_actions: list[dict[str, Any]] | None = None
    # Local-first knowledge sync result (``knowledge_sync.json``), when enabled.
    knowledge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(
    run_cfg: RunConfig,
    result: EvalResult,
    *,
    run_id: str,
    config_path: str | None = None,
    baseline: EvalResult | None = None,
    gate_verdict: str | None = None,
    extra_env: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    usage: Any = None,
    self_heal: dict[str, Any] | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from the run config + eval result.

    ``usage`` (an ``AgentEvalResult``, optional) contributes the agent
    efficiency block -- token + turn totals -- for agentic benchmarks; when
    omitted, efficiency is derived from any per-benchmark provenance the
    ``EvalResult`` itself carries.  ``self_heal`` records the agent effort the
    pipeline spent *producing* this run (fix-loop attempts + tokens), so the cost
    of reaching a green run is auditable next to the score.
    """
    quant = run_cfg.quant.model_dump() if run_cfg.quant is not None else None

    lm = run_cfg.eval.lm_eval
    eval_block: dict[str, Any] = {
        "benchmarks": list(run_cfg.eval.benchmarks),
        "backend": run_cfg.eval.backend,
        "harness": run_cfg.eval.harness,
        "batch_size": run_cfg.eval.batch_size,
        "limit": run_cfg.eval.limit,
        "seed": lm.seed,
        "gen_max_toks": run_cfg.eval.gen_max_toks,
        "generative_tasks": list(run_cfg.eval.generative_tasks),
        "apply_chat_template": lm.apply_chat_template,
        "fewshot_as_multiturn": lm.fewshot_as_multiturn,
        "model_args": dict(lm.model_args),
        "gen_kwargs": dict(lm.gen_kwargs),
        "baseline": run_cfg.eval.baseline,
    }

    results_block: dict[str, Any] = {
        "mean_score": result.mean_score(),
        "backend": result.backend,
        "benchmarks": {
            name: {"score": b.score, "metric": b.metric, "num_samples": b.num_samples}
            for name, b in result.benchmarks.items()
        },
    }
    if baseline is not None:
        results_block["baseline"] = {
            "mean_score": baseline.mean_score(),
            "benchmarks": {
                name: {"score": b.score, "metric": b.metric}
                for name, b in baseline.benchmarks.items()
            },
        }

    efficiency = _efficiency_block(result, usage)
    if efficiency is not None:
        results_block["efficiency"] = efficiency
    if self_heal:
        results_block["self_heal"] = self_heal

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": _package_versions(),
        "accelerator": _accelerator(run_cfg),
    }
    if extra_env:
        environment.update(extra_env)

    gate = {"verdict": gate_verdict} if gate_verdict is not None else None

    commands = _commands_block(result, out_dir)
    upload = run_cfg.provider.upload
    provider_block = {
        "type": run_cfg.provider.type,
        "upload": None if upload is None else {
            "repo": upload.repo,
            "repo_name": upload.repo_name,
            "private": upload.private,
        },
    }

    return RunManifest(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_id=run_id,
        model={"id": run_cfg.model, "provider": run_cfg.provider.type},
        provider=provider_block,
        quant=quant,
        eval=eval_block,
        environment=environment,
        results=results_block,
        gate=gate,
        commands=commands,
        reproduce={"command": _reproduce_command(config_path), "config": config_path},
        reproduction=(run_cfg.reproduce.model_dump()
                      if run_cfg.reproduce is not None else None),
        owner_actions=(load_owner_actions(out_dir) if out_dir is not None else None),
        knowledge=(_load_knowledge_sync(out_dir) if out_dir is not None else None),
    )


def _efficiency_block(result: EvalResult, usage: Any) -> dict[str, Any] | None:
    """Token + turn efficiency for the manifest (agent cost alongside the score).

    Prefers an explicit ``AgentEvalResult`` (``usage``); otherwise falls back to
    provenance the ``EvalResult`` carries on its agentic benchmarks. Returns
    ``None`` for a pure lm_eval run so the block never appears empty.
    """
    if usage is not None and hasattr(usage, "efficiency"):
        eff = usage.efficiency()
        return eff or None
    return result.efficiency()


def _commands_block(result: EvalResult, out_dir: Path | None) -> dict[str, Any]:
    """The *actual* executed commands (quant recipe + eval calls + self-heal).

    Unlike the ``quant``/``eval`` config blocks (which record *intent*), this
    captures what really ran -- including an agent-optimized quant command that
    replaced the builder, and any deterministic self-heal mutations applied to
    the eval invocation. That is what makes a score reproducible byte-for-byte.
    """
    block: dict[str, Any] = {}
    if out_dir is not None:
        quant_cmd = _load_quant_command(Path(out_dir))
        if quant_cmd and quant_cmd.get("command"):
            block["quantize"] = quant_cmd["command"]
            block["quantize_source"] = quant_cmd.get("source", "builder")
        post = _load_post_export(Path(out_dir))
        if post and post.get("command"):
            block["post_export"] = post["command"]
            block["post_export_repair"] = post
    if result.commands:
        block["evaluate"] = list(result.commands)
    if result.healed:
        block["healed"] = list(result.healed)
    return block


def write_manifest(manifest: RunManifest, out_dir: Path, *,
                   name: str = "run_manifest.json") -> Path:
    """Write the manifest as pretty JSON into ``out_dir``; return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path

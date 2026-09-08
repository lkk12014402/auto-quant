"""Thinking-mode detection and dual-mode evaluation helpers.

Reasoning models (qwen3, glm, deepseek-r1, ...) produce materially different
scores with chain-of-thought enabled vs disabled, so a run may need both
settings measured. The deterministic signal is the model's chat template:
thinking-capable families ship a template with an ``enable_thinking`` switch
(qwen3, glm-4.x/5, ...). Detection reads that template from a local checkpoint
dir, or from the HF hub when the model is given as a repo id (best effort:
hub lookup failures degrade to ``False`` and a warning, never crash a run).

The operator always wins: ``eval.thinking: both/on/off`` skips detection
entirely; ``auto`` uses it only when no explicit mode was requested.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import EvalConfig
from ..logging_util import get_logger

_log = get_logger("eval.thinking")

_TEMPLATE_FILES = ("chat_template.jinja", "tokenizer_config.json")
_HUB_FETCH_FILES = ("chat_template.jinja", "tokenizer_config.json")


def _template_text_from_dir(model_dir: Path) -> str:
    parts: list[str] = []
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        parts.append(jinja.read_text(errors="replace"))
    tc = model_dir / "tokenizer_config.json"
    if tc.is_file():
        try:
            data = json.loads(tc.read_text(errors="replace"))
        except json.JSONDecodeError:
            data = {}
        tpl = data.get("chat_template")
        if isinstance(tpl, str):
            parts.append(tpl)
        elif isinstance(tpl, list):  # newer multi-template form
            parts.extend(t.get("template", "") for t in tpl if isinstance(t, dict))
    return "\n".join(parts)


def _template_text_from_hub(model_id: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return ""
    parts: list[str] = []
    for name in _HUB_FETCH_FILES:
        try:
            p = hf_hub_download(model_id, name)
        except Exception:  # noqa: BLE001 - file may not exist; offline is fine
            continue
        try:
            text = Path(p).read_text(errors="replace")
        except OSError:
            continue
        if name.endswith(".json"):
            try:
                tpl = json.loads(text).get("chat_template")
            except json.JSONDecodeError:
                tpl = None
            if isinstance(tpl, str):
                parts.append(tpl)
            elif isinstance(tpl, list):
                parts.extend(t.get("template", "") for t in tpl if isinstance(t, dict))
        else:
            parts.append(text)
    return "\n".join(parts)


def detect_thinking_capable(model: str) -> bool:
    """True when the model's chat template exposes an ``enable_thinking`` switch."""
    model_dir = Path(model)
    if model_dir.is_dir():
        text = _template_text_from_dir(model_dir)
    else:
        try:
            text = _template_text_from_hub(model)
        except Exception as exc:  # noqa: BLE001 - detection must never crash a run
            _log.warning("thinking detection: hub lookup failed for %s (%s); assuming off", model, exc)
            text = ""
    capable = "enable_thinking" in text
    _log.info("thinking detection for %s: %s", model, "thinking-capable" if capable else "no switch found")
    return capable


def resolve_thinking_modes(cfg: EvalConfig, model: str) -> list[str]:
    """Which modes to evaluate, in run order (``on`` first when dual).

    ``auto`` runs both modes for thinking-capable models and a single default
    run otherwise; ``both``/``on``/``off`` are operator overrides.
    """
    if cfg.thinking == "both":
        return ["on", "off"]
    if cfg.thinking == "on":
        return ["on"]
    if cfg.thinking == "off":
        return ["off"]
    return ["on", "off"] if detect_thinking_capable(model) else ["on"]


def apply_mode_to_cfg(cfg: EvalConfig, mode: str) -> EvalConfig:
    """Return a copy of ``cfg`` adjusted for one thinking mode.

    ``on`` is the backend default (chat template decides, usually thinking);
    ``off`` merges ``cfg.thinking_off_model_args`` into the lm_eval model_args
    and drops a matching hint into the evalscope passthrough dict so that
    harness speaks the same mode.
    """
    if mode == "on":
        return cfg
    # user's explicit model_args win over the off-mode defaults on conflicts
    merged = {**cfg.thinking_off_model_args, **cfg.lm_eval.model_args}
    lm_opts = cfg.lm_eval.model_copy(update={"model_args": merged})
    scope = dict(cfg.evalscope)
    scope["chat_template_kwargs"] = {
        **cfg.thinking_off_model_args, **scope.get("chat_template_kwargs", {})}
    return cfg.model_copy(update={"lm_eval": lm_opts, "evalscope": scope})

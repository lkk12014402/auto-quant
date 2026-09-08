"""Tests for thinking-mode detection and dual-mode evaluation plumbing."""

from __future__ import annotations

import json
from pathlib import Path

from aqa.config import EvalConfig
from aqa.eval.thinking import (
    apply_mode_to_cfg,
    detect_thinking_capable,
    resolve_thinking_modes,
)


def _mk_model(tmp_path: Path, *, jinja: str | None = None,
              tokenizer_template: str | None = None) -> str:
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    if jinja is not None:
        (d / "chat_template.jinja").write_text(jinja)
    if tokenizer_template is not None:
        (d / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": tokenizer_template}))
    return str(d)


def test_detect_thinking_from_jinja(tmp_path):
    model = _mk_model(tmp_path, jinja="{% if enable_thinking %}think{% endif %}")
    assert detect_thinking_capable(model) is True


def test_detect_thinking_from_tokenizer_config(tmp_path):
    model = _mk_model(tmp_path, tokenizer_template="{{ x }} enable_thinking {{ y }}")
    assert detect_thinking_capable(model) is True


def test_detect_non_thinking_model(tmp_path):
    model = _mk_model(tmp_path, jinja="{{ messages }} plain template")
    assert detect_thinking_capable(model) is False
    missing = _mk_model(tmp_path)
    assert detect_thinking_capable(missing) is False


def test_resolve_modes_auto_dual_vs_single(tmp_path):
    cfg = EvalConfig()
    thinking = _mk_model(tmp_path / "a", jinja="enable_thinking switch")
    plain = _mk_model(tmp_path / "b", jinja="plain")
    assert resolve_thinking_modes(cfg, thinking) == ["on", "off"]
    assert resolve_thinking_modes(cfg, plain) == ["on"]


def test_resolve_modes_operator_overrides(tmp_path):
    plain = _mk_model(tmp_path, jinja="plain")
    assert resolve_thinking_modes(EvalConfig(thinking="both"), plain) == ["on", "off"]
    assert resolve_thinking_modes(EvalConfig(thinking="on"), plain) == ["on"]
    assert resolve_thinking_modes(EvalConfig(thinking="off"), plain) == ["off"]


def test_apply_mode_off_merges_model_args_user_wins():
    cfg = EvalConfig()
    off = apply_mode_to_cfg(cfg, "off")
    assert off.lm_eval.model_args == {"enable_thinking": False}
    assert off.evalscope["chat_template_kwargs"] == {"enable_thinking": False}

    cfg2 = EvalConfig(lm_eval={"model_args": {"enable_thinking": True, "max_num_seqs": 8}})
    off2 = apply_mode_to_cfg(cfg2, "off")
    # explicit user model_args beat the off-mode default on conflicts
    assert off2.lm_eval.model_args["enable_thinking"] is True
    assert off2.lm_eval.model_args["max_num_seqs"] == 8


def test_apply_mode_on_is_identity():
    cfg = EvalConfig()
    assert apply_mode_to_cfg(cfg, "on") is cfg


def test_custom_off_switch():
    cfg = EvalConfig(thinking_off_model_args={"thinking": False})
    off = apply_mode_to_cfg(cfg, "off")
    assert off.lm_eval.model_args == {"thinking": False}

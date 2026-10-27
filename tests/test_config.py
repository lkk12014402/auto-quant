import os

import pytest

from aqa.config import RunConfig, load_run_config, parse_run_config
from aqa.logging_util import ConfigError

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_minimal_config():
    cfg = parse_run_config({"model": "org/model"})
    assert isinstance(cfg, RunConfig)
    assert cfg.eval.backend == "hf"
    assert cfg.quant is None  # eval-only by default


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("MY_SCHEME", "MXFP8")
    cfg = parse_run_config({"model": "org/m", "quant": {"scheme": "${MY_SCHEME}"}})
    assert cfg.quant.scheme == "MXFP8"


def test_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    cfg = parse_run_config({"model": "org/m", "output": {"dir": "${NOPE:-./out}"}})
    assert cfg.output.dir == "./out"


def test_env_unset_no_default_raises(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(ConfigError):
        parse_run_config({"model": "${NOPE}"})


def test_base_inheritance_and_examples():
    cfg = load_run_config(os.path.join(EXAMPLES, "run.yaml"))
    assert cfg.quant.scheme == "MXFP4"
    assert cfg.eval.baseline is True
    # run_ssh.yaml inherits run.yaml via _base_ and overrides executor
    os.environ.setdefault("GPU_SSH_PASS", "x")
    ssh_cfg = load_run_config(os.path.join(EXAMPLES, "run_ssh.yaml"))
    assert ssh_cfg.executor.type == "ssh"
    assert ssh_cfg.quant.scheme == "MXFP4"  # inherited from base
    assert ssh_cfg.executor.ssh.gpus == "0,1,2,3"


def test_invalid_model_rejected():
    with pytest.raises(ConfigError):
        parse_run_config({"model": "notarepo"})


def test_nested_lm_eval_model_args_rejected():
    with pytest.raises(ConfigError, match="nested dict/list"):
        parse_run_config({
            "model": "org/m",
            "eval": {"lm_eval": {"model_args": {
                "kernel_config": {"enable_flashinfer_autotune": False}}}},
        })


def test_export_repair_field_parses():
    cfg = parse_run_config({
        "model": "org/m",
        "quant": {"scheme": "MXFP4", "export_repair": "glm5_vllm"},
    })
    assert cfg.quant is not None and cfg.quant.export_repair == "glm5_vllm"


def test_knowledge_sync_fields_parse():
    cfg = parse_run_config({
        "model": "org/m",
        "agent": {"knowledge": {"sync": "background", "sediment_sessions": False,
                                "repo": "org/knowledge"},
                  "recipe_kb": {"auto_sync": "branch"}},
    })
    assert cfg.agent.knowledge.sync == "background"
    assert cfg.agent.knowledge.sediment_sessions is False
    assert cfg.agent.recipe_kb.auto_sync == "branch"

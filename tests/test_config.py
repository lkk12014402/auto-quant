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
    ssh_cfg = load_run_config(os.path.join(EXAMPLES, "ssh_docker", "run_ssh.yaml"))
    assert ssh_cfg.executor.type == "ssh"
    assert ssh_cfg.quant.scheme == "MXFP4"  # inherited from base
    assert ssh_cfg.executor.ssh.gpus == "0,1,2,3"


def test_invalid_model_rejected():
    with pytest.raises(ConfigError):
        parse_run_config({"model": "notarepo"})


def test_validate_nudges_missing_writeback(tmp_path, monkeypatch, capsys):
    """env 里有回传变量而 yaml 没配 sink/knowledge.repo 时，validate 要提醒。"""
    import os
    from click.testing import CliRunner
    from aqa.cli import main

    y = tmp_path / "r.yaml"
    y.write_text(f"model: org/m\neval:\n  benchmarks: [piqa]\n"
                 f"output:\n  dir: {tmp_path}/out\n")
    monkeypatch.setenv("LOCAL_RESULTS_DATASET", "org/ds")
    monkeypatch.setenv("LOCAL_RESULTS_HF_TOKEN", "tok")
    r = CliRunner().invoke(main, ["validate", str(y)])
    assert r.exit_code == 0, r.output
    assert "hf_dataset" in r.output and "knowledge.repo" in r.output

    # 配全了就不提醒
    y2 = tmp_path / "r2.yaml"
    y2.write_text(f"model: org/m\neval:\n  benchmarks: [piqa]\n"
                  f"output:\n  dir: {tmp_path}/out\n"
                  f"  sinks:\n    - type: hf_dataset\n      repo: org/ds\n"
                  f"      token_env: LOCAL_RESULTS_HF_TOKEN\n"
                  f"agent:\n  knowledge:\n    repo: org/ds\n")
    r2 = CliRunner().invoke(main, ["validate", str(y2)])
    assert "hf_dataset" not in r2.output.replace("OK", "") or "note:" not in r2.output

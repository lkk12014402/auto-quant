"""Tests for the evalscope harness (command building + report parsing)."""

from __future__ import annotations

import json
from pathlib import Path

from aqa.eval.harness.evalscope import EvalScopeHarness
from aqa.eval.registry import BenchmarkRecipe
from aqa.registry import HARNESSES


def _recipe(name: str = "aime26") -> BenchmarkRecipe:
    return BenchmarkRecipe(name=name, harness="evalscope", type="generative")


def test_registered():
    assert HARNESSES.get("evalscope") is EvalScopeHarness


def test_build_serve_command():
    h = EvalScopeHarness(num_gpus=4, port=8801,
                         server_args={"max_model_len": 32768, "enforce_eager": True})
    cmd = h.build_serve_command("org/m", "srv")
    assert cmd[:2] == ["vllm", "serve"]
    assert "--served-model-name" in cmd and "srv" in cmd
    assert "--tensor-parallel-size" in cmd and "4" in cmd
    assert "--max-model-len" in cmd and "32768" in cmd
    assert "--enforce-eager" in cmd


def test_build_eval_command_defaults():
    h = EvalScopeHarness(num_gpus=2, repeats=4, limit=5)
    cmd = h.build_eval_command([_recipe("aime26"), _recipe("gpqa_diamond")], "srv",
                               Path("/out"))
    s = " ".join(cmd)
    assert "--eval-type openai_api" in s
    assert "--api-url http://127.0.0.1:8801/v1" in s
    assert "--datasets aime26 gpqa_diamond" in s
    assert "--repeats 4" in s
    assert "--limit 5" in s
    assert "--no-timestamp" in s
    ds_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    assert ds_args == {"aime26": {"few_shot_num": 0}, "gpqa_diamond": {"few_shot_num": 0}}
    gen = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert gen["max_tokens"] == 30000 and "extra_body" not in gen


def test_build_eval_command_thinking_off():
    h = EvalScopeHarness(chat_template_kwargs={"enable_thinking": False})
    cmd = h.build_eval_command([_recipe()], "srv", Path("/out"))
    gen = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert gen["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_result_for_parses_report(tmp_path):
    served = "srv"
    report = tmp_path / "reports" / served / "aime26.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "schema_version": 2,
        "metrics": [{"identity": {"name": "accuracy", "aggregation": "mean"},
                     "score": 0.7, "num": 30}],
    }))
    h = EvalScopeHarness()
    res = h._result_for(_recipe("aime26"), served, tmp_path)
    assert res.score == 0.7 and res.metric == "accuracy" and res.num_samples == 30


def test_recipe_files_load():
    from aqa.eval.registry import BenchmarkRegistry

    reg = BenchmarkRegistry()
    for name in ("aime26", "aime25", "gpqa_diamond"):
        r = reg.get(name)
        assert r.harness == "evalscope", name
        assert r.type == "generative", name


def test_runner_constructs_with_num_gpus():
    from aqa.config import EvalConfig
    from aqa.eval.runner import _make_harness

    cfg = EvalConfig(evalscope={"port": 9999, "repeats": 2})
    h = _make_harness("evalscope", cfg, 8)
    assert isinstance(h, EvalScopeHarness)
    assert h.num_gpus == 8 and h.port == 9999 and h.repeats == 2

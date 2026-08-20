"""Tests for the eval-aware self-heal layer (classification + deterministic
quick-fixes + the pipeline pre-pass that resolves failures without the agent).
"""

from __future__ import annotations

from aqa.config import EvalConfig
from aqa.eval.heal import (
    EvalHealer,
    classify_eval_failure,
    eval_agent_hint,
)


def test_classify_covers_eval_failure_classes():
    assert classify_eval_failure("torch.cuda.OutOfMemoryError: CUDA out of memory") == "oom"
    assert classify_eval_failure("No available memory for the cache blocks") == "oom"
    assert classify_eval_failure("HTTPConnectionPool: Max retries exceeded") == "endpoint"
    assert classify_eval_failure("EngineCore proc died unexpectedly") == "engine_crash"
    assert classify_eval_failure("Tokenizer does not have a chat template") == "chat_template"
    assert classify_eval_failure("Tasks were not found: gpqa_typo") == "task_not_found"
    assert classify_eval_failure("ModuleNotFoundError: No module named 'ifeval'") == "dependency"
    assert classify_eval_failure("all benchmark scores are 0 (acc=0 => fail)") == "empty_output"
    assert classify_eval_failure("something totally unexpected") == "unknown"


def test_oom_shrinks_batch_then_memory():
    cfg = EvalConfig(batch_size="auto")
    h = EvalHealer(cfg)
    qf = h.propose("CUDA out of memory")
    assert qf and qf.failure == "oom" and cfg.batch_size == "16"
    for expected in ("8", "4", "2", "1"):
        qf = h.propose("CUDA out of memory")
        assert cfg.batch_size == expected
    # batch exhausted -> now lowers gpu_memory_utilization down the ladder.
    qf = h.propose("CUDA out of memory")
    assert cfg.lm_eval.model_args["gpu_memory_utilization"] == 0.8
    for expected_mem in (0.7, 0.6, 0.5):
        qf = h.propose("CUDA out of memory")
        assert cfg.lm_eval.model_args["gpu_memory_utilization"] == expected_mem
    # fully exhausted -> escalate (None).
    assert h.propose("CUDA out of memory") is None


def test_transient_endpoint_retries_then_gives_up():
    slept: list[float] = []
    h = EvalHealer(EvalConfig(), max_transient_retries=2, sleep=slept.append)
    qf1 = h.propose("Connection refused")
    assert qf1 and qf1.transient and qf1.failure == "endpoint"
    qf2 = h.propose("Connection refused")
    assert qf2 and qf2.transient
    assert h.propose("Connection refused") is None  # budget exhausted
    assert slept == [5.0, 10.0]  # linear backoff


def test_chat_template_toggles_once():
    cfg = EvalConfig()
    assert cfg.lm_eval.apply_chat_template is None
    h = EvalHealer(cfg)
    qf = h.propose("does not have a chat template")
    assert qf and cfg.lm_eval.apply_chat_template is True
    assert h.propose("does not have a chat template") is None  # only once


def test_empty_output_routes_to_chat_template_toggle():
    cfg = EvalConfig()
    h = EvalHealer(cfg)
    qf = h.propose("all benchmark scores are 0 (acc=0 => fail)")
    assert qf and cfg.lm_eval.apply_chat_template is True
    assert h.last_failure == "empty_output"


def test_unknown_failure_has_no_quick_fix():
    assert EvalHealer(EvalConfig()).propose("segfault in some C extension") is None


def test_agent_hints_present_for_known_classes():
    for cls in ("oom", "endpoint", "engine_crash", "chat_template",
                "task_not_found", "dependency", "empty_output"):
        assert eval_agent_hint(cls)
    assert eval_agent_hint("unknown") == ""

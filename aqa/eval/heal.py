"""Eval-aware self-heal -- the "agent-as-operator" layer for the eval RUN itself.

The generic agent fix-loop (:mod:`aqa.agent.fix_loop`) can already drive a failing
``evaluate`` phase to success by editing the working tree. But many eval failures
are *operational* and don't need an LLM at all: an OOM just needs a smaller batch,
a flaky endpoint just needs a retry, an all-zero instruct run usually needs the
chat template toggled. Spending an agent tier (minutes + tokens) on those is waste.

This module adds a cheap **deterministic pre-pass**: classify the eval failure and
apply a targeted config quick-fix, retrying without the agent. Only failures that
resist the quick-fixes escalate to the agent -- and when they do, we hand the agent
an eval-specific hint so it fixes the *eval env/config*, not quantization. This is
the differentiator over launcher-style tools (NEL): the eval run heals itself.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import EvalConfig
from ..logging_util import get_logger

_log = get_logger("eval.heal")

# Ordered (class, pattern) -- first match wins. Eval-specific, distinct from the
# quant-oriented drift classifier.
_EVAL_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("oom", re.compile(
        r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemory|"
        r"No available memory for the cache|KV cache .* (?:memory|blocks)|"
        r"gpu_memory_utilization",
        re.IGNORECASE)),
    ("endpoint", re.compile(
        r"Connection .*refused|Connection refused|Read timed out|Max retries exceeded|"
        r"Connection reset|Failed to establish a new connection|"
        r"ClientConnectorError|50[234] (?:Bad Gateway|Service|Gateway Time)",
        re.IGNORECASE)),
    ("engine_crash", re.compile(
        r"EngineCore .* (?:died|failed)|AsyncLLMEngine .* died|"
        r"[Ee]ngine (?:process|core) .* (?:died|crashed|exited)|"
        r"WorkerProc .* died|RayWorker .* died",
        re.IGNORECASE)),
    ("chat_template", re.compile(
        r"chat template|apply_chat_template|does not have a chat template|"
        r"tokenizer .* template|jinja2?\.exceptions|TemplateError",
        re.IGNORECASE)),
    ("task_not_found", re.compile(
        r"Task[s]? .* not found|not a registered task|could not find task|"
        r"No tasks specified|Tasks were not found",
        re.IGNORECASE)),
    ("dependency", re.compile(
        r"ModuleNotFoundError|No module named|ImportError|cannot import name",
        re.IGNORECASE)),
    ("empty_output", re.compile(
        r"produced no benchmark results|all benchmark scores are 0|acc=0",
        re.IGNORECASE)),
]


def classify_eval_failure(text: str) -> str:
    """Return the eval failure class for ``text`` (``"unknown"`` if none match)."""
    for name, pattern in _EVAL_RULES:
        if pattern.search(text):
            return name
    return "unknown"


# Batch-size shrink ladder for OOM (each step halves the effective batch).
_BATCH_LADDER = ["auto", "16", "8", "4", "2", "1"]
# gpu_memory_utilization shrink ladder (vllm) once batch is already minimal.
_MEM_LADDER = [0.9, 0.8, 0.7, 0.6, 0.5]


@dataclass
class QuickFix:
    """A deterministic remedy applied before escalating to the agent."""

    failure: str
    description: str
    transient: bool = False  # transient => just retry (no config change)


_AGENT_HINTS: dict[str, str] = {
    "oom": (
        "EVAL FAILURE CLASS: out-of-memory. Deterministic batch/memory shrinks were "
        "already tried. Reduce memory further (lower max_model_len, max_num_seqs, or "
        "gpu_memory_utilization; enable tensor parallelism) rather than changing the model."
    ),
    "endpoint": (
        "EVAL FAILURE CLASS: the model endpoint/engine was unreachable. Verify the "
        "server is up and healthy (URL, port, health check) and restart it if needed; "
        "do NOT change the benchmark."
    ),
    "engine_crash": (
        "EVAL FAILURE CLASS: the inference engine (vLLM) crashed. Inspect the engine "
        "log for the real cause (bad dtype/kv-cache/driver) and fix the engine launch, "
        "not the eval task."
    ),
    "chat_template": (
        "EVAL FAILURE CLASS: chat-template/tokenizer problem. Ensure the tokenizer has a "
        "valid chat template and that apply_chat_template matches the model; a zero score "
        "on an instruct model usually means the template is wrong or missing."
    ),
    "task_not_found": (
        "EVAL FAILURE CLASS: the benchmark task is not registered in this harness. Install/"
        "upgrade the harness or correct the task name / lm_eval_task in the recipe."
    ),
    "dependency": (
        "EVAL FAILURE CLASS: a Python dependency is missing. pip-install the missing module "
        "into the eval environment and re-run; do not alter the benchmark logic."
    ),
    "empty_output": (
        "EVAL FAILURE CLASS: the run produced no usable scores (empty or all-zero). Check the "
        "generation/parsing path (chat template, max_gen_toks, stop tokens, answer extraction)."
    ),
}


def eval_agent_hint(failure: str) -> str:
    """An eval-focused instruction appended to the agent prompt on escalation."""
    return _AGENT_HINTS.get(failure, "")


class EvalHealer:
    """Owns the (mutable) eval config and proposes deterministic quick-fixes.

    ``propose(log)`` classifies the failure, mutates ``cfg`` in place when a cheap
    remedy exists, and returns the applied :class:`QuickFix` (or ``None`` when no
    deterministic fix applies or the ladder is exhausted -- signalling "escalate").
    Because the pipeline's eval phase reads the *same* ``cfg`` object, mutations
    take effect on the next re-run.
    """

    def __init__(
        self,
        cfg: EvalConfig,
        *,
        max_transient_retries: int = 2,
        backoff_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self.applied: list[str] = []
        self.last_failure = ""
        self._transient = 0
        self._max_transient = max_transient_retries
        self._backoff = backoff_s
        self._sleep = sleep

    def propose(self, log_text: str) -> QuickFix | None:
        cls = classify_eval_failure(log_text)
        self.last_failure = cls
        if cls == "oom":
            return self._fix_oom()
        if cls in ("endpoint", "engine_crash"):
            return self._fix_transient(cls)
        if cls in ("chat_template", "empty_output"):
            return self._fix_chat_template(cls)
        return None

    # -- individual remedies ------------------------------------------------
    def _fix_oom(self) -> QuickFix | None:
        cur = str(self.cfg.batch_size)
        idx = _BATCH_LADDER.index(cur) if cur in _BATCH_LADDER else 0
        if idx + 1 < len(_BATCH_LADDER):
            new = _BATCH_LADDER[idx + 1]
            self.cfg.batch_size = new
            self.applied.append(f"batch_size={new}")
            return QuickFix("oom", f"OOM: shrink batch_size {cur} -> {new}")
        # batch already minimal -> lower vllm gpu_memory_utilization.
        args = self.cfg.lm_eval.model_args
        cur_mem = float(args.get("gpu_memory_utilization", _MEM_LADDER[0]))
        below = [m for m in _MEM_LADDER if m < cur_mem]
        if below:
            new_mem = below[0]
            args["gpu_memory_utilization"] = new_mem
            self.applied.append(f"gpu_memory_utilization={new_mem}")
            return QuickFix("oom", f"OOM: lower gpu_memory_utilization {cur_mem} -> {new_mem}")
        return None

    def _fix_transient(self, cls: str) -> QuickFix | None:
        if self._transient >= self._max_transient:
            return None
        self._transient += 1
        wait = self._backoff * self._transient
        _log.info("transient %s failure -> retry %d/%d after %.0fs",
                  cls, self._transient, self._max_transient, wait)
        self._sleep(wait)
        return QuickFix(cls, f"{cls}: transient, retry {self._transient}/{self._max_transient}",
                        transient=True)

    def _fix_chat_template(self, cls: str) -> QuickFix | None:
        if "chat_template_toggled" in self.applied:
            return None
        current = self.cfg.lm_eval.apply_chat_template
        new = not current if current is not None else True
        self.cfg.lm_eval.apply_chat_template = new
        self.applied.append("chat_template_toggled")
        return QuickFix(cls, f"{cls}: toggle apply_chat_template {current} -> {new}")

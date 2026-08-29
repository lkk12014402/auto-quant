"""Deterministic drift signals: error classification, similarity, progress depth.

Ports ``taxonomy_classify`` / ``logs_are_similar`` / ``extract_progress`` from
``local_dispatch/agent/agent_fix_loop.sh`` into self-contained Python (no external
taxonomy module).  These feed the fix-loop's drift detector: the agent's own
ERROR_CLASS is primary, this classifier is the fallback, and text similarity is
the tie-break when both are ``unknown``.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

# Ordered (category, pattern) rules -- first match wins.  Kept small and stable so the
# same underlying error classifies identically across attempts (drift stability).
#
# Ordering matters: put SPECIFIC patterns before generic ones so, e.g., a CUDA OOM
# classifies as ``oom`` (not the generic ``runtime_error`` its RuntimeError wrapper
# would otherwise match). Categories are intentionally fine-grained: a coarse class
# lumps unrelated root causes together and hurts patch-replay precision.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    # -- memory / device capacity ---------------------------------------------
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemory|"
                       r"CUDA error: out of memory", re.IGNORECASE)),
    ("cpu_oom", re.compile(r"DefaultCPUAllocator: can't allocate|"
                           r"Cannot allocate memory|std::bad_alloc|Killed\b.*memory",
                           re.IGNORECASE)),
    ("disk_space", re.compile(r"No space left on device|OSError: \[Errno 28\]|"
                              r"Disk quota exceeded", re.IGNORECASE)),
    # -- tensor / numeric layout ----------------------------------------------
    ("shape_mismatch", re.compile(r"size mismatch|shapes? cannot be multiplied|"
                                  r"mat1 and mat2|must match the size|"
                                  r"expected .* but got .* shape|"
                                  r"The size of tensor .* must match", re.IGNORECASE)),
    ("dtype_mismatch", re.compile(r"expected .* dtype|Input type .* and weight type|"
                                  r"promote|got Half|got BFloat16|mixed dtype|"
                                  r"expected scalar type|can't convert .* dtype", re.IGNORECASE)),
    ("nan_inf", re.compile(r"\bNaN\b|\bInf\b|not finite|loss is nan|"
                           r"probability tensor contains", re.IGNORECASE)),
    ("meta_device_error", re.compile(r"meta device|Cannot copy out of meta tensor|"
                                     r"to_empty\(\)|on the meta device", re.IGNORECASE)),
    # -- CUDA runtime / kernels ------------------------------------------------
    ("cuda_unavailable", re.compile(r"CUDA .*not available|no CUDA-capable device|"
                                    r"torch\.cuda\.is_available\(\) .*False|"
                                    r"Found no NVIDIA driver", re.IGNORECASE)),
    ("cuda_arch_mismatch", re.compile(r"no kernel image is available|"
                                      r"CUDA capability sm_|kernel image .* for the device|"
                                      r"PTX .* not supported", re.IGNORECASE)),
    ("cuda_runtime_error", re.compile(r"CUDA error:|device-side assert|"
                                      r"an illegal memory access|cublas.*error|cudnn.*error",
                                      re.IGNORECASE)),
    # -- model config / weights loading ---------------------------------------
    ("unrecognized_config", re.compile(r"Unrecognized configuration class|"
                                       r"does not recognize this architecture|"
                                       r"is not supported for", re.IGNORECASE)),
    ("checkpoint_key_mismatch", re.compile(r"Missing key|Unexpected key|"
                                           r"Error\(s\) in loading state_dict|"
                                           r"size of tensor .* must match", re.IGNORECASE)),
    # context-length overflow (specific; before the broad tokenizer word below)
    ("context_length", re.compile(r"max_position_embeddings|"
                                  r"sequence length must be less than|"
                                  r"maximum sequence length|context length",
                                  re.IGNORECASE)),
    ("tokenizer_error", re.compile(r"tokenizer|SentencePiece|added_tokens|"
                                   r"vocab_size|chat[_ ]template", re.IGNORECASE)),
    # -- dependency / environment ---------------------------------------------
    # a missing CLI/tool is not an ImportError -- check BEFORE import_error
    ("missing_tool", re.compile(r"is not installed|not found on PATH|command not found|"
                                r"No such file or directory.*bin/", re.IGNORECASE)),
    ("import_error", re.compile(r"ModuleNotFoundError|ImportError|No module named|"
                                r"cannot import name|undefined symbol|"
                                r"version .* GLIBC", re.IGNORECASE)),
    ("permission_error", re.compile(r"PermissionError|Permission denied|"
                                    r"\[Errno 13\]", re.IGNORECASE)),
    # -- serving / inference engine -------------------------------------------
    # vllm-omni double-init of the diffusion parallel groups (specific; before
    # the generic engine + exception buckets)
    ("engine_reinit", re.compile(r"model parallel state must be empty|"
                                 r"already initialized.*(parallel|process group)|"
                                 r"destroy_process_group", re.IGNORECASE)),
    ("vllm_engine_error", re.compile(r"EngineCore|AsyncLLMEngine|Engine core initialization|"
                                     r"engine .* died|RayWorker|vllm.*Error", re.IGNORECASE)),
    # quantized-checkpoint artifact problems (missing packed weight, mangled
    # component) -- after the engine rules, before the coarse exception buckets
    ("quant_artifact", re.compile(r"qweight|qzeros|quantization_config|"
                                  r"no module or parameter named", re.IGNORECASE)),
    # -- network / download (transient vs permanent split) --------------------
    ("auth_error", re.compile(r"401 Client Error|403 Client Error|"
                              r"Repository Not Found|GatedRepo|requires approval|"
                              r"Access denied|Invalid.*token|Unauthorized", re.IGNORECASE)),
    ("download_missing", re.compile(r"Entry Not Found|404 Client Error|"
                                    r"EntryNotFound|does not appear to have a file", re.IGNORECASE)),
    ("network_transient", re.compile(r"Connection .*refused|Connection reset|"
                                     r"Read timed out|ConnectionError|"
                                     r"Max retries exceeded|Temporary failure|"
                                     r"ReadTimeout|502 |503 |504 ", re.IGNORECASE)),
    # -- generic Python exception classes (last-resort buckets) ---------------
    ("key_error", re.compile(r"\bKeyError\b", re.IGNORECASE)),
    ("index_error", re.compile(r"\bIndexError\b", re.IGNORECASE)),
    ("type_error", re.compile(r"\bTypeError\b", re.IGNORECASE)),
    ("attribute_error", re.compile(r"\bAttributeError\b", re.IGNORECASE)),
    ("assertion_error", re.compile(r"\bAssertionError\b", re.IGNORECASE)),
    ("value_error", re.compile(r"\bValueError\b", re.IGNORECASE)),
    ("runtime_error", re.compile(r"\bRuntimeError\b", re.IGNORECASE)),
]

# Deepest exception CLASS name in a traceback (Harbor-style structured signal):
# ``...\nSomeModule.SomeError: message`` -> ``SomeError``. Precise and maintenance-free,
# it complements the semantic taxonomy above (which groups synonyms like a CUDA OOM
# RuntimeError and OutOfMemoryError into one ``oom`` category).
_EXC_TYPE_RE = re.compile(
    r"^[\s>*\-]*(?:[A-Za-z_][\w.]*\.)?"          # optional module path prefix
    r"([A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Timeout))\s*:",
    re.MULTILINE,
)

# torch-dynamo wraps the true exception as ``raised exception <Name>(...)``; the
# deepest colon-form in the traceback is then only a generic RuntimeError wrapper.
# Recover the inner class name so replay keys off the real exception type.
_DYNAMO_EXC_TYPE_RE = re.compile(
    r"raised exception\s+([A-Za-z_][\w.]*(?:Error|Exception))\("
)

# Volatile, non-error chatter dropped before classification/similarity so the same
# crash looks the same across attempts.
_VOLATILE = re.compile(
    r"A new version of the following files was downloaded|"
    r"Make sure to double-check|"
    r"\d{3,} Client Error|Entry Not Found for url|Request ID: Root=|"
    r"You are sending unauthenticated requests|"
    r"Generating [\w-]+ split:|"
    r"^(Map|Filter|Downloading|Resolving|Loading)[\w ]*:\s*\d+%|"
    r"\b\d+%\|[#\u2588\u2589\u258a\u258b\u258c\u258d\u258e\u258f ]*\||"
    r"falling back to in-process mode|examples/s\]\s*$"
)


def _denoise_for_class(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _VOLATILE.search(ln)]
    return "\n".join(lines) or text


@dataclass
class ClassifyDetail:
    """Richer classification prior: the class plus *why* it was chosen."""

    class_name: str
    rule_id: str = ""   # "<category>#<rule index>" of the matching rule
    evidence: str = ""  # first matching (denoised) log line, capped at 200 chars


def classify_error_detail(text: str) -> ClassifyDetail:
    """Classify ``text`` and report the matching rule + evidence line."""
    cleaned = _denoise_for_class(text or "")
    for idx, (category, pat) in enumerate(_RULES):
        if not pat.search(cleaned):
            continue
        evidence = ""
        for line in cleaned.splitlines():
            if pat.search(line):
                evidence = line.strip()[:200]
                break
        return ClassifyDetail(class_name=category,
                              rule_id=f"{category}#{idx}", evidence=evidence)
    return ClassifyDetail(class_name="unknown")


def classify_error(text: str) -> str:
    """Return a stable snake_case error category (``unknown`` if nothing matches)."""
    return classify_error_detail(text).class_name


def exception_type(text: str) -> str:
    """Deepest Python exception CLASS name in a traceback, or ``""`` if none.

    A precise, maintenance-free structured signal (mirrors Harbor's
    ``type(e).__name__``): ``RuntimeError: size mismatch`` -> ``RuntimeError``,
    ``modeling_qwen.MyError: ...`` -> ``MyError``. Used as a mid-precision replay
    key between the exact error signature and the coarse semantic category.
    """
    cleaned = _denoise_for_class(text or "")
    dynamo = _DYNAMO_EXC_TYPE_RE.findall(cleaned)
    if dynamo:
        name = dynamo[-1]
        return name.rsplit(".", 1)[-1]
    matches = _EXC_TYPE_RE.findall(cleaned)
    return matches[-1] if matches else ""


def _denoise_for_sim(text: str) -> str:
    out: list[str] = []
    for ln in (text or "").splitlines():
        if re.search(r"HTTP Request|HTTP/1\.1|Client Error|Downloading|it/s\]|"
                     r"\|\s*\d+/\d+|Config was last written|allowlist contains|"
                     r"WARNING logging", ln):
            continue
        s = re.sub(r"^\S*\d{4}-\d\d-\d\dT[\d:.]+Z?\s*", "", ln)
        s = re.sub(r"\b\d{1,2}:\d{2}:\d{2}\b", "", s)
        s = re.sub(r"\[[A-Z]+\]", "", s)
        s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
        s = re.sub(r"/[^\s:]+/", "/PATH/", s)
        s = re.sub(r"\d+\.\d+\s?[GMK]i?B", "SIZE", s)
        s = re.sub(r"line \d+", "line N", s)
        s = re.sub(r"\d+", "N", s)
        s = s.strip()
        if s:
            out.append(s)
    return "\n".join(out)


def logs_similarity(a: str, b: str) -> float | None:
    """Denoised difflib ratio of two error logs, or None if either is empty."""
    da, db = _denoise_for_sim(a), _denoise_for_sim(b)
    if not da or not db:
        return None
    return difflib.SequenceMatcher(None, da, db).ratio()


def extract_progress(text: str) -> int:
    """Deepest ``layers.<N>`` index seen in a log (proxy for how far quant got); -1 if none."""
    nums = [int(m) for m in re.findall(r"layers\.(\d+)", text or "")]
    return max(nums) if nums else -1

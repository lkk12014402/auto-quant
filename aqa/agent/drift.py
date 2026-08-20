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

# Ordered (category, pattern) rules -- first match wins.  Kept small and stable so the
# same underlying error classifies identically across attempts (drift stability).
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemory", re.IGNORECASE)),
    ("shape_mismatch", re.compile(r"size mismatch|shapes? cannot be multiplied|"
                                  r"mat1 and mat2|expected .* but got .* shape", re.IGNORECASE)),
    ("dtype_mismatch", re.compile(r"expected .* dtype|Input type .* and weight type|"
                                  r"promote|got Half|got BFloat16|mixed dtype", re.IGNORECASE)),
    ("meta_device_error", re.compile(r"meta device|Cannot copy out of meta tensor|"
                                     r"to_empty\(\)", re.IGNORECASE)),
    ("unrecognized_config", re.compile(r"Unrecognized configuration class|"
                                       r"KeyError.*config|does not recognize this architecture", re.IGNORECASE)),
    ("missing_key", re.compile(r"KeyError|Missing key|Unexpected key|"
                               r"size of tensor .* must match", re.IGNORECASE)),
    ("import_error", re.compile(r"ModuleNotFoundError|ImportError|No module named", re.IGNORECASE)),
    ("attribute_error", re.compile(r"AttributeError", re.IGNORECASE)),
    ("cuda_unavailable", re.compile(r"CUDA .*not available|no CUDA-capable device|"
                                    r"torch\.cuda\.is_available\(\) .*False", re.IGNORECASE)),
    ("vllm_engine_error", re.compile(r"EngineCore|AsyncLLMEngine|vllm.*Error|"
                                     r"engine .* died|RayWorker", re.IGNORECASE)),
    ("download_error", re.compile(r"Entry Not Found|Repository Not Found|"
                                  r"HfHubHTTPError|401 Client Error|403 Client Error|"
                                  r"Connection .* refused|Read timed out", re.IGNORECASE)),
    ("assertion_error", re.compile(r"AssertionError", re.IGNORECASE)),
    ("value_error", re.compile(r"ValueError", re.IGNORECASE)),
    ("runtime_error", re.compile(r"RuntimeError", re.IGNORECASE)),
]

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


def classify_error(text: str) -> str:
    """Return a stable snake_case error category (``unknown`` if nothing matches)."""
    cleaned = _denoise_for_class(text or "")
    for category, pat in _RULES:
        if pat.search(cleaned):
            return category
    return "unknown"


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

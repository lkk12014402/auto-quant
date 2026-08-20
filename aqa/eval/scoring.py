"""Native scorer library.

Adapted from NVIDIA-NeMo/Evaluator (Apache-2.0): the built-in scoring functions
(exact / numeric / multichoice-regex / fuzzy / answer-line) that back the
``@scorer`` API upstream.  Here they are plain callables referenced by name from
benchmark recipes (``scoring: {scorer: numeric_match}``) and via the
``aqa.scorers`` entry-point group.

Every scorer takes a :class:`ScorerInput` and returns a metrics dict that always
contains a numeric ``score`` in ``[0, 1]`` plus any auxiliary fields.
"""

from __future__ import annotations

import json
import os
import re
import string
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ScorerInput:
    """A single (prediction, target) pair to score, plus optional context."""

    prediction: str
    target: str
    choices: list[str] | None = None  # for multiple-choice benchmarks
    extra: dict[str, Any] = field(default_factory=dict)


# ── Normalisation helpers ─────────────────────────────────────────────────────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lower-case, strip articles/punctuation, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_last_number(text: str) -> str | None:
    """Return the last number in ``text`` (GSM8K-style answers), or None."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "").replace("$", "")
    return cleaned.rstrip(".")


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Scorers ───────────────────────────────────────────────────────────────────
def exact_match(sample: ScorerInput) -> dict[str, Any]:
    """1.0 iff the normalised prediction equals the normalised target."""
    pred = normalize_text(sample.prediction)
    tgt = normalize_text(sample.target)
    return {"score": float(pred == tgt), "metric": "exact_match"}


def numeric_match(sample: ScorerInput) -> dict[str, Any]:
    """Compare the last number in prediction vs target (GSM8K / math word problems)."""
    pred_raw = extract_last_number(sample.prediction)
    tgt_raw = extract_last_number(sample.target) or sample.target.strip()
    pred_val = _to_float(pred_raw) if pred_raw is not None else None
    tgt_val = _to_float(tgt_raw)
    if pred_val is None or tgt_val is None:
        matched = (pred_raw or "") == (tgt_raw or "")
    else:
        matched = abs(pred_val - tgt_val) < 1e-6
    return {"score": float(matched), "metric": "numeric_match",
            "pred_number": pred_raw, "target_number": tgt_raw}


def answer_line(sample: ScorerInput) -> dict[str, Any]:
    """Extract the text after an ``Answer:`` / ``####`` marker, then exact-match."""
    text = sample.prediction
    marker = None
    for pat in (r"####\s*(.+)", r"(?i)answer\s*:\s*(.+)", r"(?i)the answer is\s*(.+)"):
        m = re.search(pat, text)
        if m:
            marker = m.group(1).strip()
            break
    extracted = marker if marker is not None else text.strip().splitlines()[-1] if text.strip() else ""
    return exact_match(ScorerInput(prediction=extracted, target=sample.target)) | {
        "metric": "answer_line", "extracted": extracted,
    }


_CHOICE_LETTER_RE = re.compile(r"\b([A-J])\b")
_CHOICE_MARKER_RES = [
    re.compile(r"(?:answer|option|choice)\b[^A-Za-z]{0,12}?([A-J])\b", re.IGNORECASE),
    re.compile(r"\(([A-J])\)"),
    re.compile(r"\b([A-J])[.):]"),
]


def _extract_choice_letter(text: str) -> str | None:
    """Extract a choice letter, preferring explicit answer markers.

    Falls back to the *last* standalone A-J token so a leading pronoun "I" in
    prose like "I pick A" does not shadow the real answer.
    """
    upper = text.upper()
    for pat in _CHOICE_MARKER_RES:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    letters = _CHOICE_LETTER_RE.findall(upper)
    return letters[-1] if letters else None


def multichoice_regex(sample: ScorerInput) -> dict[str, Any]:
    """Extract a choice letter (A, B, C, ...) from the prediction and compare.

    ``target`` may be a letter ("B") or the full choice text; when ``choices`` is
    provided the target text is resolved to its letter for comparison.
    """
    pred_letter = _extract_choice_letter(sample.prediction)

    tgt = sample.target.strip()
    tgt_letter = tgt.upper() if len(tgt) == 1 and tgt.upper().isalpha() else None
    if tgt_letter is None and sample.choices:
        for idx, choice in enumerate(sample.choices):
            if normalize_text(choice) == normalize_text(tgt):
                tgt_letter = string.ascii_uppercase[idx]
                break
    matched = pred_letter is not None and pred_letter == tgt_letter
    return {"score": float(matched), "metric": "multichoice_regex",
            "pred_letter": pred_letter, "target_letter": tgt_letter}


def fuzzy_match(sample: ScorerInput) -> dict[str, Any]:
    """Token-overlap F1 between normalised prediction and target (SQuAD/DROP-style)."""
    pred_tokens = normalize_text(sample.prediction).split()
    tgt_tokens = normalize_text(sample.target).split()
    if not pred_tokens or not tgt_tokens:
        score = float(pred_tokens == tgt_tokens)
        return {"score": score, "metric": "fuzzy_match", "f1": score}
    common: dict[str, int] = {}
    for tok in pred_tokens:
        if tok in tgt_tokens:
            common[tok] = min(pred_tokens.count(tok), tgt_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return {"score": 0.0, "metric": "fuzzy_match", "f1": 0.0}
    precision = num_same / len(pred_tokens)
    recall = num_same / len(tgt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"score": f1, "metric": "fuzzy_match", "f1": f1}


BUILTIN_SCORERS = {
    "exact_match": exact_match,
    "numeric_match": numeric_match,
    "answer_line": answer_line,
    "multichoice_regex": multichoice_regex,
    "fuzzy_match": fuzzy_match,
}


# ── LLM-as-judge scorer ───────────────────────────────────────────────────────
# Unlocks the open-ended / rubric family (NEL: AA-Omniscience Gemini judge,
# GDPVal pairwise/rubric, open-ended IF). A judge is any OpenAI-compatible chat
# endpoint; it is injected so the scorer stays unit-testable with a fake client.

DEFAULT_JUDGE_SYSTEM = "You are a strict, fair evaluator. Follow the rubric exactly."

DEFAULT_JUDGE_TEMPLATE = (
    "You are grading a model's answer to a question.\n\n"
    "Question:\n{question}\n\n"
    "Reference answer:\n{target}\n\n"
    "Model answer:\n{prediction}\n\n"
    "Rate how correct and complete the model answer is on a scale of 0 to {scale} "
    "(0 = entirely wrong, {scale} = fully correct). Respond with a single line "
    "`SCORE: <number>` and nothing else."
)

_SCORE_RE = re.compile(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_YES_RE = re.compile(r"\b(yes|correct|pass|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|incorrect|fail|false)\b", re.IGNORECASE)


class JudgeClient(Protocol):
    """Minimal chat client: one system+user turn in, judge text out."""

    def complete(self, system: str, user: str) -> str:
        ...


class HttpJudgeClient:
    """Default judge over an OpenAI-compatible ``/chat/completions`` endpoint (stdlib only)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]


def parse_judge_score(text: str, scale: float) -> float:
    """Parse a judge reply into a raw score on ``[0, scale]``.

    Prefers an explicit ``SCORE: <n>`` line; falls back to the last number, then a
    yes/no verdict. Values are clamped to ``[0, scale]``.
    """
    match = _SCORE_RE.search(text)
    if match is None:
        numbers = _NUMBER_RE.findall(text)
        if numbers:
            match_val = _to_float(numbers[-1].replace(",", "").replace("$", ""))
        elif _YES_RE.search(text) and not _NO_RE.search(text):
            match_val = scale
        elif _NO_RE.search(text):
            match_val = 0.0
        else:
            match_val = None
    else:
        match_val = _to_float(match.group(1))
    if match_val is None:
        return 0.0
    return max(0.0, min(scale, match_val))


def build_judge_scorer(spec: Any, client: JudgeClient | None = None) -> Any:
    """Build an ``llm_judge`` scorer callable from a ``JudgeSpec``-like object.

    ``spec`` may be a ``JudgeSpec`` or ``None`` (all defaults). ``client`` is
    injected in tests; when omitted a lazy :class:`HttpJudgeClient` is built on
    first use from the spec (so importing/registering never needs a live endpoint).
    """
    scale = float(getattr(spec, "scale", 10.0) or 10.0)
    system = getattr(spec, "system", None) or DEFAULT_JUDGE_SYSTEM
    template = getattr(spec, "template", None) or DEFAULT_JUDGE_TEMPLATE
    threshold = getattr(spec, "threshold", None)

    def _client() -> JudgeClient:
        nonlocal client
        if client is None:
            client = HttpJudgeClient(
                model=getattr(spec, "model", None) or "gpt-4o-mini",
                base_url=getattr(spec, "base_url", None),
                api_key_env=getattr(spec, "api_key_env", None) or "OPENAI_API_KEY",
                temperature=float(getattr(spec, "temperature", 0.0) or 0.0),
                max_tokens=int(getattr(spec, "max_tokens", 512) or 512),
            )
        return client

    def llm_judge(sample: ScorerInput) -> dict[str, Any]:
        question = str(sample.extra.get("question", sample.extra.get("prompt", "")))
        user = template.format(
            question=question, prediction=sample.prediction, target=sample.target, scale=scale,
        )
        reply = _client().complete(system, user)
        raw = parse_judge_score(reply, scale)
        if threshold is not None:
            score = 1.0 if raw >= float(threshold) else 0.0
        else:
            score = raw / scale if scale else 0.0
        return {"score": score, "metric": "llm_judge", "judge_raw": raw, "judge_output": reply}

    return llm_judge


# A zero-config default so ``scoring: {scorer: llm_judge}`` resolves by name; it
# reads a JudgeSpec off ``ScorerInput.extra["judge_spec"]`` when the harness
# provides one, else uses defaults + a lazy HTTP client.
def _default_llm_judge(sample: ScorerInput) -> dict[str, Any]:
    spec = sample.extra.get("judge_spec")
    return build_judge_scorer(spec)(sample)


BUILTIN_SCORERS["llm_judge"] = _default_llm_judge

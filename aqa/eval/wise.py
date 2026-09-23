"""WISE (Verified protocol) scorer: world-knowledge-informed T2I evaluation.

Port of PKU-YuanGroup/WISE's ``data_verified/`` + ``vllm_eval.py`` +
``calculate_verified.py`` (see NOTICE; prompts vendored under
``recipes/data/wise/``).  WISE has **two incompatible scoring protocols** and
this module implements **only the `WISE_Verified` protocol**, which is
offline-runnable:

* 1,000 prompts across six domains (culture 400, time 120, space 120,
  biology 120, physics 120, chemistry 120), keyed by ``prompt_id`` 1..1000;
* a binary 0/1 judge served by vLLM (``Qwen/Qwen3.5-35B-A3B`` by default) over
  an OpenAI-compatible ``/chat/completions`` endpoint;
* per-category means aggregated with the official weights
  ``0.40*CULTURE + 0.12*TIME + 0.12*SPACE + 0.12*BIOLOGY + 0.12*PHYSICS +
  0.12*CHEMISTRY``.

The **legacy GPT-4o protocol (``WISE_legacy/``) is deliberately NOT
implemented**: it needs a paid API and its numbers are *not* comparable to the
official leaderboard.  The widely-quoted HunyuanImage-3.0 ~0.57 figure is the
legacy protocol; on the official WISE_Verified leaderboard (judge
Qwen3.5-35B-A3B) HunyuanImage-3.0 scores **0.4350** overall (rank 16).  Never
compare the two.

The judge is a lazily-constructed, injectable component (``WiseJudge``): the
production :class:`VllmWiseJudge` only speaks HTTP to a vLLM server, so no judge
weights are ever loaded in-process and tests can inject a fake.  Heavy deps
(PIL) import lazily.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.wise")

DATA_DIR = Path(__file__).resolve().parents[2] / "recipes" / "data" / "wise"

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
MAX_EXTRACT_RETRIES = 3

# Official category boundaries (calculate_verified.py, prompt_id inclusive).
PROMPT_ID_RANGES: dict[str, tuple[int, int]] = {
    "CULTURE": (1, 400),
    "TIME": (401, 520),
    "SPACE": (521, 640),
    "BIOLOGY": (641, 760),
    "PHYSICS": (761, 880),
    "CHEMISTRY": (881, 1000),
}

# Official overall-score weights (leadboard.md / calculate_verified.py).
WISE_WEIGHTS: dict[str, float] = {
    "CULTURE": 0.40, "TIME": 0.12, "SPACE": 0.12,
    "BIOLOGY": 0.12, "PHYSICS": 0.12, "CHEMISTRY": 0.12,
}

_SYSTEM_PROMPT = (
    "You are a professional text-to-image quality auditor. "
    "Evaluate the image strictly according to the protocol."
)

# Verbatim WISE_Verified judge instruction (vllm_eval.py build_evaluation_messages).
_USER_TEMPLATE = """Please evaluate this generated image for the WISE benchmark and return ONLY one binary score.

# WISE Text-to-Image Evaluation Protocol

## What WISE Is Evaluating
WISE is a knowledge-intensive text-to-image benchmark. Many prompts do not directly state the final visual answer. Instead, the model must use commonsense, cultural, scientific, spatial, or temporal knowledge to infer what should appear in the image.

Your job is not to judge whether the image is beautiful. Your job is to judge whether the generated image correctly realizes the knowledge-based meaning of the prompt and is visually usable.

## Input Fields

**PROMPT**
The original text-to-image prompt given to the image generation model. It may contain an implicit clue rather than the explicit final answer.

**EXPLANATION**
The reference interpretation used for judging. It explains the intended answer, the required knowledge reasoning chain, and the visual evidence that should appear in a correct image. Treat EXPLANATION as the ground-truth judging guide.

For example:
- If PROMPT says "the round pastry commonly shared during Mid-Autumn Festival family gatherings", EXPLANATION may specify mooncakes. A correct image should show mooncakes, not just any festival food.
- If PROMPT says "a plant kept for many days beside a bright one-sided window", EXPLANATION may specify phototropism. A correct image should show the plant bending toward the light source.
- If PROMPT says "a street in New York when it is midnight in Beijing", EXPLANATION may specify the corresponding local time and expected lighting/activity. A correct image should reflect that inferred local time, not simply show Beijing or generic night.

## How To Judge

Evaluate the image using these checks:
1. Does the image contain the main objects or scene required by the PROMPT?
2. Does it satisfy the intended knowledge-based answer described in the EXPLANATION?
3. Are important relations correct, such as spatial layout, temporal state, physical effect, biological behavior, cultural object, or scientific phenomenon?
4. Is the image visually usable for judging, without obvious collapse, severe deformation, unreadable main objects, or major artifacts?

## Binary Score

**Score: 1**
Give 1 only when both conditions are met:
- The image is semantically correct according to both PROMPT and EXPLANATION.
- The image has no obvious generation failure that prevents reliable judging.

Minor aesthetic weakness, ordinary composition, non-photorealistic style, or lack of artistic beauty should not by itself cause rejection if the semantic target is correct and the image is clear.

**Score: 0**
Give 0 if any of the following applies:
- The image misses the intended answer in EXPLANATION.
- The image only follows surface words in PROMPT but fails the required knowledge inference.
- Key objects, attributes, states, behaviors, or relations are missing or wrong.
- The image contradicts the prompt or explanation.
- The main visual evidence is ambiguous enough that a human judge could not confidently verify correctness.
- The image has obvious visual collapse, severe deformation, garbled main objects, impossible structure, or artifacts that interfere with evaluation.

If there is serious doubt, return 0.

## Output Format

Return exactly one line and nothing else:

Score: 0

or

Score: 1

---

PROMPT: "{prompt}"
EXPLANATION: "{explanation}"

Return only `Score: 0` or `Score: 1`."""


# ── data prep ────────────────────────────────────────────────────────────────
def convert_wise_prompts(source_json: str | Path, out_path: str | Path) -> list[dict]:
    """Convert ``data_verified/merge.json`` into one-record-per-prompt JSONL.

    Records: ``{idx, prompt_id, prompt, explanation, category, subcategory}`` in
    ascending ``prompt_id`` order; ``idx = prompt_id - 1`` aligns with the
    diffusion harness's flat generation layout (``<idx>.png``).
    """
    data = json.loads(Path(source_json).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data_list", list(data.values()))
    records = []
    for item in sorted(data, key=lambda x: x["prompt_id"]):
        records.append({
            "idx": int(item["prompt_id"]) - 1,
            "prompt_id": int(item["prompt_id"]),
            "prompt": item["Prompt"],
            "explanation": item["Explanation"],
            "category": item["Category"],
            "subcategory": item["Subcategory"],
        })
    out_path = Path(out_path)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    return records


def load_records(path: str | Path) -> list[dict]:
    """Read a converted prompts.jsonl."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── judge protocol (vllm_eval.py ports) ──────────────────────────────────────
def category_for_prompt_id(prompt_id: int) -> str | None:
    """Official category for a prompt id (``calculate_verified.py`` boundaries)."""
    for category, (lo, hi) in PROMPT_ID_RANGES.items():
        if lo <= prompt_id <= hi:
            return category
    return None


def calculate_wiscore(score: float) -> float:
    """Binary WiScore for one sample (the verified protocol uses the score as-is)."""
    return float(score)


def extract_scores(txt: str) -> dict[str, float]:
    """Parse ``Score: 0|1`` (bold markers tolerated), else a bare single ``0``/``1``.

    Port of ``vllm_eval.py``'s ``extract_scores`` including its fallback.
    """
    match = re.search(r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b", txt, re.IGNORECASE)
    if match:
        return {"score": float(match.group(1))}
    nums = re.findall(r"(?m)^\s*([01])\s*$", txt)
    if len(nums) == 1:
        return {"score": float(nums[0])}
    return {}


def build_evaluation_messages(prompt_data: dict, image_base64: str) -> list[dict]:
    """Multimodal chat messages for one WISE_Verified sample (reference port).

    ``prompt_data`` supplies ``Prompt``/``Explanation``; ``image_base64`` is the
    raw base64 payload (no data-URI prefix).
    """
    text = _USER_TEMPLATE.format(
        prompt=prompt_data["Prompt"], explanation=prompt_data["Explanation"]
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]},
    ]


def encode_image(image_path: str | Path) -> str:
    """Base64-encode an image file (reference ``encode_image``)."""
    return base64.b64encode(Path(image_path).read_bytes()).decode()


def _strip_think(text: str) -> str:
    """Drop `` thinking...</think>`` spans some judges emit (reference cleanup)."""
    text = re.sub(r" thinking.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>\s*", "", text)
    return text.strip()


class WiseJudge(Protocol):
    """Returns the judge's raw text for one sample (HTTP vLLM in production, faked in tests)."""

    def evaluate(self, prompt_data: dict, image_base64: str) -> str: ...


class VllmWiseJudge:
    """WISE_Verified judge over a vLLM OpenAI-compatible chat endpoint (stdlib only).

    No judge weights are loaded in-process -- a vLLM server must be running
    (``vllm serve Qwen/Qwen3.5-35B-A3B``).  The request mirrors ``vllm_eval.py``:
    ``temperature=0.0``, ``max_tokens=500``, ``chat_template_kwargs =
    {"enable_thinking": False}`` (raw HTTP, so the kwargs stay top-level).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 300.0,
        max_tokens: int = 500,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def evaluate(self, prompt_data: dict, image_base64: str) -> str:  # pragma: no cover - network
        payload = json.dumps({
            "model": self.model,
            "messages": build_evaluation_messages(prompt_data, image_base64),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:
            raise AqaError(
                f"wise: judge request to {self.base_url} failed: {exc} "
                f"(is `vllm serve {self.model}` running and reachable?)"
            ) from exc
        return _strip_think(body["choices"][0]["message"]["content"])


# ── aggregation (calculate_verified.py port) ─────────────────────────────────
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-category means + official weighted overall WiScore.

    Only rows with a parsed binary score contribute; failures are reported via
    ``num_missing``/``missing_prompt_ids`` (the reference drops them).  ``complete``
    is True when every one of the 1,000 official prompt ids produced a score.
    """
    scored = [r for r in rows if r.get("score") is not None]
    by_category: dict[str, list[float]] = {c: [] for c in PROMPT_ID_RANGES}
    for row in scored:
        # official category comes from the prompt_id range, not the row's label
        category = category_for_prompt_id(int(row["prompt_id"]))
        if category in by_category:
            by_category[category].append(float(row["score"]))
    category_means = {
        c: (sum(v) / len(v)) if v else 0.0 for c, v in by_category.items()
    }
    counts = {c: len(v) for c, v in by_category.items()}
    overall = sum(WISE_WEIGHTS[c] * category_means[c] for c in WISE_WEIGHTS)
    missing = [
        int(r["prompt_id"]) for r in rows if r.get("score") is None
    ]
    return {
        "overall": overall,
        "categories": category_means,
        "counts": counts,
        "num_images": len(scored),
        "num_scored": len(scored),
        "num_missing": len(missing),
        "missing_prompt_ids": sorted(missing),
        # A WISE judge that is unavailable raises AqaError immediately (see
        # ``score_image``), so a None score here is the judge answering
        # unparseably -- a scoring miss, not an environmental failure.
        "num_env_failed": 0,
        "complete": not missing and all(counts[c] > 0 for c in PROMPT_ID_RANGES),
    }


# ── scorer (harness-facing) ──────────────────────────────────────────────────
class WiseScorer:
    """Evaluate a flat generated-image dir against WISE_Verified prompt records."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 300.0,
        max_retries: int = MAX_EXTRACT_RETRIES,
        judge: WiseJudge | None = None,  # test seam: inject a fake judge
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.judge_model = judge_model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self._judge = judge

    def judge(self) -> WiseJudge:
        """The judge, constructed only when scoring actually runs."""
        if self._judge is None:
            self._judge = VllmWiseJudge(
                model=self.judge_model, base_url=self.base_url,
                api_key_env=self.api_key_env, timeout=self.timeout,
            )
        return self._judge

    def score_image(self, image_path: Any, record: dict, image_base64: str) -> float | None:
        """Judge one image; retries score extraction, returns None on failure (reference)."""
        judge = self.judge()
        prompt_data = {"Prompt": record["prompt"], "Explanation": record["explanation"]}
        for attempt in range(1, self.max_retries + 1):
            try:
                evaluation = judge.evaluate(prompt_data, image_base64)
            except AqaError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient judge/parse failure -> retry
                _log.warning("wise: judge attempt %d/%d failed for %s: %s",
                             attempt, self.max_retries, image_path, exc)
                continue
            scores = extract_scores(evaluation)
            if "score" in scores:
                return calculate_wiscore(scores["score"])
            _log.warning("wise: could not parse a score (attempt %d/%d) for %s",
                         attempt, self.max_retries, image_path)
        return None

    def evaluate_rows(
        self, rows: list[dict], records: list[dict], *, image_encoder=None,
    ) -> list[dict[str, Any]]:
        """Score every generated image row (``idx`` aligns with ``records``).

        ``image_encoder`` maps an image path to a raw base64 payload; defaults to
        :func:`encode_image`.  Rows whose judge call never yields a parseable
        score keep ``score=None`` and are excluded from :func:`summarize`.
        """
        encoder = image_encoder or encode_image
        by_idx = {int(r["idx"]): r for r in records}
        out: list[dict[str, Any]] = []
        for row in rows:
            idx = int(row["idx"])
            record = by_idx.get(idx)
            if record is None:
                _log.warning("wise: no prompt record for idx %d; skipped", idx)
                continue
            image_path = row["image_path"]
            score = self.score_image(image_path, record, encoder(image_path))
            prompt_id = int(record["prompt_id"])
            out.append({
                "filename": str(image_path),
                "idx": idx,
                "prompt_id": prompt_id,
                # official aggregation category = prompt_id range (the dataset's
                # `Category` is descriptive and must not drive the weights)
                "category": category_for_prompt_id(prompt_id),
                "source_category": record.get("category", ""),
                "subcategory": record.get("subcategory", ""),
                "prompt": record["prompt"],
                "score": score,
            })
        return out

    def write_results(self, rows: list[dict[str, Any]], out_prefix: Path) -> tuple[Path, Path]:
        """Write results.jsonl plus the official-style summary txt."""
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        jsonl = out_prefix.with_suffix(".jsonl")
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary = summarize(rows)
        txt = out_prefix.with_suffix(".txt")
        lines = [
            f"{cat}: {summary['categories'][cat]:.4f} (n={summary['counts'][cat]})"
            for cat in PROMPT_ID_RANGES
        ]
        lines.append(f"overall: {summary['overall']:.4f}")
        if summary["num_missing"]:
            lines.append(f"missing scores: {summary['num_missing']}")
        txt.write_text("\n".join(lines) + "\n")
        return jsonl, txt

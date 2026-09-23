"""DPG-Bench vLLM-judge tests -- hermetic (no GPU/network/judge weights).

Exercises the substituted judge end to end at the seams: the OpenAI-compatible
payload it sends (question text + base64 image), the yes/no normalization, the
official per-proposition/dependency aggregation, and config-based judge
selection.  ``urllib.request.urlopen`` is faked; the image is a tiny in-memory
PNG (PIL only encodes, no model is loaded).
"""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest

from aqa.eval.dpgbench import (
    DEFAULT_VLLM_JUDGE_MODEL,
    DpgBenchScorer,
    MPlugJudge,
    VllmVQAJudge,
    build_vqa_messages,
    encode_image,
    extract_yes_no,
)
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError


def _png():
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    return Image.open(BytesIO(buf.getvalue()))


class _Resp:
    def __init__(self, content):
        self._body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured_request(monkeypatch):
    """Fake urlopen capturing the Request; reply content is configurable."""
    state = {"content": "Yes"}

    def fake_urlopen(req, timeout=None):
        state["url"] = req.full_url
        state["data"] = req.data
        state["timeout"] = timeout
        state["headers"] = dict(req.header_items())
        return _Resp(state["content"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return state


# ── message building + normalization ─────────────────────────────────────────
def test_build_vqa_messages_carries_question_and_image():
    msgs = build_vqa_messages("Is there a cat?", "QUJD")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    text = msgs[1]["content"][0]["text"]
    assert "Is there a cat?" in text
    assert "yes" in text.lower() and "no" in text.lower()
    assert msgs[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_extract_yes_no_variants():
    assert extract_yes_no("Yes") == "yes"
    assert extract_yes_no("No.") == "no"
    assert extract_yes_no("yes, there is a cat") == "yes"
    assert extract_yes_no("NO") == "no"
    assert extract_yes_no("maybe") == "maybe"  # non-yes/no stays non-matching
    assert extract_yes_no(" thinkinghmm, hard one</think>No") == "no"


def test_encode_image_round_trips_png():
    payload = encode_image(_png())
    raw = base64.b64decode(payload)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


# ── HTTP judge ───────────────────────────────────────────────────────────────
def test_vllm_judge_sends_openai_payload_and_parses_answer(captured_request):
    judge = VllmVQAJudge(model="Qwen/Qwen2.5-VL-7B-Instruct",
                         base_url="http://127.0.0.1:8000/v1", timeout=12.0)
    answer = judge.vqa(_png(), "Is there a cat?")
    assert answer == "yes"
    assert captured_request["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured_request["timeout"] == 12.0
    payload = json.loads(captured_request["data"])
    assert payload["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == judge.max_tokens
    user_text = payload["messages"][1]["content"][0]["text"]
    assert "Is there a cat?" in user_text
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_vllm_judge_request_failure_raises_aqa_error(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(AqaError, match="VQA judge request"):
        VllmVQAJudge(base_url="http://127.0.0.1:9/v1").vqa(_png(), "q?")


# ── scoring semantics on top of the vLLM judge ───────────────────────────────
class _RecordingJudge:
    """VLLM-style judge: records (question, answer) and scripts yes/no."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.seen = []

    def vqa(self, image, question):
        self.seen.append(question)
        return self.answers.get(question, "yes")


def _record(idx):
    return {
        "idx": idx, "item_id": f"item{idx}", "prompt": "a red cat",
        "category": "entity - whole",
        "questions": [
            {"qid": 1, "question": "Is there a cat?", "answer": "entity (cat)",
             "dependency": [0], "category": "entity - whole"},
            {"qid": 2, "question": "Is it red?", "answer": "attribute (red)",
             "dependency": [1], "category": "attribute - color"},
        ],
    }


def _image_row(idx):
    return {"idx": str(idx), "prompt": "p", "image_path": f"/x/{idx}.png", "seed": "42"}


class _Img:
    def crop(self, box):
        return self


def test_scorer_aggregates_with_dependency_zeroing_via_judge():
    judge = _RecordingJudge({"Is it red?": "no"})
    scorer = DpgBenchScorer(judge=judge)
    rows = scorer.evaluate_rows([_image_row(0)], [_record(0)], image_loader=lambda p: _Img())
    # both official propositions were asked, verbatim
    assert judge.seen == ["Is there a cat?", "Is it red?"]
    # q2 (parent) answered no -> headline per-image score 0.5 (1/2), raw kept
    assert rows[0]["score"] == pytest.approx(0.5)
    assert rows[0]["raw_scores"] == {1: 1.0, 2: 0.0}
    assert rows[0]["adj_scores"] == {1: 1.0, 2: 0.0}


# ── config-based judge selection ─────────────────────────────────────────────
def test_scorer_judge_kind_selects_backend():
    mplug = DpgBenchScorer(judge_kind="mplug")
    assert isinstance(mplug.judge(), MPlugJudge)
    assert mplug.judge_label().startswith("mplug:")
    vllm = DpgBenchScorer(judge_kind="vllm", judge_model="Qwen/Qwen2.5-VL-7B-Instruct",
                          base_url="http://127.0.0.1:8000/v1")
    assert isinstance(vllm.judge(), VllmVQAJudge)
    assert vllm.judge_label() == "vllm:Qwen/Qwen2.5-VL-7B-Instruct@http://127.0.0.1:8000/v1"
    with pytest.raises(AqaError, match="unknown judge"):
        DpgBenchScorer(judge_kind="bogus")


def test_recipe_spec_exposes_vllm_judge_defaults():
    spec = BenchmarkRegistry().get("dpgbench").scoring.dpgbench
    assert spec.judge == "mplug"  # official default stays mPLUG
    assert spec.judge_model == DEFAULT_VLLM_JUDGE_MODEL
    assert spec.base_url.startswith("http://")


def test_harness_dispatches_vllm_judge_from_config(tmp_path, monkeypatch):
    good = {"filename": "0.png", "idx": 0, "item_id": "i0", "prompt": "p",
            "score": 1.0, "crop_scores": [1.0], "answers": {"1": "yes"},
            "raw_scores": {1: 1.0}, "adj_scores": {1: 1.0},
            "categories": {1: "entity - whole"}}
    monkeypatch.setattr(DpgBenchScorer, "evaluate_rows",
                        lambda self, rows, records, **kw: [good])
    recipe = BenchmarkRecipe(
        name="dpgbench", harness="diffusion", type="generative",
        dataset=str(tmp_path / "p.jsonl"), prompt_field="prompt",
        scoring={"scorer": "dpgbench", "dpgbench": {
            "judge": "vllm", "judge_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "base_url": "http://127.0.0.1:8000/v1",
        }},
    )
    result = DiffusionHarness()._result_for(
        recipe, "org/m", tmp_path / "out", [_image_row(0)], [_record(0)]
    )
    assert result.raw["judge"] == "vllm:Qwen/Qwen2.5-VL-7B-Instruct@http://127.0.0.1:8000/v1"
    assert "vllm:Qwen/Qwen2.5-VL-7B-Instruct" in result.command

"""Benchmark recipe schema, result types, and the benchmark registry.

A benchmark recipe is a declarative YAML file under ``recipes/benchmarks/``.  It
maps 1:1 onto NeMo Evaluator's ``@benchmark`` decorator fields (``dataset`` URI,
``prompt`` template, ``target_field``, a ``scorer`` chosen from the scorer
library) so upstream benchmark definitions port directly into YAML, with a ``.py``
scorer escape hatch for the long tail that needs code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..logging_util import RecipeError

_RECIPE_ROOT = Path(__file__).resolve().parents[2] / "recipes" / "benchmarks"


class CodeExecSpec(BaseModel):
    """Config for the ``code_exec`` scorer (LiveCodeBench / HumanEval / MBPP family).

    The model's completion is extracted, assembled with the sample's test code into
    one program, executed in a sandbox, and scored 1.0 iff it exits cleanly. Field
    names map onto the dataset columns (HumanEval defaults shown).
    """

    model_config = ConfigDict(extra="forbid")

    language: str = "python"
    interpreter: str = "python3"
    timeout_s: float = 30.0
    extract: bool = True  # pull the fenced ```python block out of the completion
    include_prompt: bool = False  # prepend the prompt/signature to the completion
    prompt_field: str = "prompt"
    test_field: str = "test"
    entry_point_field: str = "entry_point"
    check_template: str = "\ncheck({entry_point})\n"
    preamble: str | None = None  # code prepended to every program (imports, helpers)


class JudgeSpec(BaseModel):
    """Config for the ``llm_judge`` scorer (LLM-as-judge / rubric grading).

    The judge is any OpenAI-compatible chat endpoint. ``template`` is the user
    message rendered per sample with ``{question}``/``{prediction}``/``{target}``/
    ``{scale}`` fields; the judge is asked to emit ``SCORE: <0..scale>`` which we
    normalise to ``[0, 1]``. Set ``threshold`` to binarise (raw score >= threshold
    -> 1.0 else 0.0) for pass/fail style rubrics.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = "gpt-4o-mini"
    base_url: str | None = None  # defaults to OPENAI_BASE_URL env, else OpenAI
    api_key_env: str = "OPENAI_API_KEY"
    system: str | None = None
    template: str | None = None
    scale: float = 10.0
    threshold: float | None = None
    temperature: float = 0.0
    max_tokens: int = 512


class GenevalSpec(BaseModel):
    """Config for the ``geneval`` scorer (diffusion harness detection scoring).

    Model ids default to the official GenEval stack (Mask2Former swin COCO
    instance segmentation + CLIP ViT-L/14), run via transformers.
    """

    model_config = ConfigDict(extra="forbid")

    detector_model: str = "facebook/mask2former-swin-small-coco-instance"
    clip_model: str = "openai/clip-vit-large-patch14"


class DpgBenchSpec(BaseModel):
    """Config for the ``dpgbench`` scorer (dense-prompt-following via VQA judge).

    ``judge`` selects the VQA backend: ``mplug`` (default; the official
    mPLUG/modelscope judge, currently uninstallable on Python 3.12) or ``vllm``
    (an OpenAI-compatible vLLM VLM endpoint, the documented substitution --
    ``judge_model`` + ``base_url`` mirror the WISE judge config).  ``resolution``
    should match the generation size -- with pic_num=1 the top-left
    resolution-sized crop is judged.
    """

    model_config = ConfigDict(extra="forbid")

    judge: Literal["mplug", "vllm"] = "mplug"
    # mPLUG path.
    vqa_model: str = "damo/mplug_visual-question-answering_coco_large_en"
    resolution: int = 1024
    device: str = "auto"
    # vLLM path (judge == "vllm").
    judge_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 300.0
    max_tokens: int = 16


class Cvtg2kSpec(BaseModel):
    """Config for the ``cvtg2k`` scorer (complex visual text generation).

    Defaults follow the TextCrafter reference: PaddleOCR ``lang="en"`` for OCR
    and the official OpenAI CLIP ``ViT-L/14`` for CLIPScore.  ``clipscore_x2_5``
    defaults to the reference's 2.5x CLIPScore-paper scale so the number is
    comparable with the published anchors; pass ``False`` for the bare 0-1 cosine.
    """

    model_config = ConfigDict(extra="forbid")

    ocr_lang: str = "en"
    clip_model: str = "ViT-L/14"
    clipscore_x2_5: bool = True
    device: str = "auto"


class GlyphDraw2Spec(BaseModel):
    """Config for the ``glyphdraw2`` scorer (text-rendering Acc/NED via PP-OCR).

    Prompt records carry their own language, so ``ocr_lang`` is only the fallback
    for records missing one.  Acc/NED semantics are borrowed from AnyText (see
    ``aqa.eval.glyphdraw2``).
    """

    model_config = ConfigDict(extra="forbid")

    ocr_lang: str = "en"
    device: str = "auto"


class WiseSpec(BaseModel):
    """Config for the ``wise`` scorer (WISE_Verified binary 0/1 judge).

    The judge is ``Qwen/Qwen3.5-35B-A3B`` served by vLLM over an
    OpenAI-compatible endpoint (no weights loaded in-process); the API key is
    read from ``api_key_env``.  Only the Verified protocol is supported.
    """

    model_config = ConfigDict(extra="forbid")

    judge_model: str = "Qwen/Qwen3.5-35B-A3B"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 300.0
    max_retries: int = 3


class ImageMetricSpec(BaseModel):
    """Config for the ``image_metrics`` scorer (multi-metric DrawBench suite).

    One recipe emits one BenchmarkResult per requested metric
    (``<recipe>.clip`` / ``.clip_iqa`` / ``.imagereward``).  ``clip`` and
    ``clip_iqa`` share one CLIP checkpoint (transformers, in-process);
    ``imagereward`` runs in an isolated transformers-v4 subprocess env (see
    ``aqa.eval.image_metrics`` for the pinned combination and rationale).
    """

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: ["clip"])
    clip_model: str = "openai/clip-vit-base-patch32"
    imagereward_model: str = "ImageReward-v1.0"


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A built-in/registered scorer name, OR a "path.py:func" escape hatch.
    scorer: str | None = None
    # lm_eval harness reads the metric straight from its own results json.
    metric: str | None = None
    prefer: list[str] = Field(default_factory=list)
    # Config for scorer == "llm_judge" (ignored otherwise).
    judge: JudgeSpec | None = None
    # Config for scorer == "code_exec" (ignored otherwise).
    code_exec: CodeExecSpec | None = None
    # Config for scorer == "geneval" (ignored otherwise).
    geneval: GenevalSpec | None = None
    # Config for scorer == "dpgbench" (ignored otherwise).
    dpgbench: DpgBenchSpec | None = None
    # Config for scorer == "cvtg2k" (ignored otherwise).
    cvtg2k: Cvtg2kSpec | None = None
    # Config for scorer == "glyphdraw2" (ignored otherwise).
    glyphdraw2: GlyphDraw2Spec | None = None
    # Config for scorer == "wise" (ignored otherwise).
    wise: WiseSpec | None = None
    # Config for scorer == "image_metrics" (ignored otherwise).
    image_metrics: ImageMetricSpec | None = None


class BenchmarkRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    harness: str = "lm_eval"  # native | lm_eval | evalscope(plugin)
    type: Literal["generative", "loglikelihood", "multichoice", "qa"] = "loglikelihood"

    # lm_eval routing: the registered lm_eval task name (defaults to `name`).
    lm_eval_task: str | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    num_fewshot: int | None = None
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)
    batch_size: str | None = None  # per-task batch size override (else harness default)

    # native harness routing.
    dataset: str | None = None  # e.g. hf://openai/gsm8k?split=test
    prompt: str | None = None  # e.g. "{question}"
    target_field: str | None = None

    # diffusion harness routing: prompt column in `dataset` (HF uri or local file).
    prompt_field: str = "Prompts"
    # diffusion harness: images generated per prompt (GenEval officially uses 4;
    # the default keeps the one-image-per-prompt layout unchanged elsewhere).
    images_per_prompt: int = Field(default=1, ge=1)

    scoring: ScoringSpec = Field(default_factory=ScoringSpec)
    description: str = ""

    def task_name(self) -> str:
        return self.lm_eval_task or self.name


@dataclass
class BenchmarkResult:
    name: str
    score: float
    metric: str
    num_samples: int = 0
    per_sample: list[float] | None = None  # per-sample correctness (0/1) for paired stats
    raw: dict[str, Any] = field(default_factory=dict)
    command: str = ""  # the exact harness command that produced this result
    stats: Any = None  # optional TokenStats (agentic benchmarks); efficiency provenance
    turns: int = 0     # total agent turns spent (agentic benchmarks)

    @classmethod
    def from_agent_result(cls, agent_result: Any, *, name: str,
                          metric: str = "resolved_rate") -> BenchmarkResult:
        """Bridge an ``AgentEvalResult`` into a benchmark row.

        Carries the agentic score plus its efficiency provenance (token totals +
        turns), so an agentic benchmark folded into an :class:`EvalResult` lands
        in the reproducibility manifest's ``efficiency`` block automatically.
        """
        return cls(
            name=name,
            score=agent_result.mean_score(),
            metric=metric,
            num_samples=agent_result.num_scored(),
            per_sample=agent_result.per_sample() or None,
            stats=agent_result.token_totals(),
            turns=agent_result.total_turns(),
        )

    def summary_dict(self) -> dict[str, Any]:
        """Compact score entry for ``accuracy.json`` / the manifest.

        When the harness marked the benchmark incomplete (its raw carries
        ``complete: False`` -- e.g. some images failed to score because a judge
        was unavailable), that is surfaced here too so a comparison table cannot
        silently treat a partial run as a full measurement.
        """
        entry: dict[str, Any] = {
            "score": self.score,
            "metric": self.metric,
            "num_samples": self.num_samples,
        }
        if self.raw.get("complete") is False:
            entry["complete"] = False
            entry["num_missing"] = self.raw.get("num_missing", 0)
        return entry


@dataclass
class EvalResult:
    """Aggregated result of evaluating a model across one or more benchmarks."""

    model: str
    backend: str
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)  # exact eval commands executed
    healed: list[str] = field(default_factory=list)     # self-heal mutations applied

    def add(self, result: BenchmarkResult) -> None:
        self.benchmarks[result.name] = result
        cmd = getattr(result, "command", "")
        if cmd and cmd not in self.commands:
            self.commands.append(cmd)

    def mean_score(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(b.score for b in self.benchmarks.values()) / len(self.benchmarks)

    def efficiency(self) -> dict[str, Any] | None:
        """Token + turn totals across agentic benchmarks (``None`` if none carry any).

        Only benchmarks produced by an agent (which attach a ``stats``/``turns``)
        contribute; a pure lm_eval run has no such provenance and yields ``None``.
        """
        from ..agent_eval.stats import TokenStats

        tokens = TokenStats()
        total_turns = 0
        contributing = 0
        for b in self.benchmarks.values():
            if b.stats is not None:
                tokens.merge(b.stats)
            if b.turns:
                total_turns += b.turns
            if b.stats is not None or b.turns:
                contributing += 1
        if contributing == 0:
            return None
        eff: dict[str, Any] = {
            "num_benchmarks": contributing,
            "total_turns": total_turns,
        }
        if not tokens.is_empty():
            eff["tokens"] = tokens.to_dict()
        return eff

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "backend": self.backend,
            "mean_score": self.mean_score(),
            "results": {
                name: b.summary_dict()
                for name, b in self.benchmarks.items()
            },
            "commands": list(self.commands),
            "healed": list(self.healed),
        }
        eff = self.efficiency()
        if eff is not None:
            out["efficiency"] = eff
        return out


class BenchmarkRegistry:
    """Loads benchmark recipes from a directory (default: repo ``recipes/benchmarks``)."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else _RECIPE_ROOT
        self._cache: dict[str, BenchmarkRecipe] = {}

    def _load_file(self, path: Path) -> BenchmarkRecipe:
        data = yaml.safe_load(path.read_text()) or {}
        try:
            return BenchmarkRecipe.model_validate(data)
        except Exception as exc:
            raise RecipeError(f"invalid benchmark recipe {path}: {exc}") from exc

    def load_all(self) -> dict[str, BenchmarkRecipe]:
        if self._cache:
            return dict(self._cache)
        if not self.root.exists():
            return {}
        for path in sorted(self.root.glob("*.yaml")):
            recipe = self._load_file(path)
            self._cache[recipe.name] = recipe
        return dict(self._cache)

    def names(self) -> list[str]:
        return sorted(self.load_all())

    def get(self, name: str) -> BenchmarkRecipe:
        recipes = self.load_all()
        if name in recipes:
            return recipes[name]
        # Fall back to a bare lm_eval task not covered by a recipe file.
        return BenchmarkRecipe(name=name, harness="lm_eval")

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


class DpgBenchSpec(BaseModel):
    """Config for the ``dpgbench`` scorer (mPLUG VQA judge)."""

    model_config = ConfigDict(extra="forbid")

    vqa_model: str = "damo/mplug_visual-question-answering_coco_large_en"
    resolution: int = 1024
    device: str = "auto"


class ImageMetricsSpec(BaseModel):
    """Config for the ``image_metrics`` scorer (DrawBench CLIP/CLIP-IQA/ImageReward)."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: ["clip"])
    clip_model: str | None = None
    imagereward_model: str | None = None


class GenevalSpec(BaseModel):
    """Config for the ``geneval`` scorer (Mask2Former + CLIP object/color checks)."""

    model_config = ConfigDict(extra="forbid")

    detector_model: str | None = None
    clip_model: str | None = None


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
    # Config for scorer == "dpgbench" (mPLUG VQA judge; ignored otherwise).
    dpgbench: DpgBenchSpec | None = None
    # Config for scorer == "image_metrics" (DrawBench CLIP/CLIP-IQA/ImageReward).
    image_metrics: ImageMetricsSpec | None = None
    # Config for scorer == "geneval" (Mask2Former + CLIP object/color checks).
    geneval: GenevalSpec | None = None


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
    # diffusion harness routing.
    prompt_field: str | None = None  # dataset column holding the prompt
    images_per_prompt: int = 1

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


@dataclass
class EvalResult:
    """Aggregated result of evaluating a model across one or more benchmarks."""

    model: str
    backend: str
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)  # exact eval commands executed
    healed: list[str] = field(default_factory=list)     # self-heal mutations applied

    def add(self, result: BenchmarkResult | list[BenchmarkResult]) -> None:
        results = result if isinstance(result, list) else [result]
        for r in results:
            self.benchmarks[r.name] = r
            cmd = getattr(r, "command", "")
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
                name: {"score": b.score, "metric": b.metric, "num_samples": b.num_samples}
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

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


@dataclass
class EvalResult:
    """Aggregated result of evaluating a model across one or more benchmarks."""

    model: str
    backend: str
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)

    def add(self, result: BenchmarkResult) -> None:
        self.benchmarks[result.name] = result

    def mean_score(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(b.score for b in self.benchmarks.values()) / len(self.benchmarks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "mean_score": self.mean_score(),
            "results": {
                name: {"score": b.score, "metric": b.metric, "num_samples": b.num_samples}
                for name, b in self.benchmarks.items()
            },
        }


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

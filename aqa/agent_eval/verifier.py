"""Structured verdict + verifier protocol for agentic evaluation.

Where the base agentic loop scores a task with a single ``float`` reward, a
:class:`EvalVerdict` carries the *reward plus why*: a category (was this a model
error or an infra/parse failure?), the exact exception type, and a free-form
``metrics`` dict for multi-dimensional sub-scores (e.g. SWE-bench
``tests_passed`` / ``patch_applied``).

The point is a **clean denominator**: results whose category is not a real
model judgement (infra crash, harness parse failure, timeout) are excluded from
``mean_score`` so a quantized-vs-base accuracy delta is not polluted by flaky
containers.  A :class:`Verifier` is the injectable component that turns a
finished trajectory + environment into a verdict; when none is supplied the loop
falls back to the legacy scalar-reward behaviour (fully backward compatible).

This mirrors Harbor's verifier/reward-dict design (structured reward + dedicated
parse/infra exception classes) and reuses our own error taxonomy
(:func:`aqa.agent.drift.classify_error` / :func:`~aqa.agent.drift.exception_type`)
for the ``category`` / ``exc_type`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from .environment import Environment
    from .solver import Task

# --------------------------------------------------------------------------
# Verdict categories
# --------------------------------------------------------------------------
# A genuine model judgement (counts toward the score / denominator).
OK = "ok"                    # solved
MODEL_WRONG = "model_wrong"  # ran fine, model got it wrong (counts as a real 0)
# Not a model judgement -- excluded from the scored denominator.
INFRA_ERROR = "infra_error"  # sandbox/endpoint/environment blew up
PARSE_ERROR = "parse_error"  # could not parse the model action / grade the log
TIMEOUT = "timeout"          # turn budget / wall-clock exhausted (see policy)

# Categories that represent a real model judgement and therefore count.
SCORED_CATEGORIES: frozenset[str] = frozenset({OK, MODEL_WRONG})


class VerifierError(RuntimeError):
    """Base for verifier-side failures (translate to a non-scored verdict)."""


class InfraError(VerifierError):
    """Environment/endpoint failure -- not the model's fault (excluded)."""


class ParseError(VerifierError):
    """The response/grade could not be parsed -- excluded from the denominator."""


@dataclass
class EvalVerdict:
    """Structured result of grading one task.

    ``reward`` stays the primary scalar (so ``mean_score`` semantics are
    unchanged for scored samples); ``metrics`` holds optional sub-scores;
    ``category`` / ``exc_type`` explain *why*.
    """

    reward: float = 0.0
    solved: bool = False
    category: str = MODEL_WRONG
    exc_type: str = ""
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.reward)

    @property
    def is_scored(self) -> bool:
        """True when this verdict is a real model judgement (counts in mean)."""
        return self.category in SCORED_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "solved": self.solved,
            "category": self.category,
            "exc_type": self.exc_type,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_reward(cls, reward: float, *, solved: bool | None = None) -> EvalVerdict:
        """Wrap a legacy scalar reward as a scored verdict."""
        s = (reward > 0.0) if solved is None else solved
        return cls(reward=float(reward), solved=s,
                   category=OK if s else MODEL_WRONG)


def coerce_verdict(value: Any, *, solved: bool | None = None) -> EvalVerdict:
    """Coerce a verifier/score return value into an :class:`EvalVerdict`.

    Accepts an existing verdict (passed through), a bool, or a numeric reward.
    """
    if isinstance(value, EvalVerdict):
        return value
    if isinstance(value, bool):
        return EvalVerdict.from_reward(1.0 if value else 0.0, solved=value)
    return EvalVerdict.from_reward(float(value), solved=solved)


def verdict_from_exception(exc: BaseException, *, phase: str = "evaluate") -> EvalVerdict:
    """Translate a raised exception into a non-scored verdict.

    A dedicated :class:`ParseError` maps to ``parse_error``; everything else is
    treated as infra.  ``exc_type`` / ``detail`` reuse the shared taxonomy so the
    failure can be grouped just like fix-loop errors.
    """
    from ..agent.drift import classify_error, exception_type

    text = f"{type(exc).__name__}: {exc}"
    category = PARSE_ERROR if isinstance(exc, ParseError) else INFRA_ERROR
    return EvalVerdict(
        reward=0.0, solved=False, category=category,
        exc_type=exception_type(text) or type(exc).__name__,
        detail=f"{classify_error(text)}: {text}"[:500],
    )


@runtime_checkable
class Verifier(Protocol):
    """Turns a finished trajectory + environment into a structured verdict."""

    async def verify(self, task: Task, trajectory: Any, env: Environment) -> EvalVerdict:
        ...

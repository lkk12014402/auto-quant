"""Pluggable phase hooks (capability components).

A :class:`PhaseHook` is a named object with per-phase methods
(``after_quantize`` / ``after_benchmark``, ...).  Hooks are where
cross-cutting *mechanism* capabilities plug in -- validation, taxonomy,
trajectory capture, soft-fail policies -- so a new capability ships as a hook
registration instead of pipeline surgery.  (Judgment capabilities keep living
in the skill packages: ``skills/``, fix playbooks.)

Semantics:

* a hook raises :class:`~aqa.logging_util.AqaError` to deliberately FAIL the
  phase (validator semantics -- the error feeds the fix-loop's signature);
* any other exception inside a hook is caught, logged at WARNING, and never
  crashes the run;
* hooks run in configured order; every invocation is logged at DEBUG.

Registration mirrors the other extension points: the ``aqa.hooks``
entry-point group (pyproject) plus the builtin fallback map in
``aqa/registry.py``.  The active chain comes from ``run_cfg.hooks``:
``None`` => the built-in defaults, an explicit list => exactly those,
``[]`` => all hooks off.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..logging_util import AqaError, get_logger
from ..registry import HOOKS

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from ..config import QuantConfig, RunConfig
    from ..eval.registry import BenchmarkRecipe

_log = get_logger("hooks")

# Hooks active when the run config does not name a chain explicitly.
DEFAULT_HOOKS = ("quant_sanity", "eval_sanity")


@runtime_checkable
class PhaseHook(Protocol):
    """A capability component invoked at phase boundaries.

    Implement only the phases you care about; the runner skips phases a hook
    does not define.  Raise ``AqaError`` to fail the phase on purpose.
    """

    name: str

    def after_quantize(
        self, run_cfg: RunConfig | None, quant_cfg: QuantConfig | None,
        output_dir: Path,
    ) -> None:
        """Runs after quantization + post-export repair, before eval."""
        ...

    def after_benchmark(
        self, run_cfg: RunConfig | None, recipe: BenchmarkRecipe,
        result_or_exception, out_dir: Path,
    ) -> None:
        """Runs per benchmark result as the evaluate phase assembles it."""
        ...


@lru_cache(maxsize=8)
def _resolve(names: tuple[str, ...]) -> tuple[PhaseHook, ...]:
    return tuple(HOOKS.create(name) for name in names)


def resolve_hooks(run_cfg: RunConfig) -> tuple[PhaseHook, ...]:
    """The run's configured hook chain (resolved once per distinct chain)."""
    names = DEFAULT_HOOKS if run_cfg.hooks is None else tuple(run_cfg.hooks)
    return _resolve(tuple(names))


def run_hook_chain(
    phase: str, chain, run_cfg: RunConfig | None = None, **ctx
) -> None:
    """Execute ``phase`` on every hook in ``chain`` that defines it, in order.

    A hook's ``AqaError`` propagates (validator verdict); any other exception
    is logged at WARNING and skipped.
    """
    for hook in chain or ():
        fn = getattr(hook, phase, None)
        if fn is None:
            continue
        _log.debug("hook %s.%s", getattr(hook, "name", hook), phase)
        try:
            fn(run_cfg, **ctx)
        except AqaError:
            raise
        except Exception as exc:  # noqa: BLE001 - hooks never crash a run
            _log.warning("hook %s.%s failed (ignored): %s",
                         getattr(hook, "name", hook), phase, exc)


def run_hooks(phase: str, run_cfg: RunConfig, **ctx) -> None:
    """Resolve the run's chain and execute ``phase`` on it (see run_hook_chain)."""
    run_hook_chain(phase, resolve_hooks(run_cfg), run_cfg, **ctx)

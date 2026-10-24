"""Evaluation orchestrator: run a set of benchmarks and aggregate results.

Routes each benchmark to its harness (native / lm_eval / plugin) per the recipe,
constructs the requested infra backend, and produces one :class:`EvalResult`.
Benchmarks that share generation parameters are merged into a single ``lm_eval``
call (e.g. piqa+hellaswag+mmlu), while tasks needing distinct params (gsm8k) run
separately -- one model load per group instead of one per task.
"""

from __future__ import annotations

from pathlib import Path

from ..config import EvalConfig
from ..logging_util import get_logger
from ..registry import BACKENDS, HARNESSES
from .registry import BenchmarkRecipe, BenchmarkRegistry, EvalResult
from .thinking import apply_mode_to_cfg, resolve_thinking_modes

_log = get_logger("eval.runner")


def _make_harness(name: str, cfg: EvalConfig, num_gpus: int):
    if name == "lm_eval":
        opts = cfg.lm_eval
        return HARNESSES.create(
            name,
            num_gpus=num_gpus,
            batch_size=cfg.batch_size,
            limit=cfg.limit,
            gen_max_toks=cfg.gen_max_toks,
            model_args=opts.model_args,
            metadata=opts.metadata,
            gen_kwargs=opts.gen_kwargs,
            seed=opts.seed,
            apply_chat_template=opts.apply_chat_template,
            fewshot_as_multiturn=opts.fewshot_as_multiturn,
        )
    if name == "native":
        return HARNESSES.create(name, limit=cfg.limit)
    if name == "evalscope":
        return HARNESSES.create(name, **cfg.evalscope)
    if name == "diffusion":
        return HARNESSES.create(name, **cfg.diffusion.model_dump())
    return HARNESSES.create(name)


def _resolve_recipe(bench_name: str, cfg: EvalConfig, registry: BenchmarkRegistry) -> BenchmarkRecipe:
    recipe = registry.get(bench_name)
    # A bare task (no recipe file) inherits the eval-config default harness.
    if bench_name not in registry.load_all() and cfg.harness != recipe.harness:
        recipe = recipe.model_copy(update={"harness": cfg.harness})
    # gsm8k-style tasks flagged generative at the eval-config level.
    if bench_name in cfg.generative_tasks and recipe.type != "generative":
        recipe = recipe.model_copy(update={"type": "generative", "apply_chat_template": True})
    return recipe


def _warn_generative_thinking_risk(model: str, cfg: EvalConfig, mode: str) -> None:
    """Fail-loud guard: a thinking-capable model evaluated with thinking ON while
    a strict-match generative task (e.g. gsm8k) is in the set silently tanks that
    score on ``<think>`` traces.  Warn so the risk is visible in the console /
    pipeline log up front, instead of being discovered as a mysterious ~0 later.
    """
    if mode != "on":
        return
    gen_in_set = sorted(set(cfg.generative_tasks) & set(cfg.benchmarks))
    if not gen_in_set:
        return
    try:
        from .thinking import detect_thinking_capable
        if not detect_thinking_capable(model):
            return
    except Exception:  # noqa: BLE001 - detection must never crash a run
        return
    _log.warning(
        "thinking RISK: %s is thinking-capable but evaluated with thinking ON while "
        "generative tasks %s are in the eval set -- strict-match will be distorted by "
        "<think> traces. Set eval.thinking: off (and thinking_off_model_args."
        "reasoning_parser) for a standard-benchmark口径.",
        model, gen_in_set,
    )


def run_evaluation(
    model: str,
    cfg: EvalConfig,
    out_dir: Path,
    *,
    num_gpus: int = 1,
    registry: BenchmarkRegistry | None = None,
) -> EvalResult:
    """Evaluate ``model`` on ``cfg.benchmarks`` and return aggregated results."""
    registry = registry or BenchmarkRegistry()
    # Resolve the reasoning mode and fold its model_args into cfg before any
    # harness is built. ``off`` injects ``enable_thinking=False`` so generative
    # tasks score on a bare answer; dual-mode (both/auto-capable) is measured one
    # mode per run_evaluation call, so warn and take the first (``on``).
    modes = resolve_thinking_modes(cfg, model)
    if len(modes) > 1:
        _log.warning(
            "eval.thinking=%s selects dual-mode (%s); run_evaluation measures one "
            "mode, evaluating '%s' only -- set eval.thinking: on|off to silence",
            cfg.thinking, "+".join(modes), modes[0],
        )
    cfg = apply_mode_to_cfg(cfg, modes[0])
    _warn_generative_thinking_risk(model, cfg, modes[0])
    backend = BACKENDS.create(cfg.backend)
    result = EvalResult(model=model, backend=cfg.backend)
    harness_cache: dict[str, object] = {}

    # Preserve request order while collecting each benchmark's resolved recipe.
    recipes = [(name, _resolve_recipe(name, cfg, registry)) for name in cfg.benchmarks]

    def _harness(name: str):
        h = harness_cache.get(name)
        if h is None:
            h = _make_harness(name, cfg, num_gpus)
            harness_cache[name] = h
        return h

    # Partition into (harness, group_key) buckets; lm_eval with group=True merges
    # tasks that share generation params into one invocation.
    buckets: list[tuple[str, tuple, list[BenchmarkRecipe]]] = []
    index: dict[tuple, int] = {}
    for _name, recipe in recipes:
        harness = _harness(recipe.harness)
        can_group = (
            recipe.harness == "lm_eval"
            and getattr(cfg.lm_eval, "group", True)
            and hasattr(harness, "group_key")
        )
        key = (recipe.harness, harness.group_key(recipe)) if can_group else (recipe.harness, id(recipe))
        if key in index:
            buckets[index[key]][2].append(recipe)
        else:
            index[key] = len(buckets)
            buckets.append((recipe.harness, key, [recipe]))

    for harness_name, _key, group in buckets:
        harness = _harness(harness_name)
        names = [r.name for r in group]
        if not harness.available():
            _log.warning("harness %s unavailable; skipping %s", harness_name, ",".join(names))
            continue
        tag = names[0] if len(names) == 1 else "_".join(names)
        _log.info("evaluating %s via harness=%s backend=%s", ",".join(names), harness_name, cfg.backend)
        if len(group) > 1 and hasattr(harness, "run_group"):
            for bench_result in harness.run_group(group, model, backend, out_dir / tag):
                result.add(bench_result)
                _log.info("  %s => %.4f (%s)", bench_result.name, bench_result.score, bench_result.metric)
        else:
            recipe = group[0]
            bench_result = harness.run(recipe, model, backend, out_dir / recipe.name)
            result.add(bench_result)
            for br in (bench_result if isinstance(bench_result, list) else [bench_result]):
                _log.info("  %s => %.4f (%s)", br.name, br.score, br.metric)

    return result

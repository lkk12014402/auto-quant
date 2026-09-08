"""Plugin registries backed by ``importlib.metadata`` entry points.

Every extension point (executor / harness / backend / sink / recipe-store /
scorer) is a named entry-point group declared in ``pyproject.toml``.  Third
parties add an implementation by publishing a pip package that declares the same
group -- no fork of the core repo required (the open-source extension story).

A small builtin fallback map is kept so the package works from a source checkout
that has not been ``pip install``-ed yet (entry points are only discoverable
after install).
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

from .logging_util import RegistryError, get_logger

_log = get_logger("registry")

# Builtin fallbacks: group -> {name: "module:attr"}.  Mirrors pyproject entry
# points so `python -m aqa` works from a bare source tree.
_BUILTINS: dict[str, dict[str, str]] = {
    "aqa.providers": {
        "pretrained": "aqa.providers.builtin:PretrainedProvider",
        "quantize": "aqa.providers.builtin:QuantizeProvider",
        "endpoint": "aqa.providers.builtin:EndpointProvider",
    },
    "aqa.executors": {
        "local": "aqa.executors.local:LocalExecutor",
        "ssh": "aqa.executors.ssh:SshExecutor",
    },
    "aqa.harnesses": {
        "native": "aqa.eval.harness.native:NativeHarness",
        "lm_eval": "aqa.eval.harness.lm_eval:LmEvalHarness",
        "diffusion": "aqa.eval.harness.diffusion:DiffusionHarness",
        "evalscope": "aqa.eval.harness.evalscope:EvalScopeHarness",
    },
    "aqa.backends": {
        "hf": "aqa.backends.hf:HfBackend",
        "vllm": "aqa.backends.vllm:VllmBackend",
        "vllm_omni": "aqa.backends.vllm_omni:VllmOmniBackend",
    },
    "aqa.sinks": {
        "local_fs": "aqa.report.sinks:LocalFsSink",
        "hf_dataset": "aqa.report.sinks:HfDatasetSink",
        "hf_model": "aqa.report.sinks:HfModelSink",
    },
    "aqa.sandboxes": {
        "local": "aqa.sandbox.local:LocalSandbox",
        "docker": "aqa.sandbox.docker:DockerSandbox",
    },
    "aqa.recipe_stores": {
        "repo_dir": "aqa.recipe_store:RepoDirStore",
        "hf_dataset": "aqa.recipe_store:HfDatasetStore",
    },
    "aqa.scorers": {
        "exact_match": "aqa.eval.scoring:exact_match",
        "numeric_match": "aqa.eval.scoring:numeric_match",
        "multichoice_regex": "aqa.eval.scoring:multichoice_regex",
        "fuzzy_match": "aqa.eval.scoring:fuzzy_match",
        "answer_line": "aqa.eval.scoring:answer_line",
    },
    "aqa.hooks": {
        "quant_sanity": "aqa.hooks.sanity:QuantSanityHook",
        "eval_sanity": "aqa.hooks.sanity:EvalSanityHook",
    },
}


def _load_target(target: str) -> Any:
    """Import a ``module:attr`` (or ``module.attr``) target string."""
    if ":" in target:
        module_name, _, attr = target.partition(":")
    else:
        module_name, _, attr = target.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class Registry:
    """A named plugin group resolved from entry points + builtin fallbacks."""

    def __init__(self, group: str):
        self.group = group
        self._extra: dict[str, str] = {}  # runtime registrations (tests / dynamic plugins)

    def register(self, name: str, target: str) -> None:
        """Register (or override) ``name`` -> ``module:attr`` at runtime."""
        self._extra[name] = target

    def _discover(self) -> dict[str, str]:
        table: dict[str, str] = dict(_BUILTINS.get(self.group, {}))
        try:
            eps = metadata.entry_points(group=self.group)
        except TypeError:  # pragma: no cover - very old importlib.metadata
            eps = metadata.entry_points().get(self.group, [])  # type: ignore[attr-defined]
        for ep in eps:
            table[ep.name] = ep.value
        table.update(self._extra)
        return table

    def names(self) -> list[str]:
        return sorted(self._discover())

    def get(self, name: str) -> Any:
        table = self._discover()
        if name not in table:
            raise RegistryError(
                f"unknown {self.group} plugin {name!r}; available: {', '.join(sorted(table)) or '<none>'}"
            )
        return _load_target(table[name])

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Resolve ``name`` and instantiate it (for class-based plugins)."""
        return self.get(name)(*args, **kwargs)


EXECUTORS = Registry("aqa.executors")
HARNESSES = Registry("aqa.harnesses")
BACKENDS = Registry("aqa.backends")
SINKS = Registry("aqa.sinks")
SANDBOXES = Registry("aqa.sandboxes")
RECIPE_STORES = Registry("aqa.recipe_stores")
SCORERS = Registry("aqa.scorers")
PROVIDERS = Registry("aqa.providers")
HOOKS = Registry("aqa.hooks")

_ALL = {
    "providers": PROVIDERS,
    "executors": EXECUTORS,
    "harnesses": HARNESSES,
    "backends": BACKENDS,
    "sinks": SINKS,
    "sandboxes": SANDBOXES,
    "recipe_stores": RECIPE_STORES,
    "scorers": SCORERS,
    "hooks": HOOKS,
}


def all_registries() -> dict[str, Registry]:
    return dict(_ALL)

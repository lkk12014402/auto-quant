"""Recipe stores: pluggable source/sink for quant recipes (pin-as-artifact).

``repo_dir`` reads/writes ``recipes/quant/`` inside the repo (versioned, git);
``hf_dataset`` (stub) can pull/push recipes alongside results.  The agent-optimize
flow ``pin``s its proposed recipe here so later runs read a validated recipe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .logging_util import AqaError

_QUANT_ROOT = Path(__file__).resolve().parent.parent / "recipes" / "quant"


@runtime_checkable
class RecipeStore(Protocol):
    name: str

    def load(self, key: str) -> dict[str, Any] | None:
        ...

    def pin(self, key: str, recipe: dict[str, Any]) -> str:
        ...


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


class RepoDirStore:
    name = "repo_dir"

    def __init__(self, root: str | None = None, **_: Any):
        self.root = Path(root) if root else _QUANT_ROOT

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{_safe_key(key)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def pin(self, key: str, recipe: dict[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_safe_key(key)}.json"
        path.write_text(json.dumps(recipe, indent=2))
        return str(path)


class HfDatasetStore:
    name = "hf_dataset"

    def __init__(self, repo: str | None = None, **_: Any):
        self.repo = repo

    def load(self, key: str) -> dict[str, Any] | None:  # pragma: no cover - network
        raise AqaError("HfDatasetStore.load is not implemented in this build")

    def pin(self, key: str, recipe: dict[str, Any]) -> str:  # pragma: no cover - network
        raise AqaError("HfDatasetStore.pin is not implemented in this build")

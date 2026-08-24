"""SWE-bench result records + report writing.

Emits the two artifacts the SWE-bench ecosystem expects:

* ``predictions.jsonl`` — one row per instance in the *official* shape
  (``instance_id``, ``model_name_or_path``, ``model_patch``), so the output can be
  fed straight into the upstream ``swebench`` harness for an independent re-grade.
* ``report.json`` — an aggregate summary (resolved count / rate, per-instance
  detail) plus the resolved-id list, matching ``swebench``'s report layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..agent_eval.stats import TokenStats
    from .spec import ResolutionReport


@dataclass
class InstanceResult:
    """Outcome of evaluating one SWE-bench instance."""

    instance_id: str
    resolved: bool = False
    model_patch: str = ""
    report: ResolutionReport | None = None
    error: str | None = None
    stats: TokenStats | None = None
    turns: int = 0

    def prediction(self, model_name: str) -> dict[str, Any]:
        """The official prediction row for this instance."""
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": model_name,
            "model_patch": self.model_patch,
        }


def write_predictions(path: str | Path, results: list[InstanceResult], model_name: str) -> Path:
    """Write ``predictions.jsonl`` (upstream-harness compatible)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.prediction(model_name), ensure_ascii=False) + "\n")
    return p


def build_report(results: list[InstanceResult], model_name: str) -> dict[str, Any]:
    """Aggregate a resolved-rate report (mirrors ``swebench``'s report shape)."""
    total = len(results)
    completed = [r for r in results if r.error is None]
    resolved = [r for r in results if r.resolved]
    errored = [r for r in results if r.error is not None]
    efficiency = _efficiency(results)
    report: dict[str, Any] = {
        "model_name_or_path": model_name,
        "total_instances": total,
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(completed) - len(resolved),
        "error_instances": len(errored),
        "resolved_rate": (len(resolved) / total) if total else 0.0,
        "resolved_ids": sorted(r.instance_id for r in resolved),
        "error_ids": sorted(r.instance_id for r in errored),
        "per_instance": {
            r.instance_id: {
                "resolved": r.resolved,
                "error": r.error,
                **({"turns": r.turns} if r.turns else {}),
                **({"tokens": r.stats.to_dict()}
                   if r.stats is not None and not r.stats.is_empty() else {}),
                **(r.report.to_dict() if r.report is not None else {}),
            }
            for r in results
        },
    }
    if efficiency is not None:
        report["efficiency"] = efficiency
    return report


def _efficiency(results: list[InstanceResult]) -> dict[str, Any] | None:
    """Aggregate token + turn efficiency across instances (None when unreported)."""
    from ..agent_eval.stats import TokenStats

    tokens = TokenStats()
    total_turns = 0
    for r in results:
        tokens.merge(r.stats)
        total_turns += r.turns
    if tokens.is_empty() and total_turns == 0:
        return None
    n = len(results) or 1
    eff: dict[str, Any] = {"num_instances": len(results),
                           "total_turns": total_turns,
                           "avg_turns": round(total_turns / n, 2)}
    if not tokens.is_empty():
        eff["tokens"] = tokens.to_dict()
    return eff


def write_report(path: str | Path, results: list[InstanceResult], model_name: str) -> Path:
    """Write ``report.json`` and return its path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(build_report(results, model_name), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


@dataclass
class SWEBenchRunReport:
    """In-memory summary returned by the runner."""

    model_name: str
    results: list[InstanceResult] = field(default_factory=list)

    @property
    def resolved_rate(self) -> float:
        return (sum(1 for r in self.results if r.resolved) / len(self.results)
                if self.results else 0.0)

    def summary(self) -> dict[str, Any]:
        return build_report(self.results, self.model_name)

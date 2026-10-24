"""Reporting helpers: aggregate an EvalResult (+ optional baseline) and gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import GateConfig
from ..eval.registry import EvalResult
from .stats import compare_scores


@dataclass
class GateDecision:
    verdict: str  # GO | NO-GO | INCONCLUSIVE
    reasons: list[str]


def write_accuracy_json(result: EvalResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "accuracy.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def read_eval_result(out_dir: Path) -> EvalResult:
    """Reconstruct an :class:`EvalResult` from a prior ``accuracy.json``.

    Used by resume to reuse a completed eval/baseline without re-running it. Only
    the gate-/manifest-relevant fields are restored (scores, metrics, sample
    counts, commands, healed list); agentic efficiency provenance — ``stats`` /
    ``turns`` — is not persisted in ``accuracy.json`` and is intentionally lost on
    a resumed run.
    """
    from ..eval.registry import BenchmarkResult

    path = Path(out_dir) / "accuracy.json"
    data = json.loads(path.read_text())
    result = EvalResult(model=data.get("model", ""), backend=data.get("backend", ""),
                        healed=list(data.get("healed", [])))
    for name, row in (data.get("results") or {}).items():
        result.benchmarks[name] = BenchmarkResult(
            name=name, score=float(row.get("score", 0.0)),
            metric=row.get("metric", ""), num_samples=int(row.get("num_samples", 0)))
    result.commands = list(data.get("commands", []))
    return result


def evaluate_gate(
    quantized: EvalResult, baseline: EvalResult | None, gate: GateConfig
) -> GateDecision:
    """GO/NO-GO decision comparing quantized vs baseline (NeMo `nel gate` style)."""
    reasons: list[str] = []
    verdict = "GO"
    if gate.min_accuracy is not None:
        mean = quantized.mean_score()
        if mean < gate.min_accuracy:
            verdict = "NO-GO"
            reasons.append(f"mean accuracy {mean:.4f} < min_accuracy {gate.min_accuracy}")
    if gate.max_accuracy_drop is not None:
        if baseline is None:
            return GateDecision("INCONCLUSIVE", ["max_accuracy_drop set but no baseline available"])
        for name, q in quantized.benchmarks.items():
            b = baseline.benchmarks.get(name)
            if b is None:
                continue
            drop = b.score - q.score
            if drop <= gate.max_accuracy_drop:
                continue
            # The drop exceeds the threshold. When significance testing is on, only
            # fail if the drop is also statistically significant -- a drop within
            # sampling noise is INCONCLUSIVE, not a hard NO-GO.
            if gate.significance:
                sig = compare_scores(
                    base_score=b.score, quant_score=q.score,
                    n=max(b.num_samples, q.num_samples),
                    base_correct=b.per_sample, quant_correct=q.per_sample,
                    alpha=gate.alpha,
                )
                if not sig.significant:
                    if verdict == "GO":
                        verdict = "INCONCLUSIVE"
                    reasons.append(
                        f"{name}: drop {drop:.4f} > {gate.max_accuracy_drop} but NOT "
                        f"significant (p={sig.p_value:.3g}, {sig.test}; {sig.detail})"
                    )
                    continue
                verdict = "NO-GO"
                reasons.append(
                    f"{name}: drop {drop:.4f} > {gate.max_accuracy_drop} and significant "
                    f"(p={sig.p_value:.3g}, {sig.test})"
                )
            else:
                verdict = "NO-GO"
                reasons.append(
                    f"{name}: drop {drop:.4f} > max_accuracy_drop {gate.max_accuracy_drop}"
                )
    if not reasons:
        reasons.append("all gate checks passed")
    return GateDecision(verdict, reasons)

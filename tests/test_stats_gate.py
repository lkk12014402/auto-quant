"""Tests for statistical significance tests and the significance-aware gate."""

from __future__ import annotations

from aqa.config import GateConfig
from aqa.eval.registry import BenchmarkResult, EvalResult
from aqa.report import evaluate_gate
from aqa.report.stats import (
    compare_scores,
    discordant_counts,
    mcnemar_test,
    paired_permutation_test,
    sign_test,
    two_proportion_ztest,
    wilson_interval,
)

# ── primitives ───────────────────────────────────────────────────────────────


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_no_discordance_is_one():
    assert mcnemar_test(0, 0) == 1.0


def test_mcnemar_strong_asymmetry_significant():
    # 20 vs 0 discordant -> clearly significant.
    assert mcnemar_test(20, 0) < 0.01


def test_mcnemar_symmetric_not_significant():
    assert mcnemar_test(10, 10) > 0.5


def test_mcnemar_large_sample_uses_chisquare():
    p = mcnemar_test(40, 10)  # n=50 > 25 -> chi-square path
    assert 0.0 < p < 0.05


def test_discordant_counts():
    base = [1, 1, 0, 1, 0]
    quant = [1, 0, 0, 0, 1]
    b, c = discordant_counts(base, quant)
    assert (b, c) == (2, 1)


def test_sign_test():
    assert sign_test([1, 1, 1, 1, 1]) < 0.1
    assert sign_test([0, 0, 0]) == 1.0


def test_two_proportion_ztest():
    assert two_proportion_ztest(90, 100, 60, 100) < 0.01  # big gap
    assert two_proportion_ztest(50, 100, 51, 100) > 0.5   # tiny gap


def test_paired_permutation_test():
    base = [1.0] * 20
    quant = [0.0] * 20
    assert paired_permutation_test(base, quant, iterations=2000) < 0.05
    same = [1.0, 0.0, 1.0, 0.0]
    assert paired_permutation_test(same, same, iterations=2000) == 1.0


# ── compare_scores dispatch ──────────────────────────────────────────────────


def test_compare_scores_paired_uses_mcnemar():
    base = [1] * 18 + [0] * 2
    quant = [0] * 18 + [1] * 2  # big discordance
    r = compare_scores(base_score=0.9, quant_score=0.1, n=20,
                       base_correct=base, quant_correct=quant)
    assert r.test == "mcnemar" and r.significant


def test_compare_scores_unpaired_uses_ztest():
    r = compare_scores(base_score=0.9, quant_score=0.6, n=100)
    assert r.test == "two_proportion" and r.significant


# ── significance-aware gate ──────────────────────────────────────────────────


def _result(name: str, score: float, per_sample: list[float] | None = None) -> EvalResult:
    ev = EvalResult(model="m", backend="hf")
    n = len(per_sample) if per_sample else 100
    ev.add(BenchmarkResult(name=name, score=score, metric="acc", num_samples=n,
                           per_sample=per_sample))
    return ev


def test_gate_significant_drop_is_nogo():
    base = _result("piqa", 0.90, [1.0] * 90 + [0.0] * 10)
    quant = _result("piqa", 0.60, [1.0] * 60 + [0.0] * 40)
    gate = GateConfig(max_accuracy_drop=0.05, significance=True)
    d = evaluate_gate(quant, base, gate)
    assert d.verdict == "NO-GO"


def test_gate_insignificant_drop_is_inconclusive():
    # Drop of 0.02 on tiny n -> exceeds a 0.01 threshold but not significant.
    base = _result("piqa", 0.52, [1.0] * 13 + [0.0] * 12)
    quant = _result("piqa", 0.48, [1.0] * 12 + [0.0] * 13)
    gate = GateConfig(max_accuracy_drop=0.01, significance=True)
    d = evaluate_gate(quant, base, gate)
    assert d.verdict == "INCONCLUSIVE"
    assert any("NOT significant" in r for r in d.reasons)


def test_gate_without_significance_still_hard_fails():
    base = _result("piqa", 0.52)
    quant = _result("piqa", 0.50)
    gate = GateConfig(max_accuracy_drop=0.01, significance=False)
    d = evaluate_gate(quant, base, gate)
    assert d.verdict == "NO-GO"

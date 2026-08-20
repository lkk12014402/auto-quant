"""Statistical significance tests for the quantization accuracy gate.

Dependency-free (``math`` only) ports of the significance machinery from NeMo
Evaluator's ``metrics/paired_tests.py`` + ``confidence.py``.  The gate uses these
to decide whether a quantized model's accuracy *drop* is real signal or sampling
noise, instead of tripping on any drop above a fixed threshold.

Two regimes:
  * **Paired** (per-sample correctness available for both models): McNemar's test
    (exact binomial for small discordant counts, chi-square with continuity
    correction otherwise) and a paired sign / permutation test.
  * **Unpaired** (only aggregate score + n): two-proportion z-test, and Wilson
    score confidence intervals per model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_normal_p(z: float) -> float:
    """Two-sided p-value for a z statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for observing ``k`` of ``n`` (H0: prob ``p``)."""
    if n == 0:
        return 1.0
    from math import comb

    def pmf(i: int) -> float:
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    observed = pmf(k)
    # Sum probabilities of outcomes at least as extreme (<= observed pmf), with a
    # small epsilon for float wobble -- the standard two-sided binomial construction.
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed * (1 + 1e-9))
    return min(1.0, total)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion ``k/n`` (default 95%)."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_test(b: int, c: int) -> float:
    """McNemar's paired test p-value from discordant counts.

    ``b`` = baseline-correct & quantized-wrong, ``c`` = baseline-wrong &
    quantized-correct.  Exact binomial when ``b + c`` is small; chi-square with
    continuity correction otherwise.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n <= 25:
        return _binom_two_sided_p(min(b, c), n, 0.5)
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with 1 df == erfc(sqrt(chi2/2)).
    return math.erfc(math.sqrt(chi2 / 2.0))


def discordant_counts(base_correct: list[float], quant_correct: list[float]) -> tuple[int, int]:
    """Return ``(b, c)`` discordant pair counts from two 0/1 correctness vectors."""
    if len(base_correct) != len(quant_correct):
        raise ValueError("correctness vectors must be the same length")
    b = c = 0
    for bc, qc in zip(base_correct, quant_correct, strict=True):
        bb, qq = bc >= 0.5, qc >= 0.5
        if bb and not qq:
            b += 1
        elif not bb and qq:
            c += 1
    return b, c


def sign_test(diffs: list[float]) -> float:
    """Two-sided paired sign test p-value (ignoring exact-zero differences)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    return _binom_two_sided_p(min(pos, neg), n, 0.5)


def paired_permutation_test(
    base_scores: list[float], quant_scores: list[float], *,
    iterations: int = 10000, seed: int = 0,
) -> float:
    """Two-sided paired permutation test on per-sample score differences."""
    if len(base_scores) != len(quant_scores):
        raise ValueError("score vectors must be the same length")
    diffs = [q - b for b, q in zip(quant_scores, base_scores, strict=True)]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs) / n)
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    at_least = 0
    for _ in range(iterations):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= observed - 1e-12:
            at_least += 1
    return (at_least + 1) / (iterations + 1)  # add-one smoothing


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value (unpaired fallback)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return _two_sided_normal_p((p1 - p2) / se)


@dataclass
class SignificanceResult:
    """Outcome of a paired/unpaired significance test between two models."""

    test: str            # mcnemar | two_proportion | sign | permutation
    p_value: float
    significant: bool    # p_value < alpha
    effect: float        # baseline_score - quantized_score (positive => a drop)
    detail: str = ""


def compare_scores(
    *,
    base_score: float,
    quant_score: float,
    n: int,
    base_correct: list[float] | None = None,
    quant_correct: list[float] | None = None,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Test whether quantized vs baseline differ significantly.

    Uses McNemar's paired test when per-sample correctness vectors are supplied for
    both models; otherwise a two-proportion z-test on aggregate score * n.
    """
    effect = base_score - quant_score
    if base_correct is not None and quant_correct is not None:
        b, c = discordant_counts(base_correct, quant_correct)
        p = mcnemar_test(b, c)
        return SignificanceResult(
            test="mcnemar", p_value=p, significant=p < alpha, effect=effect,
            detail=f"discordant b={b} c={c}",
        )
    kb = round(base_score * n)
    kq = round(quant_score * n)
    p = two_proportion_ztest(kb, n, kq, n)
    return SignificanceResult(
        test="two_proportion", p_value=p, significant=p < alpha, effect=effect,
        detail=f"n={n} k_base={kb} k_quant={kq}",
    )

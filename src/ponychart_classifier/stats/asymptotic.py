"""Asymptotic chi-square goodness-of-fit tests.

Uses the chi-square distribution approximation for Pearson X^2 and
likelihood-ratio G^2 statistics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from ponychart_classifier.stats.statistics import g_stat, pearson_stat

_SMALL_EXPECTED_THRESHOLD = 5.0


def asymptotic_pearson(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
) -> tuple[float, float, int, list[str]]:
    """Asymptotic Pearson chi-square test.

    Returns (statistic, p_value, df, warnings).
    """
    effective_k = int(np.sum(probs > 0))
    df = effective_k - 1
    warnings: list[str] = []

    if df <= 0:
        return 0.0, 1.0, 0, ["Degenerate: effective categories <= 1"]

    stat = pearson_stat(counts, probs, n)
    expected = n * probs
    active_expected = expected[probs > 0]
    if np.any(active_expected < _SMALL_EXPECTED_THRESHOLD):
        warnings.append(
            f"Some expected counts < {_SMALL_EXPECTED_THRESHOLD}; "
            "asymptotic approximation may be unreliable"
        )

    p_value = float(1.0 - chi2.cdf(stat, df))
    return stat, p_value, df, warnings


def asymptotic_lr(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
) -> tuple[float, float, int, list[str]]:
    """Asymptotic likelihood-ratio (G-test) test.

    Returns (statistic, p_value, df, warnings).
    """
    effective_k = int(np.sum(probs > 0))
    df = effective_k - 1
    warnings: list[str] = []

    if df <= 0:
        return 0.0, 1.0, 0, ["Degenerate: effective categories <= 1"]

    stat = g_stat(counts, probs, n)
    expected = n * probs
    active_expected = expected[probs > 0]
    if np.any(active_expected < _SMALL_EXPECTED_THRESHOLD):
        warnings.append(
            f"Some expected counts < {_SMALL_EXPECTED_THRESHOLD}; "
            "asymptotic approximation may be unreliable"
        )

    p_value = float(1.0 - chi2.cdf(stat, df))
    return stat, p_value, df, warnings

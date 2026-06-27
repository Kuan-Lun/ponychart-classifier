"""Exact multinomial goodness-of-fit tests via full enumeration.

Enumerates all compositions of n into k non-negative integer parts and
computes exact tail probabilities under the multinomial null hypothesis.

When the number of compositions C(n+k-1, k-1) exceeds ``MAX_COMPOSITIONS``,
the caller should fall back to asymptotic tests instead.
"""

from collections.abc import Iterator
from math import comb

import numpy as np
from numpy.typing import NDArray

from ponychart_classifier.stats.statistics import (
    g_stat,
    multinomial_logpmf,
    pearson_stat,
)

MAX_COMPOSITIONS: int = 50_000


def num_compositions(n: int, k: int) -> int:
    """Number of compositions of n into k non-negative parts: C(n+k-1, k-1)."""
    return comb(n + k - 1, k - 1)


def _compositions(n: int, k: int) -> Iterator[list[int]]:
    """Yield all compositions of n into k non-negative integer parts.

    Each yielded list sums to n. Uses an iterative stars-and-bars approach
    to avoid deep recursion and excessive memory.
    """
    if k == 1:
        yield [n]
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, k - 1):
            yield [first] + rest


def _logsumexp(log_values: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not log_values:
        return float("-inf")
    arr = np.array(log_values)
    max_val: float = np.max(arr)
    if np.isinf(max_val) and max_val < 0:
        return float("-inf")
    return float(max_val + np.log(np.sum(np.exp(arr - max_val))))


def exact_pearson_pvalue(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
    mid_p: bool = False,
) -> tuple[float, dict[str, object]]:
    """Exact p-value using Pearson statistic ordering.

    p = P_H0(X^2(X) >= X^2(obs))

    Returns (p_value, metadata).
    """
    k = len(counts)
    obs_stat = pearson_stat(counts, probs, n)
    # Mask to only consider categories with positive probability
    active = probs > 0

    tail_logs: list[float] = []
    tie_logs: list[float] = []
    total_enum = 0

    for comp in _compositions(n, k):
        total_enum += 1
        x = np.array(comp, dtype=np.int64)
        # Skip impossible configurations (positive count for zero-prob category)
        if np.any(x[~active] > 0):
            continue
        logp = multinomial_logpmf(x, probs, n)
        if np.isinf(logp) and logp < 0:
            continue
        stat = pearson_stat(x, probs, n)
        if stat > obs_stat + 1e-12:
            tail_logs.append(logp)
        elif abs(stat - obs_stat) <= 1e-12:
            tie_logs.append(logp)

    tail_mass = np.exp(_logsumexp(tail_logs)) if tail_logs else 0.0
    tie_mass = np.exp(_logsumexp(tie_logs)) if tie_logs else 0.0

    if mid_p:
        p_value = float(tail_mass + 0.5 * tie_mass)
    else:
        p_value = float(tail_mass + tie_mass)

    metadata: dict[str, object] = {
        "observed_statistic": obs_stat,
        "enumeration_size": total_enum,
        "tail_set_size": len(tail_logs) + len(tie_logs),
        "mid_p": mid_p,
    }
    return min(p_value, 1.0), metadata


def exact_lr_pvalue(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
    mid_p: bool = False,
) -> tuple[float, dict[str, object]]:
    """Exact p-value using likelihood-ratio (G) statistic ordering.

    p = P_H0(G^2(X) >= G^2(obs))
    """
    k = len(counts)
    obs_stat = g_stat(counts, probs, n)
    active = probs > 0

    tail_logs: list[float] = []
    tie_logs: list[float] = []
    total_enum = 0

    for comp in _compositions(n, k):
        total_enum += 1
        x = np.array(comp, dtype=np.int64)
        if np.any(x[~active] > 0):
            continue
        logp = multinomial_logpmf(x, probs, n)
        if np.isinf(logp) and logp < 0:
            continue
        stat = g_stat(x, probs, n)
        if stat > obs_stat + 1e-12:
            tail_logs.append(logp)
        elif abs(stat - obs_stat) <= 1e-12:
            tie_logs.append(logp)

    tail_mass = np.exp(_logsumexp(tail_logs)) if tail_logs else 0.0
    tie_mass = np.exp(_logsumexp(tie_logs)) if tie_logs else 0.0

    if mid_p:
        p_value = float(tail_mass + 0.5 * tie_mass)
    else:
        p_value = float(tail_mass + tie_mass)

    metadata: dict[str, object] = {
        "observed_statistic": obs_stat,
        "enumeration_size": total_enum,
        "tail_set_size": len(tail_logs) + len(tie_logs),
        "mid_p": mid_p,
    }
    return min(p_value, 1.0), metadata


def exact_probability_pvalue(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
    mid_p: bool = False,
) -> tuple[float, dict[str, object]]:
    """Exact p-value using multinomial probability ordering.

    p = P_H0(P_H0(X) <= P_H0(obs))

    This uses sample-point probability to define extremeness rather than
    a test statistic.
    """
    k = len(counts)
    obs_logp = multinomial_logpmf(counts, probs, n)
    active = probs > 0

    tail_logs: list[float] = []
    tie_logs: list[float] = []
    total_enum = 0

    for comp in _compositions(n, k):
        total_enum += 1
        x = np.array(comp, dtype=np.int64)
        if np.any(x[~active] > 0):
            continue
        logp = multinomial_logpmf(x, probs, n)
        if np.isinf(logp) and logp < 0:
            continue
        # More extreme = lower probability = smaller logpmf
        if logp < obs_logp - 1e-12:
            tail_logs.append(logp)
        elif abs(logp - obs_logp) <= 1e-12:
            tie_logs.append(logp)

    tail_mass = np.exp(_logsumexp(tail_logs)) if tail_logs else 0.0
    tie_mass = np.exp(_logsumexp(tie_logs)) if tie_logs else 0.0

    if mid_p:
        p_value = float(tail_mass + 0.5 * tie_mass)
    else:
        p_value = float(tail_mass + tie_mass)

    metadata: dict[str, object] = {
        "observed_logpmf": obs_logp,
        "enumeration_size": total_enum,
        "tail_set_size": len(tail_logs) + len(tie_logs),
        "mid_p": mid_p,
    }
    return min(p_value, 1.0), metadata

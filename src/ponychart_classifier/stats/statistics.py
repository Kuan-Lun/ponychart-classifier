"""Core test statistics for multinomial goodness-of-fit.

Provides:
- ``pearson_stat``: Pearson chi-square statistic X^2
- ``g_stat``: Likelihood-ratio / G-test statistic G^2
- ``multinomial_logpmf``: Log of multinomial probability under H0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]


def pearson_stat(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int | None = None,
) -> float:
    """Pearson chi-square statistic: sum_i (O_i - E_i)^2 / E_i.

    Categories where probs[i] == 0 are skipped (counts[i] must be 0
    for those; the caller is responsible for validation).
    """
    if n is None:
        n = int(np.sum(counts))
    expected = n * probs
    mask = probs > 0
    diff = counts[mask] - expected[mask]
    return float(np.sum(diff**2 / expected[mask]))


def g_stat(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int | None = None,
) -> float:
    """Likelihood-ratio (G-test) statistic: 2 * sum_i O_i * log(O_i / E_i).

    Convention: 0 * log(0/E) = 0.
    Categories where probs[i] == 0 are skipped.
    """
    if n is None:
        n = int(np.sum(counts))
    expected = n * probs
    mask = (probs > 0) & (counts > 0)
    o = counts[mask].astype(np.float64)
    e = expected[mask]
    return float(2.0 * np.sum(o * np.log(o / e)))


def multinomial_logpmf(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int | None = None,
) -> float:
    """Log of multinomial probability P_H0(x).

    log P(x) = log(n!) - sum log(x_i!) + sum x_i * log(p_i)

    Convention: if p_i == 0 and x_i == 0, that term contributes 0.
    If p_i == 0 and x_i > 0, returns -inf.
    """
    if n is None:
        n = int(np.sum(counts))
    # Check impossible observations
    impossible = (probs == 0) & (counts > 0)
    if np.any(impossible):
        return float("-inf")

    log_n_fact = float(gammaln(n + 1))
    log_xi_fact = float(np.sum(gammaln(counts + 1)))
    mask = (probs > 0) & (counts > 0)
    log_prob_term = float(np.sum(counts[mask] * np.log(probs[mask])))
    return log_n_fact - log_xi_fact + log_prob_term

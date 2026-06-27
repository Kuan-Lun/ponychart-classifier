"""Unified entry point for multinomial goodness-of-fit tests.

The ``goodness_of_fit_test`` function dispatches to exact or asymptotic
implementations based on the chosen method. When exact enumeration is
infeasible (too many compositions), it automatically falls back to the
asymptotic chi-square approximation.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ponychart_classifier.stats.asymptotic import asymptotic_lr, asymptotic_pearson
from ponychart_classifier.stats.exact import (
    MAX_COMPOSITIONS,
    exact_lr_pvalue,
    exact_pearson_pvalue,
    exact_probability_pvalue,
    num_compositions,
)
from ponychart_classifier.stats.result import GoFTestResult
from ponychart_classifier.stats.statistics import (
    g_stat,
    multinomial_logpmf,
    pearson_stat,
)

Method = Literal[
    "pearson_asymptotic",
    "lr_asymptotic",
    "pearson_exact",
    "lr_exact",
    "probability_exact",
]

_ALL_METHODS: tuple[Method, ...] = (
    "pearson_asymptotic",
    "lr_asymptotic",
    "pearson_exact",
    "lr_exact",
    "probability_exact",
)


def _validate_inputs(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
) -> list[str]:
    """Validate counts and probs, returning a list of error messages."""
    errors: list[str] = []
    if counts.ndim != 1:
        errors.append("counts must be 1-dimensional")
    if probs.ndim != 1:
        errors.append("probs must be 1-dimensional")
    if len(counts) != len(probs):
        errors.append(
            f"counts and probs must have same length, "
            f"got {len(counts)} and {len(probs)}"
        )
    if np.any(counts < 0):
        errors.append("counts must be non-negative")
    if not np.issubdtype(counts.dtype, np.integer):
        errors.append("counts must be integers")
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        errors.append("probs contains NaN or Inf")
    if np.any(probs < 0):
        errors.append("probs must be non-negative")
    return errors


def goodness_of_fit_test(
    counts: list[int] | NDArray[np.int64],
    probs: list[float] | NDArray[np.float64] | None = None,
    method: Method = "pearson_exact",
    mid_p: bool = False,
) -> GoFTestResult:
    """Multinomial goodness-of-fit test.

    Tests whether observed category counts follow a specified discrete
    distribution using exact enumeration or asymptotic approximation.

    Args:
        counts: Observed counts per category.
        probs: Null hypothesis probability vector. If None, assumes uniform.
        method: One of ``"pearson_asymptotic"``, ``"lr_asymptotic"``,
            ``"pearson_exact"``, ``"lr_exact"``, ``"probability_exact"``.
        mid_p: If True, use mid-p adjustment for exact tests
            (strict tail + 0.5 * tie mass). Not a conservative exact p-value.

    Returns:
        GoFTestResult with test results and metadata.

    Raises:
        ValueError: If inputs are invalid or method is unrecognized.

    Notes:
        For exact tests, all multinomial compositions are enumerated.
        If the number of compositions exceeds the threshold (2M), the test
        automatically falls back to asymptotic approximation with a warning.

        **Exact tests do not rely on the chi-square approximation.** The
        p-value is computed directly from the multinomial distribution:

            p = P_H0(T(X) >= T(obs))

        For probability ordering:

            p = P_H0(P_H0(X) <= P_H0(obs))

        Different orderings (Pearson, LR, probability) may yield different
        exact p-values — there is no single canonical two-sided exact test
        for multinomial goodness-of-fit.
    """
    counts_arr: NDArray[np.int64] = np.asarray(counts, dtype=np.int64)
    k = len(counts_arr)

    probs_arr: NDArray[np.float64]
    if probs is None:
        probs_arr = np.full(k, 1.0 / k)
    else:
        probs_arr = np.asarray(probs, dtype=np.float64)

    # Validate
    errors = _validate_inputs(counts_arr, probs_arr)
    if errors:
        raise ValueError("; ".join(errors))

    if method not in _ALL_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(_ALL_METHODS)}")

    # Normalize probs if close to 1
    prob_sum = float(np.sum(probs_arr))
    if abs(prob_sum - 1.0) > 0.01:
        raise ValueError(f"probs sum to {prob_sum}, expected ~1.0")
    probs_arr = probs_arr / prob_sum

    n = int(np.sum(counts_arr))
    expected = n * probs_arr
    warnings: list[str] = []

    # Check impossible observations (positive count for zero-prob category)
    impossible = (probs_arr == 0) & (counts_arr > 0)
    if np.any(impossible):
        return GoFTestResult(
            method=method,
            statistic=None,
            p_value=0.0,
            df=None,
            counts=counts_arr,
            probs=probs_arr,
            expected=expected,
            n=n,
            exact=True,
            warnings=["Observed counts in zero-probability categories; p-value is 0"],
            metadata={"impossible_categories": int(np.sum(impossible))},
        )

    # Dispatch
    if method == "pearson_asymptotic":
        return _asymptotic_test(counts_arr, probs_arr, n, expected, warnings, "pearson")
    elif method == "lr_asymptotic":
        return _asymptotic_test(counts_arr, probs_arr, n, expected, warnings, "lr")
    else:
        return _exact_test(counts_arr, probs_arr, n, expected, warnings, method, mid_p)


def _asymptotic_test(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
    expected: NDArray[np.float64],
    warnings: list[str],
    variant: str,
) -> GoFTestResult:
    if variant == "pearson":
        stat, p_value, df, ws = asymptotic_pearson(counts, probs, n)
        method_name = "pearson_asymptotic"
    else:
        stat, p_value, df, ws = asymptotic_lr(counts, probs, n)
        method_name = "lr_asymptotic"
    warnings.extend(ws)
    return GoFTestResult(
        method=method_name,
        statistic=stat,
        p_value=p_value,
        df=df,
        counts=counts,
        probs=probs,
        expected=expected,
        n=n,
        exact=False,
        warnings=warnings,
        metadata={"effective_categories": int(np.sum(probs > 0))},
    )


def _exact_test(
    counts: NDArray[np.int64],
    probs: NDArray[np.float64],
    n: int,
    expected: NDArray[np.float64],
    warnings: list[str],
    method: Method,
    mid_p: bool,
) -> GoFTestResult:
    k = len(counts)
    n_comp = num_compositions(n, k)
    fallback = False

    if n_comp > MAX_COMPOSITIONS:
        fallback = True
        warnings.append(
            f"Exact enumeration infeasible ({n_comp:,} compositions > "
            f"{MAX_COMPOSITIONS:,}); falling back to asymptotic approximation"
        )

    if fallback:
        # Fall back to asymptotic version of the corresponding statistic
        if method == "probability_exact":
            # Probability ordering has no asymptotic equivalent; use Pearson
            variant = "pearson"
            warnings.append(
                "probability_exact has no asymptotic equivalent; "
                "using Pearson asymptotic as fallback"
            )
        elif method == "lr_exact":
            variant = "lr"
        else:
            variant = "pearson"
        return _asymptotic_test(counts, probs, n, expected, warnings, variant)

    # Exact enumeration
    if method == "pearson_exact":
        p_value, metadata = exact_pearson_pvalue(counts, probs, n, mid_p)
        stat: float | None = pearson_stat(counts, probs, n)
        method_name = "pearson_exact"
    elif method == "lr_exact":
        p_value, metadata = exact_lr_pvalue(counts, probs, n, mid_p)
        stat = g_stat(counts, probs, n)
        method_name = "lr_exact"
    else:  # probability_exact
        p_value, metadata = exact_probability_pvalue(counts, probs, n, mid_p)
        stat = None
        method_name = "probability_exact"
        metadata["logpmf_obs"] = multinomial_logpmf(counts, probs, n)

    return GoFTestResult(
        method=method_name,
        statistic=stat,
        p_value=p_value,
        df=None,
        counts=counts,
        probs=probs,
        expected=expected,
        n=n,
        exact=True,
        warnings=warnings,
        metadata=metadata,
    )

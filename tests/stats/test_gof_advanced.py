"""Advanced goodness-of-fit tests: fallback, scipy validation, mid-p, result structure."""

import numpy as np
import pytest
from scipy.stats import chisquare

from ponychart_classifier.stats import (
    GoFTestResult,
    goodness_of_fit_test,
)

# ── Asymptotic fallback for large n ─────────────────────────────


class TestAsymptoticFallback:
    """When exact enumeration is infeasible, should auto-fallback."""

    def test_large_n_fallback(self) -> None:
        # n=100, k=6 => C(105,5) = 96,560,646 >> MAX_COMPOSITIONS
        counts = [20, 20, 20, 15, 15, 10]
        r = goodness_of_fit_test(counts, method="pearson_exact")
        assert not r.exact  # fell back to asymptotic
        assert any("infeasible" in w.lower() for w in r.warnings)
        assert r.df == 5

    def test_probability_exact_fallback_uses_pearson(self) -> None:
        counts = [20, 20, 20, 15, 15, 10]
        r = goodness_of_fit_test(counts, method="probability_exact")
        assert not r.exact
        assert any("pearson" in w.lower() for w in r.warnings)


# ── Scipy cross-validation ──────────────────────────────────────


class TestScipyCrossValidation:
    """Verify asymptotic results match scipy.stats.chisquare."""

    def test_pearson_matches_scipy(self) -> None:
        counts = np.array([16, 18, 16, 14, 12, 12], dtype=np.int64)
        r = goodness_of_fit_test(counts.tolist(), method="pearson_asymptotic")
        scipy_stat, scipy_p = chisquare(counts)
        assert r.statistic == pytest.approx(float(scipy_stat), rel=1e-10)
        assert r.p_value == pytest.approx(float(scipy_p), rel=1e-6)

    def test_skewed_matches_scipy(self) -> None:
        counts = np.array([30, 5, 5, 5, 5, 10], dtype=np.int64)
        r = goodness_of_fit_test(counts.tolist(), method="pearson_asymptotic")
        scipy_stat, scipy_p = chisquare(counts)
        assert r.statistic == pytest.approx(float(scipy_stat), rel=1e-10)
        assert r.p_value == pytest.approx(float(scipy_p), rel=1e-6)

    def test_non_uniform_probs_matches_scipy(self) -> None:
        counts = np.array([10, 20, 30], dtype=np.int64)
        probs = [0.2, 0.3, 0.5]
        r = goodness_of_fit_test(
            counts.tolist(), probs=probs, method="pearson_asymptotic"
        )
        expected_counts = np.array(probs) * 60
        scipy_stat, scipy_p = chisquare(counts, f_exp=expected_counts)
        assert r.statistic == pytest.approx(float(scipy_stat), rel=1e-10)
        assert r.p_value == pytest.approx(float(scipy_p), rel=1e-6)


# ── Mid-p adjustment ────────────────────────────────────────────


class TestMidP:
    def test_mid_p_smaller_than_exact(self) -> None:
        counts = [4, 1, 1]
        r_exact = goodness_of_fit_test(counts, method="pearson_exact", mid_p=False)
        r_mid = goodness_of_fit_test(counts, method="pearson_exact", mid_p=True)
        # Mid-p should be <= exact p-value (since it subtracts 0.5 * tie mass)
        assert r_mid.p_value <= r_exact.p_value + 1e-12


# ── GoFTestResult structure ─────────────────────────────────────


class TestResult:
    def test_result_fields(self) -> None:
        r = goodness_of_fit_test([5, 5, 5], method="pearson_asymptotic")
        assert isinstance(r, GoFTestResult)
        assert r.method == "pearson_asymptotic"
        assert r.n == 15
        assert len(r.counts) == 3
        assert len(r.probs) == 3
        assert len(r.expected) == 3
        np.testing.assert_allclose(r.expected, [5.0, 5.0, 5.0])

    def test_probability_exact_has_logpmf(self) -> None:
        r = goodness_of_fit_test([3, 2, 1], method="probability_exact")
        assert "logpmf_obs" in r.metadata

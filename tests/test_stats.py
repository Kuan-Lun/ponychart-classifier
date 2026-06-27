"""Comprehensive tests for the multinomial goodness-of-fit module."""

import math

import numpy as np
import pytest
from scipy.stats import chisquare

from ponychart_classifier.stats import (
    GoFTestResult,
    g_stat,
    goodness_of_fit_test,
    multinomial_logpmf,
    pearson_stat,
)
from ponychart_classifier.stats.exact import (
    _compositions,
    num_compositions,
)

# ── Helper statistics tests ─────────────────────────────────────


class TestPearsonStat:
    def test_perfect_fit(self) -> None:
        counts = np.array([10, 10, 10], dtype=np.int64)
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        assert pearson_stat(counts, probs) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self) -> None:
        counts = np.array([20, 10], dtype=np.int64)
        probs = np.array([0.5, 0.5])
        # (20-15)^2/15 + (10-15)^2/15 = 25/15 + 25/15 = 50/15
        assert pearson_stat(counts, probs) == pytest.approx(50 / 15)

    def test_zero_prob_skipped(self) -> None:
        counts = np.array([10, 10, 0], dtype=np.int64)
        probs = np.array([0.5, 0.5, 0.0])
        # Only the first two categories matter
        assert pearson_stat(counts, probs) == pytest.approx(0.0, abs=1e-10)


class TestGStat:
    def test_perfect_fit(self) -> None:
        counts = np.array([10, 10, 10], dtype=np.int64)
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        assert g_stat(counts, probs) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self) -> None:
        counts = np.array([20, 10], dtype=np.int64)
        probs = np.array([0.5, 0.5])
        # 2 * (20*log(20/15) + 10*log(10/15))
        expected = 2 * (20 * math.log(20 / 15) + 10 * math.log(10 / 15))
        assert g_stat(counts, probs) == pytest.approx(expected)

    def test_zero_count_safe(self) -> None:
        counts = np.array([10, 0], dtype=np.int64)
        probs = np.array([0.5, 0.5])
        # 0 * log(0/5) is treated as 0
        expected = 2 * 10 * math.log(10 / 5)
        assert g_stat(counts, probs) == pytest.approx(expected)


class TestMultinomialLogpmf:
    def test_coin_flip(self) -> None:
        # P(X=3) for Binomial(5, 0.5) = C(5,3) * 0.5^5 = 10/32
        counts = np.array([3, 2], dtype=np.int64)
        probs = np.array([0.5, 0.5])
        expected_log = math.log(10 / 32)
        assert multinomial_logpmf(counts, probs) == pytest.approx(expected_log)

    def test_impossible_observation(self) -> None:
        counts = np.array([5, 0, 1], dtype=np.int64)
        probs = np.array([0.5, 0.5, 0.0])
        assert multinomial_logpmf(counts, probs) == float("-inf")

    def test_zero_count_zero_prob_ok(self) -> None:
        counts = np.array([5, 5, 0], dtype=np.int64)
        probs = np.array([0.5, 0.5, 0.0])
        # Should be same as Binomial(10, 0.5) P(X=5)
        expected = math.lgamma(11) - 2 * math.lgamma(6) + 10 * math.log(0.5)
        assert multinomial_logpmf(counts, probs) == pytest.approx(expected)


# ── Composition enumeration ─────────────────────────────────────


class TestCompositions:
    def test_count(self) -> None:
        n, k = 5, 3
        comps = list(_compositions(n, k))
        assert len(comps) == num_compositions(n, k)
        for c in comps:
            assert sum(c) == n
            assert all(x >= 0 for x in c)

    def test_num_compositions(self) -> None:
        assert num_compositions(10, 6) == math.comb(15, 5)
        assert num_compositions(0, 3) == 1


# ── Uniform distribution: basic cases ───────────────────────────


class TestUniformBasic:
    """Perfectly uniform counts — p-values should be large."""

    def test_pearson_asymptotic(self) -> None:
        counts = [10, 10, 10, 10, 10, 10]
        r = goodness_of_fit_test(counts, method="pearson_asymptotic")
        assert r.statistic == pytest.approx(0.0, abs=1e-10)
        assert r.p_value == pytest.approx(1.0, abs=1e-5)
        assert r.df == 5
        assert not r.exact

    def test_lr_asymptotic(self) -> None:
        counts = [10, 10, 10, 10, 10, 10]
        r = goodness_of_fit_test(counts, method="lr_asymptotic")
        assert r.statistic == pytest.approx(0.0, abs=1e-10)
        assert r.p_value == pytest.approx(1.0, abs=1e-5)
        assert r.df == 5

    def test_pearson_exact(self) -> None:
        # Use smaller n so exact enumeration is feasible for k=6
        counts = [2, 2, 2, 2, 2, 2]  # n=12, C(17,5)=6188
        r = goodness_of_fit_test(counts, method="pearson_exact")
        assert r.p_value == pytest.approx(1.0, abs=1e-10)
        assert r.exact

    def test_lr_exact(self) -> None:
        counts = [2, 2, 2, 2, 2, 2]
        r = goodness_of_fit_test(counts, method="lr_exact")
        assert r.p_value == pytest.approx(1.0, abs=1e-10)
        assert r.exact

    def test_probability_exact(self) -> None:
        counts = [2, 2, 2, 2, 2, 2]
        r = goodness_of_fit_test(counts, method="probability_exact")
        assert r.p_value == pytest.approx(1.0, abs=1e-10)
        assert r.statistic is None
        assert r.exact


# ── Obvious deviation ───────────────────────────────────────────


class TestObviousDeviation:
    """Clearly non-uniform distributions — p-values should be small."""

    def test_pearson_asymptotic_small_p(self) -> None:
        counts = [30, 5, 5, 5, 5, 10]
        r = goodness_of_fit_test(counts, method="pearson_asymptotic")
        assert r.p_value < 0.001

    def test_lr_asymptotic_small_p(self) -> None:
        counts = [30, 5, 5, 5, 5, 10]
        r = goodness_of_fit_test(counts, method="lr_asymptotic")
        assert r.p_value < 0.001

    def test_pearson_exact_small_p(self) -> None:
        # Small n version: 8 out of 12 in one category
        counts = [8, 1, 1, 1, 1, 0]  # n=12, very skewed
        r = goodness_of_fit_test(counts, method="pearson_exact")
        assert r.p_value < 0.01
        assert r.exact

    def test_lr_exact_small_p(self) -> None:
        counts = [8, 1, 1, 1, 1, 0]
        r = goodness_of_fit_test(counts, method="lr_exact")
        assert r.p_value < 0.01
        assert r.exact

    def test_probability_exact_small_p(self) -> None:
        counts = [8, 1, 1, 1, 1, 0]
        r = goodness_of_fit_test(counts, method="probability_exact")
        assert r.p_value < 0.01
        assert r.exact

    def test_large_skewed_asymptotic(self) -> None:
        """Large n falls back to asymptotic, but p-value still small."""
        counts = [30, 5, 5, 5, 5, 10]
        r = goodness_of_fit_test(counts, method="pearson_exact")
        # Falls back to asymptotic due to large n
        assert r.p_value < 0.001


# ── Small sample cases ──────────────────────────────────────────


class TestSmallSample:
    """Compare exact methods on small samples where they may disagree."""

    def test_three_categories_n6(self) -> None:
        counts = [4, 1, 1]
        results = {}
        for m in ("pearson_exact", "lr_exact", "probability_exact"):
            r = goodness_of_fit_test(counts, method=m)
            results[m] = r.p_value
            assert r.exact
            assert 0.0 <= r.p_value <= 1.0

        # All should detect some non-uniformity but may differ
        for m, p in results.items():
            assert p < 1.0, f"{m} should not give p=1"

    def test_two_categories_binomial(self) -> None:
        # Binomial case: 8 heads out of 10 fair coin flips
        counts = [8, 2]
        r_pe = goodness_of_fit_test(counts, method="pearson_exact")
        r_le = goodness_of_fit_test(counts, method="lr_exact")
        r_pr = goodness_of_fit_test(counts, method="probability_exact")
        # All should give valid p-values
        for r in (r_pe, r_le, r_pr):
            assert 0 < r.p_value < 1
            assert r.exact

    def test_exact_vs_asymptotic_small_n(self) -> None:
        """For small n, exact and asymptotic may differ noticeably."""
        counts = [3, 0, 0]
        r_exact = goodness_of_fit_test(counts, method="pearson_exact")
        r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
        # Both valid but potentially different
        assert r_exact.exact
        assert not r_asymp.exact
        assert 0 <= r_exact.p_value <= 1
        assert 0 <= r_asymp.p_value <= 1


# ── Zero counts ─────────────────────────────────────────────────


class TestZeroCounts:
    """Some observed categories are 0 but probs > 0."""

    def test_zero_in_counts(self) -> None:
        counts = [5, 5, 0, 0, 0, 0]
        r = goodness_of_fit_test(counts, method="pearson_asymptotic")
        assert r.p_value < 0.05  # clearly non-uniform
        assert len(r.warnings) > 0  # expected counts < 5

    def test_all_in_one_category(self) -> None:
        counts = [10, 0, 0]
        r = goodness_of_fit_test(counts, method="pearson_exact")
        assert r.p_value < 0.01


# ── Zero probability categories ─────────────────────────────────


class TestZeroProbs:
    def test_legal_observation(self) -> None:
        # probs[2]=0 and counts[2]=0: OK
        counts = [5, 5, 0]
        probs = [0.5, 0.5, 0.0]
        r = goodness_of_fit_test(counts, probs, method="pearson_exact")
        # Effectively a 2-category test, perfectly uniform
        assert r.p_value == pytest.approx(1.0, abs=0.01)

    def test_illegal_observation(self) -> None:
        # probs[2]=0 but counts[2]=3: impossible under H0
        counts = [5, 5, 3]
        probs = [0.5, 0.5, 0.0]
        r = goodness_of_fit_test(counts, probs, method="pearson_exact")
        assert r.p_value == 0.0
        assert "zero-probability" in r.warnings[0].lower()

    def test_illegal_asymptotic(self) -> None:
        counts = [5, 5, 3]
        probs = [0.5, 0.5, 0.0]
        r = goodness_of_fit_test(counts, probs, method="pearson_asymptotic")
        assert r.p_value == 0.0


# ── API error handling ──────────────────────────────────────────


class TestErrorHandling:
    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            goodness_of_fit_test([1, 2, 3], probs=[0.5, 0.5])

    def test_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            goodness_of_fit_test([-1, 2, 3])

    def test_probs_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            goodness_of_fit_test([1, 2, 3], probs=[0.1, 0.1, 0.1])

    def test_probs_with_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            goodness_of_fit_test([1, 2], probs=[0.5, float("nan")])

    def test_probs_with_inf(self) -> None:
        with pytest.raises(ValueError, match="NaN or Inf"):
            goodness_of_fit_test([1, 2], probs=[0.5, float("inf")])

    def test_probs_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            goodness_of_fit_test([1, 2], probs=[1.5, -0.5])

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            goodness_of_fit_test([1, 2], method="bogus")  # type: ignore[arg-type]

    def test_default_uniform_probs(self) -> None:
        r = goodness_of_fit_test([5, 5, 5], method="pearson_asymptotic")
        np.testing.assert_allclose(r.probs, [1 / 3, 1 / 3, 1 / 3])


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

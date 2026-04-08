"""Comprehensive tests for the multinomial goodness-of-fit module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chisquare  # type: ignore[import-untyped]

from ponychart_classifier.stats import (
    GoFTestResult,
    g_stat,
    goodness_of_fit_test,
    multinomial_logpmf,
    pearson_stat,
)
from ponychart_classifier.stats.exact import (
    MAX_COMPOSITIONS,
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
            r = goodness_of_fit_test(counts, method=m)  # type: ignore[arg-type]
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


# ── Exact vs asymptotic convergence study ───────────────────────


class TestConvergence:
    """Empirically check when asymptotic becomes close to exact for k=6.

    This test iterates over increasing n (with fixed proportional counts)
    and records the gap between exact and asymptotic p-values.
    """

    def test_convergence_k3(self) -> None:
        """For k=3, check convergence at various n."""
        # Slightly non-uniform proportions
        proportions = [0.5, 0.3, 0.2]
        for n in (6, 9, 12, 18, 24, 30):
            counts = [round(p * n) for p in proportions]
            # Adjust to ensure sum == n
            counts[-1] = n - sum(counts[:-1])
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                # Just verify both give valid results
                assert 0 <= r_exact.p_value <= 1
                assert 0 <= r_asymp.p_value <= 1

    def test_convergence_k6_small(self) -> None:
        """For k=6 with small n, exact is feasible.

        n=10, k=6: C(15,5) = 3003 compositions — very fast.
        """
        counts = [3, 2, 2, 1, 1, 1]
        r_exact = goodness_of_fit_test(counts, method="pearson_exact")
        r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
        assert r_exact.exact
        # Record the gap for reference
        gap = abs(r_exact.p_value - r_asymp.p_value)
        assert gap >= 0  # just checking it computes

    def test_exact_feasibility_boundary_k6(self) -> None:
        """Find approximate n where exact becomes infeasible for k=6."""
        k = 6
        for n in range(1, 100):
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                # n where we cross the threshold
                assert n < 100  # should happen well before 100
                break

    def test_convergence_report_k6(self) -> None:
        """Print exact vs asymptotic gap for k=6 at feasible n values.

        This test always passes; it prints a convergence table to help
        determine when asymptotic approximation becomes reliable.
        """
        # Non-uniform proportions (typical real-world scenario)
        proportions = [0.35, 0.15, 0.15, 0.15, 0.10, 0.10]
        gaps: list[tuple[int, float, float, float]] = []

        for n in (12, 15, 18, 20, 24):
            k = len(proportions)
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                break
            raw = [max(0, round(p * n)) for p in proportions]
            raw[-1] = n - sum(raw[:-1])
            if raw[-1] < 0:
                continue
            counts = raw
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                gaps.append(
                    (
                        n,
                        r_exact.p_value,
                        r_asymp.p_value,
                        abs(r_exact.p_value - r_asymp.p_value),
                    )
                )

        # Print convergence table (visible with pytest -s)
        print("\n=== Exact vs Asymptotic convergence (k=6, non-uniform) ===")
        print(f"{'n':>4}  {'p_exact':>10}  {'p_asymp':>10}  {'|gap|':>10}")
        for n, pe, pa, gap in gaps:
            print(f"{n:>4}  {pe:>10.6f}  {pa:>10.6f}  {gap:>10.6f}")

    def test_convergence_report_k3(self) -> None:
        """Print exact vs asymptotic gap for k=3."""
        proportions = [0.5, 0.3, 0.2]
        gaps: list[tuple[int, float, float, float]] = []

        for n in (6, 10, 15, 20, 30, 50, 80):
            k = len(proportions)
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                break
            counts = [round(p * n) for p in proportions]
            counts[-1] = n - sum(counts[:-1])
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                gaps.append(
                    (
                        n,
                        r_exact.p_value,
                        r_asymp.p_value,
                        abs(r_exact.p_value - r_asymp.p_value),
                    )
                )

        print("\n=== Exact vs Asymptotic convergence (k=3, non-uniform) ===")
        print(f"{'n':>4}  {'p_exact':>10}  {'p_asymp':>10}  {'|gap|':>10}")
        for n, pe, pa, gap in gaps:
            print(f"{n:>4}  {pe:>10.6f}  {pa:>10.6f}  {gap:>10.6f}")


# ── Type I error calibration via Monte Carlo ────────────────────

# Methods to run for each scenario. Grouped so that all methods sharing
# the same (n, probs) draw from the *same* set of multinomial samples.
_EXACT_METHODS = ("pearson_exact", "lr_exact", "probability_exact")
_ASYMP_METHODS = ("pearson_asymptotic", "lr_asymptotic")


def _simulate_rejection_rates(
    n: int,
    probs: list[float],
    methods: tuple[str, ...],
    alpha: float,
    n_sim: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Draw n_sim samples from Multinomial(n, probs) once, test all methods.

    Returns {method: rejection_rate}.
    """
    p_arr = np.array(probs)
    rejections: dict[str, int] = {m: 0 for m in methods}
    for _ in range(n_sim):
        sample = rng.multinomial(n, p_arr)
        sample_list = sample.tolist()
        for m in methods:
            r = goodness_of_fit_test(
                sample_list,
                probs=probs,
                method=m,  # type: ignore[arg-type]
            )
            if r.p_value <= alpha:
                rejections[m] += 1
    return {m: rejections[m] / n_sim for m in methods}


class TestTypeIErrorCalibration:
    """Under H0, the fraction of rejections at level alpha should be ~alpha.

    For exact tests, the rejection rate must be <= alpha (conservative).
    For asymptotic tests, the rejection rate should be approximately alpha
    when n is large enough.

    Each scenario draws samples once and evaluates all applicable methods
    on the same draws.
    """

    def test_exact_conservative_k3_uniform(self) -> None:
        """All exact methods on k=3 uniform, n=9 should be conservative."""
        rates = _simulate_rejection_rates(
            n=9,
            probs=[1 / 3, 1 / 3, 1 / 3],
            methods=_EXACT_METHODS,
            alpha=0.05,
            n_sim=2000,
            rng=np.random.default_rng(42),
        )
        for m, rate in rates.items():
            assert (
                rate <= 0.05 + 0.02
            ), f"{m} k=3 n=9: rejection rate {rate:.3f} > 0.05 + margin"

    def test_exact_conservative_k3_non_uniform(self) -> None:
        """Exact tests with non-uniform probs should still be conservative."""
        rates = _simulate_rejection_rates(
            n=12,
            probs=[0.5, 0.3, 0.2],
            methods=_EXACT_METHODS,
            alpha=0.05,
            n_sim=2000,
            rng=np.random.default_rng(303),
        )
        for m, rate in rates.items():
            assert (
                rate <= 0.05 + 0.02
            ), f"{m} k=3 non-uniform n=12: rate {rate:.3f} > 0.05 + margin"

    def test_asymptotic_calibration_k6_large_n(self) -> None:
        """With large n, asymptotic methods should reject ~alpha."""
        alpha = 0.05
        n_sim = 5000
        rates = _simulate_rejection_rates(
            n=200,
            probs=[1 / 6] * 6,
            methods=_ASYMP_METHODS,
            alpha=alpha,
            n_sim=n_sim,
            rng=np.random.default_rng(789),
        )
        se = (alpha * (1 - alpha) / n_sim) ** 0.5
        for m, rate in rates.items():
            assert abs(rate - alpha) < 3 * se, (
                f"{m} k=6 n=200: rejection rate {rate:.3f}, "
                f"expected ~{alpha} (3*SE={3*se:.3f})"
            )

    def test_asymptotic_calibration_k4_non_uniform(self) -> None:
        """Calibration with non-uniform null distribution."""
        alpha = 0.05
        n_sim = 5000
        rates = _simulate_rejection_rates(
            n=150,
            probs=[0.4, 0.3, 0.2, 0.1],
            methods=_ASYMP_METHODS,
            alpha=alpha,
            n_sim=n_sim,
            rng=np.random.default_rng(202),
        )
        se = (alpha * (1 - alpha) / n_sim) ** 0.5
        for m, rate in rates.items():
            assert abs(rate - alpha) < 3 * se, (
                f"{m} k=4 non-uniform n=150: rate {rate:.3f}, " f"expected ~{alpha}"
            )

    def test_calibration_report(self) -> None:
        """Print full calibration table. One simulation per (n, probs) group."""
        alpha = 0.05
        n_sim = 2000
        rng = np.random.default_rng(999)
        probs_k3 = [1 / 3, 1 / 3, 1 / 3]
        probs_k6 = [1 / 6] * 6

        # (label_prefix, n, probs, methods)
        scenarios: list[tuple[str, int, list[float], tuple[str, ...]]] = [
            ("k=3  n=6   exact", 6, probs_k3, _EXACT_METHODS),
            ("k=3  n=12  exact", 12, probs_k3, _EXACT_METHODS),
            ("k=3  n=30  asymp", 30, probs_k3, _ASYMP_METHODS),
            ("k=3  n=100 asymp", 100, probs_k3, _ASYMP_METHODS),
            ("k=6  n=12  exact", 12, probs_k6, _EXACT_METHODS),
            ("k=6  n=60  asymp", 60, probs_k6, _ASYMP_METHODS),
            ("k=6  n=200 asymp", 200, probs_k6, _ASYMP_METHODS),
        ]

        print(f"\n=== Type I error calibration (alpha={alpha}, n_sim={n_sim}) ===")
        print(f"{'config':<30} {'rej_rate':>8}  {'alpha':>5}  {'status'}")
        for label_prefix, n, probs, methods in scenarios:
            rates = _simulate_rejection_rates(
                n=n,
                probs=probs,
                methods=methods,
                alpha=alpha,
                n_sim=n_sim,
                rng=rng,
            )
            for m, rate in rates.items():
                short = m.replace("_asymptotic", "_a").replace("_exact", "_e")
                label = f"{label_prefix} {short}"
                is_exact = "exact" in m
                if is_exact:
                    ok = rate <= alpha + 0.02
                    status = "OK (conservative)" if ok else "FAIL"
                else:
                    se = (alpha * (1 - alpha) / n_sim) ** 0.5
                    ok = abs(rate - alpha) < 3 * se
                    status = f"OK (3SE={3*se:.3f})" if ok else "DRIFT"
                print(f"{label:<30} {rate:>8.3f}  {alpha:>5.3f}  {status}")

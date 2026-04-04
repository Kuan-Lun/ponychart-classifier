"""Basic goodness-of-fit tests: uniform, deviation, small sample, zeros, errors."""

from __future__ import annotations

import numpy as np
import pytest

from ponychart_classifier.stats import (
    goodness_of_fit_test,
)

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

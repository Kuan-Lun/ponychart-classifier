"""Tests for helper statistics: pearson_stat, g_stat, multinomial_logpmf."""

import math

import numpy as np
import pytest

from ponychart_classifier.stats import (
    g_stat,
    multinomial_logpmf,
    pearson_stat,
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

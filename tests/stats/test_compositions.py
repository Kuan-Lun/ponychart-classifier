"""Tests for composition enumeration."""

from __future__ import annotations

import math

from ponychart_classifier.stats.exact import (
    _compositions,
    num_compositions,
)

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

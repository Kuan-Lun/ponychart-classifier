from __future__ import annotations

import numpy as np

from app.label_images.data_viewer.sections.distribution_test import (
    DistributionTestSection,
)
from ponychart_classifier.stats import GoFTestResult


def _result(*, exact: bool, statistic: float | None) -> GoFTestResult:
    return GoFTestResult(
        method="pearson_exact",
        statistic=statistic,
        p_value=0.5,
        df=None,
        counts=np.array([1, 1], dtype=np.int64),
        probs=np.array([0.5, 0.5], dtype=np.float64),
        expected=np.array([1.0, 1.0], dtype=np.float64),
        n=2,
        exact=exact,
    )


def test_fmt_stat_for_probability_exact_exact_result_is_na() -> None:
    assert (
        DistributionTestSection._fmt_stat(
            "probability_exact",
            _result(exact=True, statistic=None),
        )
        == "N/A"
    )


def test_fmt_stat_for_probability_exact_fallback_marks_statistic() -> None:
    assert (
        DistributionTestSection._fmt_stat(
            "probability_exact",
            _result(exact=False, statistic=1.234),
        )
        == "1.23*"
    )


def test_fmt_stat_for_other_methods_uses_plain_statistic() -> None:
    assert (
        DistributionTestSection._fmt_stat(
            "pearson_exact",
            _result(exact=True, statistic=1.234),
        )
        == "1.23"
    )

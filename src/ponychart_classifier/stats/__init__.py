"""Multinomial goodness-of-fit testing module.

Supports exact and asymptotic tests for whether observed category counts
follow a specified discrete distribution.

Supported methods:
- Pearson chi-square (asymptotic & exact)
- Likelihood-ratio / G-test (asymptotic & exact)
- Multinomial probability ordering (exact)

For exact tests, all multinomial compositions are enumerated when feasible.
When the number of compositions exceeds a threshold, the test automatically
falls back to the asymptotic chi-square approximation.
"""

from ponychart_classifier.stats.gof import goodness_of_fit_test as goodness_of_fit_test
from ponychart_classifier.stats.result import GoFTestResult as GoFTestResult
from ponychart_classifier.stats.statistics import (
    g_stat as g_stat,
)
from ponychart_classifier.stats.statistics import (
    multinomial_logpmf as multinomial_logpmf,
)
from ponychart_classifier.stats.statistics import (
    pearson_stat as pearson_stat,
)

__all__ = [
    "GoFTestResult",
    "g_stat",
    "goodness_of_fit_test",
    "multinomial_logpmf",
    "pearson_stat",
]

"""GoFTestResult: structured return type for goodness-of-fit tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GoFTestResult:
    """Result of a multinomial goodness-of-fit test.

    Attributes:
        method: Human-readable name of the test method used.
        statistic: Test statistic value (None for probability ordering).
        p_value: The p-value of the test.
        df: Degrees of freedom (only for asymptotic tests).
        counts: Observed counts array.
        probs: Null hypothesis probability vector.
        expected: Expected counts under H0 (n * probs).
        n: Total sample size.
        exact: Whether the p-value was computed exactly.
        warnings: Any warnings generated during computation.
        metadata: Additional information (e.g. enumeration size, logpmf_obs).
    """

    method: str
    statistic: float | None
    p_value: float
    df: int | None
    counts: NDArray[np.int64]
    probs: NDArray[np.float64]
    expected: NDArray[np.float64]
    n: int
    exact: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

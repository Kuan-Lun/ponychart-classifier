"""Shared thresholds and helpers for recommendation-style conclusions."""

from __future__ import annotations

SIGNIFICANT_DELTA = 0.005
"""Default threshold for calling an F1 change meaningful."""

SMALL_DELTA = 0.003
"""Threshold for "close enough" choices where differences are negligible."""

LIMITED_HEADROOM = 0.03
"""Headroom below this is limited but still non-trivial."""

SATURATED_HEADROOM = 0.01
"""Headroom below this is effectively saturated for current modeling."""

MODERATE_CORRELATION = 0.3
"""Absolute Pearson r above this suggests a meaningful relationship."""

STRONG_CORRELATION = 0.5
"""Absolute Pearson r above this suggests a strong relationship."""


def is_significant_improvement(delta: float, *, threshold: float = SIGNIFICANT_DELTA) -> bool:
    """Return True when ``delta`` is meaningfully positive."""
    return delta > threshold


def is_significant_regression(delta: float, *, threshold: float = SIGNIFICANT_DELTA) -> bool:
    """Return True when ``delta`` is meaningfully negative."""
    return delta < -threshold


def is_negligible_delta(delta: float, *, threshold: float = SIGNIFICANT_DELTA) -> bool:
    """Return True when ``delta`` stays within the no-effect band."""
    return abs(delta) < threshold


def is_close_result(delta: float, *, threshold: float = SMALL_DELTA) -> bool:
    """Return True when two results should be treated as practically tied."""
    return abs(delta) < threshold

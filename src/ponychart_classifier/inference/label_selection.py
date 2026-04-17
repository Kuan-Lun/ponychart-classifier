"""Inference-time label selection helpers."""

from __future__ import annotations

from collections.abc import Sequence

MAX_K = 3


def select_predictions(
    probs: Sequence[float],
    thresholds: Sequence[float],
    *,
    min_k: int = 1,
    max_k: int = MAX_K,
) -> list[int]:
    """Return 0-based class indices selected by thresholds with min/max-k capping."""
    picked = [i for i, (p, t) in enumerate(zip(probs, thresholds)) if p >= t]
    if len(picked) < min_k:
        picked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:max_k]
    elif len(picked) > max_k:
        picked = sorted(picked, key=lambda i: probs[i], reverse=True)[:max_k]
    return picked

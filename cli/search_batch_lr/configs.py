"""Batch-size / LR scaling configuration for Stage 1 search."""

from __future__ import annotations

from dataclasses import dataclass

from ponychart_classifier.training import (
    BATCH_SIZE,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
)

# Stage 1 fixes lr_scale=1.0; Stage 2 fine-tunes lr_scale on the winning batch.
LR_SCALE = 1.0

# Suggested batch sizes shown in --help.  Any positive integer is accepted.
SUGGESTED_BATCH_SIZES = [16, 32, 48, 64, 96, 128]


@dataclass(frozen=True)
class BatchLrConfig:
    """Scaled LR triplet for a given batch size."""

    batch_size: int
    lr_scale: float
    lr_head: float
    lr_features: float
    lr_classifier: float


def build_config(batch_size: int) -> BatchLrConfig:
    """Apply Linear Scaling Rule to compute LRs for *batch_size*."""
    linear_factor = batch_size / BATCH_SIZE
    return BatchLrConfig(
        batch_size=batch_size,
        lr_scale=LR_SCALE,
        lr_head=LR_HEAD * linear_factor * LR_SCALE,
        lr_features=LR_FEATURES * linear_factor * LR_SCALE,
        lr_classifier=LR_CLASSIFIER * linear_factor * LR_SCALE,
    )

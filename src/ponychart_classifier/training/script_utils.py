"""Shared utilities for experiment scripts.

Eliminates boilerplate that was duplicated across ``scripts/*.py``:
seed initialisation, device detection, sample loading, holdout splitting,
test-loader construction, and seed-resetting train helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from .constants import BATCH_SIZE, CLASS_NAMES, SEED
from .dataset import PonyChartDataset, get_transforms, make_dataloader
from .device import get_device, get_performance_cpu_count
from .sampling import is_original, load_samples
from .splitting import HoldoutSplit, prepare_holdout_split
from .training import TrainResult, train_model


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def seed_all(seed: int = SEED) -> np.random.RandomState:
    """Set torch and numpy global seeds, return a fresh RandomState."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# Device / workers
# ---------------------------------------------------------------------------
def setup_device_and_workers(
    logger: logging.Logger,
) -> tuple[torch.device, int]:
    """Detect device and CPU count, log them, and return ``(device, num_workers)``."""
    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)
    return device, num_workers


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------
def load_samples_or_exit(
    logger: logging.Logger,
) -> list[tuple[str, list[int]]]:
    """Load samples, log count, and raise ``SystemExit`` if empty."""
    all_samples = load_samples()
    if not all_samples:
        logger.error("No samples found. Check rawimage/ and rawimage/labels.json.")
        raise SystemExit(1)
    logger.info("Total samples loaded: %d", len(all_samples))
    return all_samples


# ---------------------------------------------------------------------------
# Holdout split with logging
# ---------------------------------------------------------------------------
def prepare_holdout_split_logged(
    all_samples: list[tuple[str, list[int]]],
    rng: np.random.RandomState,
    logger: logging.Logger,
    test_size: float,
    val_size: float,
) -> HoldoutSplit:
    """Run :func:`prepare_holdout_split` and log the resulting split sizes."""
    split = prepare_holdout_split(
        all_samples, rng, test_size=test_size, val_size=val_size
    )
    logger.info("Test set (originals only): %d images", len(split.test))
    logger.info("Train: %s  Val: %s", f"{len(split.train):,}", f"{len(split.val):,}")
    return split


# ---------------------------------------------------------------------------
# Group-based original-only test extraction
# ---------------------------------------------------------------------------
def extract_original_test_samples(
    all_samples: list[tuple[str, list[int]]],
    test_group_keys: list[str],
    groups: dict[str, list[int]],
) -> list[tuple[str, list[int]]]:
    """Extract only original (non-crop) images from *test_group_keys*."""
    test_indices: list[int] = []
    for gk in test_group_keys:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                test_indices.append(idx)
    return [all_samples[i] for i in test_indices]


# ---------------------------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------------------------
def make_test_loader(
    test_samples: list[tuple[str, list[int]]],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> torch.utils.data.DataLoader[Any]:
    """Build a test ``DataLoader`` with standard eval transforms."""
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    return make_dataloader(
        test_ds,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )


# ---------------------------------------------------------------------------
# Train with seed reset
# ---------------------------------------------------------------------------
def train_with_seed_reset(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
    label: str,
    seed: int = SEED,
    **kwargs: Any,
) -> TrainResult:
    """Reset global RNG seeds, then run :func:`train_model`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        label,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Per-class comparison table
# ---------------------------------------------------------------------------
def log_per_class_table(
    logger: logging.Logger,
    col_labels: list[str],
    get_cell: Any,  # Callable[[int, int], str]
    col_width: int = 12,
) -> None:
    """Print a per-class table with :data:`CLASS_NAMES` as rows.

    Parameters
    ----------
    col_labels:
        Column header strings.
    get_cell:
        ``(class_index, col_index) -> formatted cell string``.
    col_width:
        Minimum width for each column.
    """
    header = f"  {'Class':<20s}"
    for lbl in col_labels:
        header += f"  {lbl:<{col_width}s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + (2 + col_width) * len(col_labels)))
    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for j in range(len(col_labels)):
            row += f"  {get_cell(i, j):<{col_width}s}"
        logger.info(row)

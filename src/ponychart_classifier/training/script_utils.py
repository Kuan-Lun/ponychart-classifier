"""Shared utilities for experiment scripts.

Eliminates boilerplate that was duplicated across ``scripts/*.py``:
seed initialisation, device detection, sample loading, holdout splitting,
test-loader construction, and seed-resetting train helpers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .constants import BATCH_SIZE, CLASS_NAMES, HOLDOUT_TEST_SIZE, SEED, VAL_SIZE
from .dataset import PonyChartDataset, TensorBatch, get_transforms, make_dataloader
from .device import get_device, get_performance_cpu_count
from .experiment_io import HASH_PREFIX_LEN, hash_samples
from .sampling import Sample, is_original, load_samples, prepare_balanced_samples
from .splitting import (
    GroupSplit,
    HoldoutSplit,
    build_groups,
    prepare_holdout_split,
    split_by_groups,
)
from .training import EvalResult, TrainResult, evaluate, train_model


def configure_logging(level: int = logging.INFO) -> None:
    """Apply the project's standard logging config once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


@dataclass(frozen=True)
class HoldoutExperimentSetup:
    """Shared device/split/data-fingerprint context for holdout experiments."""

    device: torch.device
    num_workers: int
    split: HoldoutSplit
    data_hash: str


@dataclass(frozen=True)
class GroupHoldoutExperimentSetup:
    """Shared context for group-based holdout experiments."""

    device: torch.device
    num_workers: int
    all_samples: list[Sample]
    groups: dict[str, list[int]]
    split: GroupSplit
    test_samples: list[Sample]
    data_hash: str

    @property
    def val_group_keys(self) -> set[str]:
        """Validation group keys as a set for fast membership checks."""
        return set(self.split.val)

    @property
    def train_val_group_keys(self) -> list[str]:
        """Combined train+val group keys in original split order."""
        return self.split.train + self.split.val


@dataclass(frozen=True)
class EvaluatedTrainRun:
    """Training result plus shared-test evaluation and reusable state dict."""

    train_result: TrainResult
    eval_result: EvalResult
    state_dict: dict[str, Any]


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
def load_samples_logged(
    logger: logging.Logger,
) -> list[Sample]:
    """Load samples, log count, and raise ``RuntimeError`` if empty.

    The error is logged at the raise site so callers (typically a script's
    ``main()``) can simply translate the exception to an exit code without
    re-logging the same message.
    """
    all_samples = load_samples()
    if not all_samples:
        msg = "No samples found. Check rawimage/ and rawimage/labels.json."
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Total samples loaded: %d", len(all_samples))
    return all_samples


# ---------------------------------------------------------------------------
# Holdout split with logging
# ---------------------------------------------------------------------------
def prepare_holdout_split_logged(
    all_samples: list[Sample],
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


def prepare_holdout_setup_logged(
    logger: logging.Logger,
    *,
    seed: int = SEED,
    test_size: float = HOLDOUT_TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> HoldoutExperimentSetup:
    """Build the standard holdout experiment context with logging."""
    rng = seed_all(seed)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_logged(logger)
    data_hash = hash_samples(all_samples)
    logger.info(
        "Data fingerprint: %s (full=%s)", data_hash[:HASH_PREFIX_LEN], data_hash
    )
    split = prepare_holdout_split_logged(
        all_samples,
        rng,
        logger,
        test_size=test_size,
        val_size=val_size,
    )
    return HoldoutExperimentSetup(
        device=device,
        num_workers=num_workers,
        split=split,
        data_hash=data_hash,
    )


def prepare_group_holdout_setup_logged(
    logger: logging.Logger,
    *,
    seed: int = SEED,
    test_size: float = HOLDOUT_TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> GroupHoldoutExperimentSetup:
    """Build a group-based holdout context for research scripts."""
    seed_all(seed)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_logged(logger)
    data_hash = hash_samples(all_samples)
    logger.info(
        "Data fingerprint: %s (full=%s)", data_hash[:HASH_PREFIX_LEN], data_hash
    )

    groups = build_groups(all_samples)
    split = split_by_groups(all_samples, test_size=test_size, val_size=val_size)
    test_samples = extract_original_test_samples(all_samples, split.test, groups)
    logger.info("Test set (originals only): %d images", len(test_samples))

    return GroupHoldoutExperimentSetup(
        device=device,
        num_workers=num_workers,
        all_samples=all_samples,
        groups=groups,
        split=split,
        test_samples=test_samples,
        data_hash=data_hash,
    )


# ---------------------------------------------------------------------------
# Group-based original-only test extraction
# ---------------------------------------------------------------------------
def extract_original_test_samples(
    all_samples: list[Sample],
    test_group_keys: list[str],
    groups: dict[str, list[int]],
) -> list[Sample]:
    """Extract only original (non-crop) images from *test_group_keys*."""
    test_indices: list[int] = []
    for gk in test_group_keys:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx].path)
            if is_original(fname):
                test_indices.append(idx)
    return [all_samples[i] for i in test_indices]


def collect_group_samples(
    all_samples: list[Sample],
    groups: dict[str, list[int]],
    group_keys: list[str],
) -> list[Sample]:
    """Collect all samples belonging to the given group keys."""
    indices = [idx for gk in group_keys for idx in groups[gk]]
    return [all_samples[i] for i in indices]


def split_balanced_train_val(
    samples: list[Sample],
    *,
    seed: int = SEED,
    val_group_keys: set[str],
) -> tuple[list[Sample], list[Sample]]:
    """Balance samples, then split them into train/val by group membership."""
    balanced = prepare_balanced_samples(samples, np.random.RandomState(seed))
    groups = build_groups(balanced)
    train_samples = [
        balanced[i]
        for gk, indices in groups.items()
        if gk not in val_group_keys
        for i in indices
    ]
    val_samples = [
        balanced[i]
        for gk, indices in groups.items()
        if gk in val_group_keys
        for i in indices
    ]
    return train_samples, val_samples


def prepare_balanced_train_val_for_group_keys(
    all_samples: list[Sample],
    groups: dict[str, list[int]],
    group_keys: list[str],
    *,
    seed: int = SEED,
    val_group_keys: set[str],
) -> tuple[list[Sample], list[Sample]]:
    """Collect selected groups, then build balanced train/val samples."""
    selected_samples = collect_group_samples(all_samples, groups, group_keys)
    return split_balanced_train_val(
        selected_samples,
        seed=seed,
        val_group_keys=val_group_keys,
    )


# ---------------------------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------------------------
def make_test_loader(
    test_samples: list[Sample],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> torch.utils.data.DataLoader[TensorBatch]:
    """Build a test ``DataLoader`` with standard eval transforms."""
    if device is None:
        device = torch.device("cpu")
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
    train_samples: list[Sample],
    val_samples: list[Sample],
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


def train_and_evaluate_on_test(
    train_samples: list[Sample],
    val_samples: list[Sample],
    *,
    device: torch.device,
    num_workers: int,
    label: str,
    test_loader: torch.utils.data.DataLoader[TensorBatch],
    seed: int = SEED,
    reset_seed: bool = True,
    criterion: nn.Module | None = None,
    **train_kwargs: Any,
) -> EvaluatedTrainRun:
    """Train one model and evaluate it on a shared test loader."""
    if reset_seed:
        train_result = train_with_seed_reset(
            train_samples,
            val_samples,
            device,
            num_workers,
            label,
            seed=seed,
            **train_kwargs,
        )
    else:
        train_result = train_model(
            train_samples,
            val_samples,
            device,
            num_workers,
            label,
            **train_kwargs,
        )

    eval_criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
    eval_result = evaluate(
        train_result.model,
        test_loader,
        eval_criterion,
        device,
        train_result.thresholds,
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in train_result.model.state_dict().items()
    }
    return EvaluatedTrainRun(
        train_result=train_result,
        eval_result=eval_result,
        state_dict=state_dict,
    )


# ---------------------------------------------------------------------------
# Per-class comparison table
# ---------------------------------------------------------------------------
def log_per_class_table(
    logger: logging.Logger,
    col_labels: list[str],
    get_cell: Callable[[int, int], str],
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

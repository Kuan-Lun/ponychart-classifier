"""Train/val/test splitting strategies using hash-based assignment.

Each group's assignment is determined solely by hashing its key,
making the split stable regardless of how many other samples are added or
removed.  This prevents data leakage when resuming training with new data.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .sampling import get_base_timestamp, is_original, prepare_balanced_samples

_HASH_MODULUS = 1000


@dataclass(frozen=True)
class GroupSplit:
    """Result of a hash-based group split."""

    train: list[str]
    val: list[str]
    test: list[str]


def _hash_value(group_key: str) -> float:
    """Return a deterministic hash in [0, 1) for a group key."""
    h = hashlib.md5(group_key.encode()).hexdigest()
    return (int(h, 16) % _HASH_MODULUS) / _HASH_MODULUS


def _ratios_to_thresholds(ratios: list[float]) -> list[float]:
    """Convert sequential relative ratios to absolute hash thresholds.

    Each ratio cuts from the remaining pool.  For ``[0.20, 0.15]``::

        threshold[0] = 0.20
        threshold[1] = 0.20 + 0.80 * 0.15 = 0.32
    """
    thresholds: list[float] = []
    cumulative = 0.0
    for ratio in ratios:
        cumulative += (1.0 - cumulative) * ratio
        thresholds.append(cumulative)
    return thresholds


def _split_n_way(
    group_keys: list[str],
    ratios: list[float],
) -> list[list[str]]:
    """Assign *group_keys* into ``len(ratios) + 1`` buckets in one pass.

    *ratios* are sequential relative cuts (each is a fraction of the
    remaining pool).  The last bucket collects whatever is left over.
    """
    thresholds = _ratios_to_thresholds(ratios)
    buckets: list[list[str]] = [[] for _ in range(len(ratios) + 1)]
    for gk in group_keys:
        h = _hash_value(gk)
        assigned = len(thresholds)  # default: remainder bucket
        for i, t in enumerate(thresholds):
            if h < t:
                assigned = i
                break
        buckets[assigned].append(gk)
    return buckets


def build_groups(
    samples: list[tuple[str, list[int]]],
) -> dict[str, list[int]]:
    """Build a mapping from base timestamp group key to sample indices."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path).replace(".png", "").replace(".jpg", "")
        parts = fname.split("_")
        base = "_".join(parts[:4])
        groups[base].append(idx)
    return groups


def group_hash_split(
    samples: list[tuple[str, list[int]]],
    test_size: float = 0.15,
) -> tuple[list[int], list[int]]:
    """Hash-based group split returning (train_idx, val_idx).

    Each group's assignment depends only on its own key, so adding or
    removing samples never changes existing assignments.
    """
    groups = build_groups(samples)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for gk, indices in groups.items():
        if _hash_value(gk) < test_size:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return train_idx, val_idx


def split_by_groups(
    samples: list[tuple[str, list[int]]],
    test_size: float,
    val_size: float = 0.0,
) -> GroupSplit:
    """Split timestamp groups into train / val / test sets.

    Parameters
    ----------
    samples:
        Full sample list.
    test_size:
        Fraction of the total pool allocated to the test set.
    val_size:
        Fraction of the *remaining* pool (after test) allocated to
        the validation set.  Defaults to 0 (no validation split).

    Returns
    -------
    GroupSplit
        Group keys for each split.  When *val_size* is 0 the
        ``val`` field is an empty list.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    group_keys = list(groups.keys())

    if val_size > 0:
        buckets = _split_n_way(group_keys, [test_size, val_size])
        return GroupSplit(train=buckets[2], val=buckets[1], test=buckets[0])

    buckets = _split_n_way(group_keys, [test_size])
    return GroupSplit(train=buckets[1], val=[], test=buckets[0])


logger = logging.getLogger(__name__)

_Sample = tuple[str, list[int]]


@dataclass(frozen=True)
class HoldoutSplit:
    """Ready-to-use train / val / test sample lists for holdout evaluation."""

    train: list[_Sample]
    val: list[_Sample]
    test: list[_Sample]


def prepare_holdout_split(
    samples: list[_Sample],
    rng: np.random.RandomState,
    test_size: float,
    val_size: float,
) -> HoldoutSplit:
    """Split samples into train / val / test with balanced crops.

    1. Hash-based group split into test / val / train groups.
    2. Test set contains only original images from test groups.
    3. Train+val pool is balanced via :func:`prepare_balanced_samples`.
    4. Balanced pool is split into train / val by group keys.
    """
    gsp = split_by_groups(samples, test_size=test_size, val_size=val_size)
    groups = build_groups(samples)

    # Test: originals only
    test = [
        samples[idx]
        for gk in gsp.test
        for idx in groups[gk]
        if is_original(os.path.basename(samples[idx][0]))
    ]

    # Train+val: balanced crops
    train_val_all = [samples[idx] for gk in gsp.train + gsp.val for idx in groups[gk]]
    balanced = prepare_balanced_samples(train_val_all, rng)

    # Split balanced pool by val group keys
    val_gk_set = set(gsp.val)
    tv_groups = build_groups(balanced)
    train = [
        balanced[idx]
        for gk, indices in tv_groups.items()
        if gk not in val_gk_set
        for idx in indices
    ]
    val = [
        balanced[idx]
        for gk, indices in tv_groups.items()
        if gk in val_gk_set
        for idx in indices
    ]

    logger.info(
        "Train: %d  Val: %d  Test: %d",
        len(train),
        len(val),
        len(test),
    )
    return HoldoutSplit(train=train, val=val, test=test)

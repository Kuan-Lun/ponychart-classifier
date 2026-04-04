"""Sample loading, balancing, and file-naming utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import NamedTuple

import numpy as np
import torch

from ponychart_classifier.stats import goodness_of_fit_test

from .constants import (
    GOF_ALPHA,
    LABEL_SIZE_PROBS,
    LABELS_FILE,
    NUM_CLASSES,
    RAWIMAGE_DIR,
)

logger = logging.getLogger(__name__)

ORIG_PATTERN = re.compile(r"^pony_chart_\d{8}_\d{6}\.png$")


class Sample(NamedTuple):
    """A labeled image sample: file path and 1-indexed label list."""

    path: str
    labels: list[int]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def is_original(filename: str) -> bool:
    """Check if a filename matches the original image pattern."""
    return bool(ORIG_PATTERN.match(filename))


def separate_orig_crop(
    samples: list[Sample],
) -> tuple[list[Sample], list[Sample]]:
    """Separate samples into originals and crops based on filename pattern."""
    orig = [s for s in samples if is_original(os.path.basename(s.path))]
    crop = [s for s in samples if not is_original(os.path.basename(s.path))]
    return orig, crop


def get_base_timestamp(filename: str) -> str:
    """Extract pony_chart_YYYYMMDD_HHMMSS from any variant."""
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    return "_".join(parts[:4])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels() -> dict[str, list[int]]:
    """Load raw labels.json as a dict mapping keys to label lists."""
    with open(LABELS_FILE, encoding="utf-8") as f:
        result: dict[str, list[int]] = json.load(f)
    return result


def load_samples() -> list[Sample]:
    """Load labeled samples from labels.json.

    Returns list of (image_path, [1-indexed labels]).
    Supports both flat (rawimage/xxx.png) and organized
    (rawimage/1/twilight/xxx.png) key formats.
    """
    raw = load_labels()
    samples = []
    for key, label_list in raw.items():
        # key 相對於 RAWIMAGE_DIR（例如 3/twilight/xxx.png）
        filepath_from_key = str(RAWIMAGE_DIR / key)
        if os.path.isfile(filepath_from_key):
            samples.append(Sample(filepath_from_key, label_list))
        else:
            # Fallback: 僅用檔名在 RAWIMAGE_DIR 根目錄尋找
            filename = key.split("/")[-1]
            filepath = str(RAWIMAGE_DIR / filename)
            if os.path.isfile(filepath):
                samples.append(Sample(filepath, label_list))
    logger.info(
        "Loaded %s samples (of %s labels.json entries)",
        f"{len(samples):,}",
        f"{len(raw):,}",
    )
    return samples


def compute_class_rates(
    samples: list[Sample],
) -> list[float]:
    """計算每個 class 的出現比例 (positive rate)。"""
    counts = [0] * NUM_CLASSES
    for _, labels in samples:
        for lbl in labels:
            counts[lbl - 1] += 1
    n = max(len(samples), 1)
    return [c / n for c in counts]


def compute_pos_weight(
    samples: list[Sample],
) -> torch.Tensor:
    """從訓練樣本計算 BCEWithLogitsLoss 的 pos_weight。

    公式: pos_weight[cls] = (N - pos_count) / pos_count
    """
    counts = [0] * NUM_CLASSES
    for _, labels in samples:
        for lbl in labels:
            counts[lbl - 1] += 1
    n = len(samples)
    return torch.tensor([(n - c) / max(c, 1) for c in counts], dtype=torch.float32)


def combo_counts_flat(
    samples: list[Sample],
    label_size: int,
) -> list[int]:
    """計算指定標籤數量的組合出現次數。

    回傳長度為 C(NUM_CLASSES, label_size) 的 list，
    順序依 itertools.combinations(range(1, NUM_CLASSES+1), label_size)。
    """
    combos = [tuple(sorted(s.labels)) for s in samples if len(s.labels) == label_size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, NUM_CLASSES + 1), label_size))
    return [cnt.get(c, 0) for c in all_combos]


def _should_balance(counts: list[int]) -> bool:
    """對 combo counts 做均勻性檢定，回傳是否應平衡（不拒絕 H0）。"""
    if sum(counts) == 0:
        return False
    result = goodness_of_fit_test(counts)
    logger.info(
        "GoF test (k=%d, n=%d): p=%.4f, %s",
        len(counts),
        sum(counts),
        result.p_value,
        "balance" if result.p_value > GOF_ALPHA else "skip",
    )
    return result.p_value > GOF_ALPHA


def _combo_key(labels: list[int]) -> tuple[int, ...]:
    return tuple(sorted(labels))


def _balance_one_side(
    samples: list[Sample],
    balance_flags: dict[int, bool],
    target_probs: list[float],
    rng: np.random.RandomState,
) -> list[Sample]:
    """對一側（orig 或 crop）做組合均勻化 + 比例調整，一步完成。

    對每個 label_size 組：
      - balance_flags[label_size] 為 True：調整到 target_per_combo
      - balance_flags[label_size] 為 False：原樣保留
    """
    total = len(samples)
    if total == 0:
        return []

    result: list[Sample] = []
    for label_size in (1, 2, 3):
        group = [s for s in samples if len(s.labels) == label_size]
        if not balance_flags.get(label_size, False):
            result.extend(group)
            continue
        if not group:
            continue

        group_target = total * target_probs[label_size - 1]
        n_combos = len(list(combinations(range(1, NUM_CLASSES + 1), label_size)))
        target_per_combo = max(round(group_target / n_combos), 1)

        by_combo: dict[tuple[int, ...], list[Sample]] = defaultdict(list)
        for s in group:
            by_combo[_combo_key(s.labels)].append(s)

        for combo_samples in by_combo.values():
            n_have = len(combo_samples)
            if n_have < target_per_combo:
                result.extend(combo_samples)
                extra = rng.choice(n_have, size=target_per_combo - n_have, replace=True)
                result.extend(combo_samples[i] for i in extra)
            elif n_have > target_per_combo:
                chosen = rng.choice(n_have, size=target_per_combo, replace=False)
                result.extend(combo_samples[i] for i in chosen)
            else:
                result.extend(combo_samples)

    return result


def prepare_balanced_samples(
    samples: list[Sample],
    rng: np.random.RandomState,
) -> list[Sample]:
    """按標籤組合平衡 orig 和 crop，維持標籤數量比例。

    1. 分離 orig / crop
    2. 對 orig 各標籤數量組做均勻性檢定，決定 balance_flags
    3. 檢定標籤數量比例，決定 target_probs
    4. 對 orig 和 crop 各自呼叫 _balance_one_side 一步完成
    5. 合併
    """
    orig, crop = separate_orig_crop(samples)

    # 逐組檢定
    balance_flags: dict[int, bool] = {}
    for label_size in (1, 2, 3):
        counts = combo_counts_flat(orig, label_size)
        balance_flags[label_size] = _should_balance(counts)

    # 檢定標籤數量比例，決定 target_probs
    size_counts = [sum(1 for s in orig if len(s.labels) == n) for n in (1, 2, 3)]
    total_orig = sum(size_counts)
    if total_orig > 0:
        size_result = goodness_of_fit_test(size_counts, probs=LABEL_SIZE_PROBS)
        logger.info(
            "GoF test (label-size ratio): p=%.4f, %s",
            size_result.p_value,
            "use expected" if size_result.p_value > GOF_ALPHA else "use actual",
        )
        if size_result.p_value > GOF_ALPHA:
            target_probs = LABEL_SIZE_PROBS
        else:
            target_probs = [c / total_orig for c in size_counts]
    else:
        target_probs = LABEL_SIZE_PROBS

    return _balance_one_side(
        orig, balance_flags, target_probs, rng
    ) + _balance_one_side(crop, balance_flags, target_probs, rng)


def labels_to_binary(label_list: list[int]) -> torch.Tensor:
    """Convert 1-indexed label list to binary vector."""
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in label_list:
        vec[lbl - 1] = 1.0
    return vec

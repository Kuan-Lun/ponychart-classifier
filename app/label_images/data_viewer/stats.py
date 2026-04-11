"""原圖樣本的聚合統計 helper（無 UI 依賴）。"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

from ..constants import LABEL_MAP
from ..file_ops import is_raw_image
from ..label_store import LabelStore

NUM_CLASSES = len(LABEL_MAP)


def snapshot_orig_samples(store: LabelStore) -> dict[str, list[int]]:
    """UI 執行緒呼叫：擷取 store 中「原圖」的標籤快照。"""
    orig: dict[str, list[int]] = {}
    for key in store.all_keys():
        labels = store.get(key)
        if not labels:
            continue
        if is_raw_image(Path(Path(key).name)):
            orig[key] = labels
    return orig


def count_by_label_size(samples: dict[str, list[int]]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for n in (1, 2, 3):
        counts = [0] * NUM_CLASSES
        for labels in samples.values():
            if len(labels) == n:
                for lbl in labels:
                    counts[lbl - 1] += 1
        result[n] = counts
    return result


def overall_counts(samples: dict[str, list[int]]) -> list[int]:
    counts = [0] * NUM_CLASSES
    for labels in samples.values():
        for lbl in labels:
            counts[lbl - 1] += 1
    return counts


def combo_counts_flat(samples: dict[str, list[int]], size: int) -> list[int]:
    combos = [tuple(sorted(v)) for v in samples.values() if len(v) == size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, NUM_CLASSES + 1), size))
    return [cnt.get(c, 0) for c in all_combos]

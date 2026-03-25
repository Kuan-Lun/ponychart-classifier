"""篩選函式的組合建構。"""

import re
from collections.abc import Callable
from pathlib import Path

from ponychart_classifier.model_spec import select_predictions
from ponychart_classifier.training.constants import VAL_SIZE
from ponychart_classifier.training.splitting import group_hash_split

from .constants import IMAGE_DIR
from .file_ops import is_raw_image
from .label_store import LabelStore


class FilterConfig:
    """篩選條件的資料容器。"""

    def __init__(
        self,
        *,
        raw_only: bool = False,
        uncropped_only: bool = False,
        unlabeled_only: bool = False,
        crop_mismatch: bool = False,
        train_only: bool = False,
        selected_classes: list[int] | None = None,
        show_mislabel: bool = False,
        show_missing: bool = False,
        model_probs: dict[str, list[float]] | None = None,
        model_thresholds: list[float] | None = None,
    ):
        self.raw_only = raw_only
        self.uncropped_only = uncropped_only
        self.unlabeled_only = unlabeled_only
        self.crop_mismatch = crop_mismatch
        self.train_only = train_only
        self.selected_classes = selected_classes or []
        self.show_mislabel = show_mislabel
        self.show_missing = show_missing
        self.model_probs = model_probs
        self.model_thresholds = model_thresholds


def build_filter_fn(
    config: FilterConfig,
    all_paths: list[Path],
    store: LabelStore,
) -> Callable[[Path], bool] | None:
    """根據 FilterConfig 組合出篩選函式，若無篩選條件則回傳 None。"""
    class_filter_active = len(config.selected_classes) > 0
    suspicious_active = config.model_probs is not None and (
        config.show_mislabel or config.show_missing
    )

    if (
        not config.raw_only
        and not config.uncropped_only
        and not config.unlabeled_only
        and not config.crop_mismatch
        and not config.train_only
        and not class_filter_active
        and not suspicious_active
    ):
        return None

    # 預計算 train group 的 base timestamp 集合
    train_base_timestamps: set[str] | None = None
    if config.train_only:
        samples = [
            (str(p), store.get(store.path_to_key(p)))
            for p in all_paths
            if store.has(store.path_to_key(p))
        ]
        train_idx, _ = group_hash_split(samples, test_size=VAL_SIZE)
        train_base_timestamps = {
            "_".join(Path(samples[i][0]).stem.split("_")[:4]) for i in train_idx
        }

    # 預計算已有裁切圖的 raw stem 集合
    raw_stems_with_crops: set[str] | None = None
    if config.uncropped_only:
        raw_stems_with_crops = set()
        for p in all_paths:
            if not is_raw_image(p):
                m = re.match(r"(pony_chart_\d{8}_\d{6})", p.stem)
                if m:
                    raw_stems_with_crops.add(m.group(1))

    # 預計算裁切標籤不符的 raw stem 集合
    crop_mismatch_stems: set[str] | None = None
    if config.crop_mismatch:
        crop_label_union: dict[str, set[int]] = {}
        for p in all_paths:
            if is_raw_image(p):
                continue
            m = re.match(r"(pony_chart_\d{8}_\d{6})", p.stem)
            if not m:
                continue
            raw_stem = m.group(1)
            key = store.path_to_key(p)
            crop_labels = store.get(key)
            if crop_labels:
                crop_label_union.setdefault(raw_stem, set()).update(crop_labels)
        crop_mismatch_stems = set()
        for raw_stem, union_labels in crop_label_union.items():
            raw_key = store.path_to_key(Path(IMAGE_DIR / f"{raw_stem}.png"))
            raw_labels = set(store.get(raw_key))
            if not union_labels.issubset(raw_labels):
                crop_mismatch_stems.add(raw_stem)

    model_probs = config.model_probs
    model_thresholds = config.model_thresholds
    selected_classes = config.selected_classes
    show_mislabel = config.show_mislabel
    show_missing = config.show_missing
    raw_only = config.raw_only
    uncropped_only = config.uncropped_only
    unlabeled_only = config.unlabeled_only

    # Suspicious checks: use selected classes if any, otherwise all 6
    from .constants import CLASS_NAMES_LIST

    suspicious_classes = (
        selected_classes if selected_classes else list(range(len(CLASS_NAMES_LIST)))
    )

    def predicate(p: Path) -> bool:
        if raw_only or uncropped_only:
            if not is_raw_image(p):
                return False
        if raw_stems_with_crops is not None and p.stem in raw_stems_with_crops:
            return False
        if unlabeled_only and store.has(store.path_to_key(p)):
            return False
        if train_base_timestamps is not None:
            base_ts = "_".join(p.stem.split("_")[:4])
            if base_ts not in train_base_timestamps:
                return False
        if crop_mismatch_stems is not None:
            if not is_raw_image(p):
                return False
            if p.stem not in crop_mismatch_stems:
                return False
        if class_filter_active:
            labels = store.get(store.path_to_key(p))
            if not all((ci + 1) in labels for ci in selected_classes):
                return False
        if (
            suspicious_active
            and model_probs is not None
            and model_thresholds is not None
        ):
            key = store.path_to_key(p)
            if key not in model_probs:
                return False
            probs = model_probs[key]
            labels = store.get(key)
            predicted_set = set(select_predictions(probs, model_thresholds))
            match = False
            for ci in suspicious_classes:
                has_label = (ci + 1) in labels
                pred = ci in predicted_set
                if show_mislabel and has_label and not pred:
                    match = True
                    break
                if show_missing and not has_label and pred:
                    match = True
                    break
            if not match:
                return False
        return True

    return predicate

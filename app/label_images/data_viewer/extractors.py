"""Checkpoint 載入、資料萃取與訓練／驗證集分割計算（無 UI 依賴）。"""

from pathlib import Path
from typing import Any

from ..constants import IMAGE_DIR

CHECKPOINT_PATH = IMAGE_DIR / "checkpoint.pt"


def load_checkpoint(path: Path) -> dict[str, Any]:
    """背景執行緒呼叫：載入 checkpoint 原始 dict。"""
    import torch

    result: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return result


def recompute_val_f1(path: Path) -> None:
    """背景執行緒呼叫：以目前資料集重新計算 val_f1 / per_class_f1 並寫回 checkpoint。"""
    from ponychart_classifier.training import recompute_checkpoint_val_f1

    recompute_checkpoint_val_f1(path)


def extract_counts(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取圖片數量統計。

    三欄資料皆來自 labels.json 快照（完整訓練／上次存檔／當前），
    避免掃描磁碟誤把未標註圖片計入。未標註數另外從 rawimage 根目錄
    與 rawimage/unlabeled 兩個位置取得。
    """
    from ponychart_classifier.training.constants import RAWIMAGE_DIR
    from ponychart_classifier.training.sampling import is_original, load_samples

    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    samples = load_samples()
    labels_current = {
        str(Path(p).relative_to(RAWIMAGE_DIR)): labels for p, labels in samples
    }

    def count_orig_crop(labels: dict[str, list[int]]) -> tuple[int, int]:
        n_o = sum(1 for k in labels if is_original(k.split("/")[-1]))
        return n_o, len(labels) - n_o

    n_orig_full, n_crop_full = count_orig_crop(labels_full)
    n_orig_last, n_crop_last = count_orig_crop(labels_last)
    n_cur_orig, n_cur_crop = count_orig_crop(labels_current)

    def _count_unlabeled(dir_: Path) -> int:
        if not dir_.is_dir():
            return 0
        return sum(
            1
            for f in dir_.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg")
        )

    n_unlabeled = _count_unlabeled(RAWIMAGE_DIR) + _count_unlabeled(
        RAWIMAGE_DIR / "unlabeled"
    )

    return {
        "orig_full": n_orig_full,
        "crop_full": n_crop_full,
        "orig_last": n_orig_last,
        "crop_last": n_crop_last,
        "orig_cur": n_cur_orig,
        "crop_cur": n_cur_crop,
        "total_full": len(labels_full),
        "total_last": len(labels_last),
        "total_cur": len(labels_current),
        "unlabeled": n_unlabeled,
    }


def extract_changes(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取變更明細。"""
    from ponychart_classifier.training.constants import RAWIMAGE_DIR
    from ponychart_classifier.training.sampling import is_original, load_samples

    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    samples = load_samples()
    labels_current = {
        str(Path(p).relative_to(RAWIMAGE_DIR)): labels for p, labels in samples
    }

    def _filename(key: str) -> str:
        return key.split("/")[-1]

    def diff_labels(
        baseline: dict[str, list[int]],
        current: dict[str, list[int]],
    ) -> tuple[set[str], set[str], set[str]]:
        base_keys = set(baseline)
        cur_keys = set(current)
        added = cur_keys - base_keys
        removed = base_keys - cur_keys

        # 檔案重新標注後可能被搬到不同子資料夾，key 會改變。
        # 用檔名配對 added/removed 來偵測這類「搬移」。
        added_by_name = {_filename(k): k for k in added}
        moved_names = {_filename(k) for k in removed if _filename(k) in added_by_name}
        moved_base = {k for k in removed if _filename(k) in moved_names}
        moved_cur = {added_by_name[n] for n in moved_names}

        # 搬移的檔案：若 label 有變動算 relabeled，否則忽略
        relabeled = {k for k in base_keys & cur_keys if baseline[k] != current[k]}
        for bk in moved_base:
            ck = added_by_name[_filename(bk)]
            if baseline[bk] != current[ck]:
                relabeled.add(ck)

        added -= moved_cur
        removed -= moved_base
        return added, removed, relabeled

    def split_orig_crop(keys: set[str]) -> tuple[int, int]:
        n_o = sum(1 for k in keys if is_original(k.split("/")[-1]))
        return n_o, len(keys) - n_o

    def diff_detail(
        baseline: dict[str, list[int]],
    ) -> list[tuple[int, int, int]]:
        added, removed, relabeled = diff_labels(baseline, labels_current)
        return [(len(s), *split_orig_crop(s)) for s in (added, removed, relabeled)]

    return {
        "full": diff_detail(labels_full),
        "last": diff_detail(labels_last),
    }


def extract_model(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取模型架構資訊。"""
    import torch

    from ponychart_classifier.training.model import (
        BACKBONE_REGISTRY,
        build_model,
    )

    state_dict: dict[str, Any] = ckpt.get("state_dict", {})
    n_params = sum(
        p.numel()
        for p in (
            torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for v in state_dict.values()
        )
    )

    backbone_name = ckpt.get("backbone")
    if not backbone_name:
        for name in BACKBONE_REGISTRY:
            model = build_model(backbone=name, pretrained=False)
            try:
                model.load_state_dict(state_dict)
                backbone_name = name
                break
            except RuntimeError:
                continue
        else:
            backbone_name = "unknown"

    val_f1 = ckpt.get("val_f1")

    return {
        "backbone": backbone_name,
        "input_size": ckpt.get("input_size", "N/A"),
        "num_classes": ckpt.get("num_classes", "N/A"),
        "n_params": n_params,
        "n_keys": len(state_dict),
        "val_size": ckpt.get("val_size", "N/A"),
        "val_f1": f"{val_f1:.4f}" if val_f1 is not None else "N/A",
        "per_class_f1": ckpt.get("per_class_f1"),
    }


def extract_hyperparams(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取訓練超參數。"""
    hp_keys = [
        ("seed", "Seed"),
        ("batch_size", "Batch size"),
        ("lr_head", "LR head"),
        ("lr_features", "LR features"),
        ("lr_classifier", "LR classifier"),
        ("weight_decay", "Weight decay"),
        ("label_smoothing", "Label smoothing"),
    ]
    return {label: ckpt[key] for key, label in hp_keys if ckpt.get(key) is not None}


def extract_split_counts() -> dict[str, Any]:
    """計算目前資料的訓練／驗證集張數與比例。"""
    import os

    from ponychart_classifier.training.constants import VAL_SIZE
    from ponychart_classifier.training.sampling import is_original, load_samples
    from ponychart_classifier.training.splitting import group_hash_split

    samples = load_samples()
    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)

    n_train = sum(1 for i in train_idx if is_original(os.path.basename(samples[i][0])))
    n_val = sum(1 for i in val_idx if is_original(os.path.basename(samples[i][0])))
    n_total = n_train + n_val
    return {
        "train_size": n_train,
        "val_size": n_val,
        "total_size": n_total,
        "train_ratio": n_train / n_total if n_total else 0.0,
        "val_ratio": n_val / n_total if n_total else 0.0,
    }

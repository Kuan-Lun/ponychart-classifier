"""檔案操作：整理、去重、hash、清理空資料夾。"""

import hashlib
import re
import shutil
from pathlib import Path

from .constants import CONFLICT_SUBDIR, IMAGE_DIR, LABEL_DIR_NAMES, UNLABELED_SUBDIR


def labels_to_subdir(labels: list[int]) -> str:
    """根據標籤組合計算子資料夾相對路徑（相對於 IMAGE_DIR）。

    Examples:
        [1]    -> "1/twilight"
        [1, 3] -> "2/twilight+fluttershy"
        []     -> "unlabeled"
    """
    if not labels:
        return UNLABELED_SUBDIR
    sorted_labels = sorted(set(labels))
    n = len(sorted_labels)
    combo = "+".join(LABEL_DIR_NAMES[lbl] for lbl in sorted_labels)
    return f"{n}/{combo}"


def target_path_for(filename: str, labels: list[int]) -> Path:
    """計算圖片在整理後的完整路徑。"""
    return IMAGE_DIR / labels_to_subdir(labels) / filename


def file_hash(path: Path) -> str:
    """計算檔案的 SHA-256 hash。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def organize_single(current_path: Path, labels: list[int]) -> Path:
    """將單張圖片搬到正確的子資料夾，回傳新路徑。

    如果已在正確位置則不搬移。
    目標位置已有同名檔案時：
    - hash 相同：刪除來源，視為同一檔案
    - hash 不同：將來源搬到 _conflicts/ 資料夾
    """
    target = target_path_for(current_path.name, labels)
    if current_path == target:
        return current_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(current_path) == file_hash(target):
            current_path.unlink()
            return target
        conflict_dir = IMAGE_DIR / CONFLICT_SUBDIR
        conflict_dir.mkdir(parents=True, exist_ok=True)
        conflict_path = conflict_dir / current_path.name
        n = 1
        while conflict_path.exists():
            stem = f"{current_path.stem}_{n}"
            conflict_path = conflict_dir / f"{stem}{current_path.suffix}"
            n += 1
        shutil.move(str(current_path), str(conflict_path))
        return conflict_path
    shutil.move(str(current_path), str(target))
    return target


def dedup_images(paths: list[Path]) -> list[tuple[Path, Path]]:
    """找出 hash 相同的重複圖片，保留最舊的（檔名排序最小的）。

    Returns:
        list of (duplicate_to_remove, original_to_keep) pairs.
    """
    hash_map: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    for p in sorted(paths):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in hash_map:
            duplicates.append((p, hash_map[h]))
        else:
            hash_map[h] = p
    return duplicates


def cleanup_empty_dirs(base: Path) -> None:
    """遞迴刪除空的子資料夾。"""
    for dirpath in sorted(base.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()


def is_raw_image(p: Path) -> bool:
    """判斷是否為原始圖片（pony_chart_YYYYMMDD_HHMMSS，無額外後綴）。"""
    return bool(re.fullmatch(r"pony_chart_\d{8}_\d{6}", p.stem))

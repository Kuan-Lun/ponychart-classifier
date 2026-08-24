"""檔案操作：整理、去重、hash、清理空資料夾。"""

import glob
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ponychart_classifier.image_names import (
    SUPPORTED_IMAGE_SUFFIXES,
    is_crop,
    is_original,
    parse_image_name,
)

from .constants import (
    CONFLICT_SUBDIR,
    IMAGE_DIR,
    LABEL_DIR_NAMES,
    NEAR_DUP_MAX_DIFF,
    NEAR_DUP_MAX_RATIO,
    UNLABELED_SUBDIR,
)


def scan_image_paths(base: Path) -> list[Path]:
    """掃描資料夾下所有圖片檔案（png/jpg/jpeg/webp）路徑。"""
    paths = [Path(p) for p in glob.glob(str(base / "**" / "*"), recursive=True)]
    return [
        p for p in paths if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES and p.is_file()
    ]


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
        parsed = parse_image_name(current_path.name)
        identity_stem = current_path.stem
        if parsed is not None:
            identity_stem = parsed.source_stem
            if parsed.crop_index is not None:
                identity_stem += f"_crop{parsed.crop_index}"
        n = 1
        conflict_path = conflict_dir / (
            f"{identity_stem}_conflict{n}{current_path.suffix}"
        )
        while conflict_path.exists():
            n += 1
            conflict_path = conflict_dir / (
                f"{identity_stem}_conflict{n}{current_path.suffix}"
            )
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


def _dhash(img_path: Path, hash_size: int = 8) -> int:
    """計算圖片的 difference hash（降採樣灰階後比較相鄰像素亮度）。"""
    img = (
        Image.open(img_path)
        .convert("L")
        .resize((hash_size + 1, hash_size), Image.LANCZOS)
    )
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def _is_near_duplicate(path_a: Path, path_b: Path) -> bool:
    """像素級驗證兩張圖片是否為 near-duplicate。"""
    a = np.array(Image.open(path_a))
    b = np.array(Image.open(path_b))
    if a.shape != b.shape:
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int(diff.max()) <= NEAR_DUP_MAX_DIFF and (
        np.count_nonzero(diff) / diff.size <= NEAR_DUP_MAX_RATIO
    )


def dedup_near_images(paths: list[Path]) -> list[tuple[Path, Path]]:
    """找出像素幾乎相同的 near-duplicate 圖片，保留最舊的。

    流程：dhash 分組 → 同組內 pixel-level 驗證。

    Returns:
        list of (duplicate_to_remove, original_to_keep) pairs.
    """
    hash_map: dict[int, list[Path]] = {}
    for p in sorted(paths):
        h = _dhash(p)
        hash_map.setdefault(h, []).append(p)

    duplicates: list[tuple[Path, Path]] = []
    for group in hash_map.values():
        if len(group) < 2:
            continue
        keep = group[0]
        for other in group[1:]:
            if _is_near_duplicate(keep, other):
                duplicates.append((other, keep))
    return duplicates


def cleanup_empty_dirs(base: Path) -> None:
    """遞迴刪除空的子資料夾。"""
    for dirpath in sorted(base.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()


def is_raw_image(p: Path) -> bool:
    """判斷是否為目前命名格式的原始圖片。"""
    return is_original(p.name)


def is_crop_image(p: Path) -> bool:
    """判斷是否為帶有明確裁切後綴的圖片；未知格式不視為裁切圖。"""
    return is_crop(p.name)


def next_crop_name(parent: Path, base_stem: str, suffix: str) -> Path:
    """找出 base_stem 尚未使用的最小 _cropN 編號，回傳 parent 下的新檔案路徑。

    裁切圖標註後會被搬到依標籤分類的子資料夾，因此已用過的編號在 `parent`
    目錄中可能已不存在；必須在整個 IMAGE_DIR 樹狀結構中檢查，避免編號重複
    造成不同裁切圖檔名衝突。
    """
    pattern = re.compile(rf"^{re.escape(base_stem)}_crop(\d+){re.escape(suffix)}$")
    used = {
        int(m.group(1))
        for p in IMAGE_DIR.rglob(f"{base_stem}_crop*{suffix}")
        if (m := pattern.match(p.name))
    }
    n = 1
    while n in used:
        n += 1
    return parent / f"{base_stem}_crop{n}{suffix}"


def has_existing_crop(stem: str) -> bool:
    """依完整原圖 stem 判斷是否已有任何衍生裁切圖。"""
    return any(IMAGE_DIR.rglob(f"{stem}_crop*"))

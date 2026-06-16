"""
清理短邊過小的裁切圖。

條件：短邊（min(width, height)）小於裁切框短邊下限。
  裁切框長寬比鎖定為 CROP_ASPECT_RATIO，
  短邊下限 = CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO；
  短於此下限代表訓練時 resize 會放大、產生模糊。

預設為 dry-run，僅列出符合條件的檔案；加上 --execute 才會實際刪除。
刪除後請在 app.label_images 執行一次「整理全部」，
以同步 labels.json 與 analysis_cache.json 的孤兒 entry。

使用方式（需以 -m 執行，才能載入 app.label_images 套件）：
  uv run python -m scripts.prune_small_confident_crops
  uv run python -m scripts.prune_small_confident_crops --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.label_images.constants import (
    CROP_ASPECT_RATIO,
    CROP_MIN_WIDTH_ORIG,
    IMAGE_DIR,
)
from app.label_images.file_ops import scan_image_paths
from ponychart_classifier.training import is_original

# CROP_MIN_WIDTH_ORIG 是裁切框長邊下限；裁切框長寬比鎖定為 CROP_ASPECT_RATIO，
# 換算成短邊下限（對應 crop_handler._min_width_for 在 portrait 方向的下限）。
_CROP_MIN_SHORT_SIDE = CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO


def _is_too_small(path: Path) -> bool:
    with Image.open(path) as img:
        w, h = img.size
        size: int = min(w, h)
    return size < _CROP_MIN_SHORT_SIDE


def find_candidates() -> list[Path]:
    all_paths = scan_image_paths(IMAGE_DIR)
    return [p for p in all_paths if not is_original(p.name) and _is_too_small(p)]


def main() -> None:
    parser = argparse.ArgumentParser(description="清理短邊過小的裁切圖")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="實際刪除檔案（預設僅 dry-run 列出清單）",
    )
    args = parser.parse_args()

    candidates = find_candidates()
    if not candidates:
        print("沒有符合條件的裁切圖。")
        return

    for p in sorted(candidates):
        print(p.relative_to(IMAGE_DIR))
    print(f"\n共 {len(candidates)} 張裁切圖符合條件。")

    if not args.execute:
        print("此為 dry-run，加上 --execute 才會實際刪除。")
        return

    for p in candidates:
        p.unlink()
    print(
        f"已刪除 {len(candidates)} 張裁切圖。"
        "請在 app.label_images 執行一次「整理全部」以同步"
        " labels.json / analysis_cache.json。"
    )


if __name__ == "__main__":
    main()

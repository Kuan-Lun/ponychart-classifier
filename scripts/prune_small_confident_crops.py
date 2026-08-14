"""
清理短邊過小的裁切圖。

條件一：短邊（min(width, height)）小於裁切框短邊下限。
  裁切框長寬比鎖定為 CROP_ASPECT_RATIO，
  短邊下限 = CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO；
  短於此下限代表訓練時 resize 會放大、產生模糊。

條件二（可選，需加 --require）：原圖每個類別的 AnalysisTable 狀態
  都必須在指定的允許集合內。狀態符號（對應 app.label_images 顯示）：
    ==  有標籤 + 正確預測 + 信心足夠（|prob − thr| ≥ SUSPICIOUS_MARGIN）
    =   有標籤 + 正確預測 + 信心不足
    --  有標籤 + 漏判 + 信心足夠
    -   有標籤 + 漏判 + 信心不足
    ++  無標籤 + 誤判 + 信心足夠
    +   無標籤 + 誤判 + 信心不足
    （空白 = 無標籤 + 無預測，永遠允許）

預設為 dry-run，僅列出符合條件的檔案；加上 --execute 才會實際刪除。
刪除後請在 app.label_images 執行一次「整理全部」，
以同步 labels.json 與 analysis_cache.json 的孤兒 entry。

使用方式（需以 -m 執行，才能載入 app.label_images 套件）：
  uv run python -m scripts.prune_small_confident_crops
  uv run python -m scripts.prune_small_confident_crops --require ==
  uv run python -m scripts.prune_small_confident_crops --require ==,=
  uv run python -m scripts.prune_small_confident_crops --execute --require ==,=,+
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.label_images.analysis import AnalysisManager
from app.label_images.constants import (
    CROP_ASPECT_RATIO,
    CROP_MIN_WIDTH_ORIG,
    IMAGE_DIR,
    IMAGE_SUBDIR,
    LABEL_FILE,
    SUSPICIOUS_MARGIN,
)
from app.label_images.file_ops import scan_image_paths
from app.label_images.label_store import LabelStore
from ponychart_classifier.inference import select_predictions
from ponychart_classifier.training import extract_source_stem, is_crop, is_original

# CROP_MIN_WIDTH_ORIG 是裁切框長邊下限；裁切框長寬比鎖定為 CROP_ASPECT_RATIO，
# 換算成短邊下限（對應 crop_handler._min_width_for 在 portrait 方向的下限）。
_CROP_MIN_SHORT_SIDE = CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO

_VALID_REQUIRE_VALUES: frozenset[str] = frozenset({"==", "=", "--", "-", "++", "+"})


def _is_too_small(path: Path) -> bool:
    with Image.open(path) as img:
        w, h = img.size
        size: int = min(w, h)
    return size < _CROP_MIN_SHORT_SIDE


def _class_state(prob: float, threshold: float, has_label: bool, pred: bool) -> str:
    """回傳與 AnalysisTable 顯示對應的狀態符號（CLI 以 ASCII 表示 −/−−）。"""
    confident = abs(prob - threshold) >= SUSPICIOUS_MARGIN
    if has_label and pred:
        return "==" if confident else "="
    if has_label and not pred:
        return "--" if confident else "-"
    if not has_label and pred:
        return "++" if confident else "+"
    return ""


def _all_states_acceptable(
    probs: list[float],
    thresholds: list[float],
    labels: list[int],
    allowed: frozenset[str],
) -> bool:
    """所有類別的狀態都在 allowed 集合內（空白 true negative 永遠允許）。"""
    label_set = {c - 1 for c in labels}
    predicted_set = set(select_predictions(probs, thresholds))
    for c in range(len(probs)):
        state = _class_state(
            probs[c], thresholds[c], c in label_set, c in predicted_set
        )
        if state and state not in allowed:
            return False
    return True


def find_candidates(require: frozenset[str] | None = None) -> list[Path]:
    all_paths = scan_image_paths(IMAGE_DIR)
    small_crops = [p for p in all_paths if is_crop(p.name) and _is_too_small(p)]

    if require is None:
        return small_crops

    analysis = AnalysisManager()
    if analysis.model_probs is None or analysis.model_thresholds is None:
        raise SystemExit(
            "缺少模型分析快取或 thresholds，請先在 app.label_images 執行一次模型分析。"
        )
    thresholds = analysis.model_thresholds

    store = LabelStore(LABEL_FILE, IMAGE_SUBDIR)
    originals_by_stem = {p.stem: p for p in all_paths if is_original(p.name)}

    candidates: list[Path] = []
    for p in small_crops:
        stem = extract_source_stem(p.name)
        if stem is None:
            continue
        orig_path = originals_by_stem.get(stem)
        if orig_path is None:
            continue
        orig_key = store.path_to_key(orig_path)
        labels = store.get(orig_key)
        if not labels:
            continue
        probs = analysis.get_image_probs(orig_key)
        if probs is None:
            continue
        if _all_states_acceptable(probs, thresholds, labels, require):
            candidates.append(p)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="清理短邊過小的裁切圖")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="實際刪除檔案（預設僅 dry-run 列出清單）",
    )
    parser.add_argument(
        "--require",
        metavar="STATES",
        help=(
            "以逗號分隔的 AnalysisTable 狀態碼，例如 == 或 ==,= 或 ==,=,+。"
            f" 有效值：{', '.join(sorted(_VALID_REQUIRE_VALUES))}。"
            " 指定後，僅當原圖所有類別的狀態都在此集合內才納入候選。"
            " 未指定時不進行模型狀態檢查。"
        ),
    )
    args = parser.parse_args()

    require: frozenset[str] | None = None
    if args.require:
        states = frozenset(s.strip() for s in args.require.split(","))
        invalid = states - _VALID_REQUIRE_VALUES
        if invalid:
            parser.error(
                f"無效的狀態碼：{', '.join(sorted(invalid))}。"
                f" 有效值為：{', '.join(sorted(_VALID_REQUIRE_VALUES))}"
            )
        require = states

    candidates = find_candidates(require)
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

"""自動挖掘會誤導模型的裁切圖（hard negative crop mining）。

流程：隨機選一張「已標籤、且尚無任何衍生裁切圖」的原圖，套用跟
app.label_images 手動裁切相同的旋轉裁切框數學（隨機中心點/大小/方向/角度，
保證 4 角點落在原圖範圍內），裁出一張候選圖後跑一次模型推論：若有任何
不在該原圖標籤集合內的角色，其 sigmoid 分數超過該角色的閾值，就視為一次
成功的「誤導裁切」，存到 rawimage/unlabeled/ 供使用者之後在 app.label_images
複核並標籤。

命中率沒有特別優化的必要：命中率低到一定程度，本身就代表已經沒有更多
值得挖的困難案例，--max-failures 會在那之前自動停止。

使用方式（需以 -m 執行，才能載入 app.label_images 套件）：
  uv run python -m scripts.mine_hard_negative_crops --count 20
  uv run python -m scripts.mine_hard_negative_crops --count 20 --max-failures 50 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import tempfile
from pathlib import Path

from PIL import Image

import ponychart_classifier
from app.label_images import crop_geometry
from app.label_images.constants import (
    CROP_ASPECT_RATIO,
    CROP_MIN_WIDTH_ORIG,
    IMAGE_DIR,
    UNLABELED_SUBDIR,
)
from app.label_images.file_ops import has_existing_crop, next_crop_name
from ponychart_classifier.inference import ClassThresholds, PredictionResult
from ponychart_classifier.model_spec import PONY_CLASSES
from ponychart_classifier.training import (
    Sample,
    extract_source_stem,
    load_samples,
    separate_orig_crop,
)

logger = logging.getLogger(__name__)

_MAX_ANGLE_DEG = 30.0
_GEOMETRY_RETRY_LIMIT = 10


def find_candidate_pool() -> list[Sample]:
    """回傳目前可挖掘的候選原圖：已標籤、且尚無任何衍生裁切圖。"""
    originals, _ = separate_orig_crop(load_samples())
    pool: list[Sample] = []
    for sample in originals:
        if not sample.labels:
            continue
        stem = extract_source_stem(Path(sample.path).name)
        if stem is not None and not has_existing_crop(stem):
            pool.append(sample)
    return pool


def _random_crop_box(
    img_w: int, img_h: int, rng: random.Random
) -> tuple[crop_geometry.CropCorners, int, int] | None:
    """在原圖像素座標內，隨機產生一個合法的旋轉裁切框；放不下回傳 None。

    裁切框長邊（原圖像素）下限為 ``CROP_MIN_WIDTH_ORIG``，與 app.label_images
    手動裁切的品質下限一致（避免訓練 resize 時放大模糊）；若該角度/方向放不下
    這個下限，視為這次幾何嘗試失敗（不像 GUI 為了可用性而降級），重抽角度/方向。

    回傳裁切框角點與對應的輸出 ``(width, height)``（四捨五入為整數像素）。
    """
    max_angle = math.radians(_MAX_ANGLE_DEG)
    for _ in range(_GEOMETRY_RETRY_LIMIT):
        portrait = rng.choice([True, False])
        angle_rad = rng.uniform(-max_angle, max_angle)
        max_w = crop_geometry.max_width(
            img_w, img_h, CROP_ASPECT_RATIO, portrait, angle_rad
        )
        floor = (
            CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO if portrait else CROP_MIN_WIDTH_ORIG
        )
        if max_w < floor:
            continue
        width = rng.uniform(floor, max_w)
        height = crop_geometry.height_for(width, CROP_ASPECT_RATIO, portrait)
        half_w, half_h = crop_geometry.half_extents(width, height, angle_rad)
        center = crop_geometry.Point(
            rng.uniform(half_w, img_w - half_w),
            rng.uniform(half_h, img_h - half_h),
        )
        corners = crop_geometry.compute_corners(
            center, width, CROP_ASPECT_RATIO, angle_rad, portrait
        )
        return corners, max(1, round(width)), max(1, round(height))
    return None


def _misclassified_characters(
    result: PredictionResult, thresholds: ClassThresholds, labels: list[int]
) -> list[tuple[str, float]]:
    """回傳不在 ``labels``（1-indexed）內、但分數超過閾值的角色與分數。"""
    label_set = {c - 1 for c in labels}
    hits: list[tuple[str, float]] = []
    for i, pony_class in enumerate(PONY_CLASSES):
        if i in label_set:
            continue
        score = getattr(result, pony_class.value)
        threshold = getattr(thresholds, pony_class.value)
        if score > threshold:
            hits.append((pony_class.value, score))
    return hits


def try_one_crop(
    sample: Sample, thresholds: ClassThresholds, rng: random.Random
) -> tuple[Path, list[tuple[str, float]]] | None:
    """嘗試對一張原圖裁出一個隨機裁切框並推論；成功回傳暫存檔路徑與誤判角色。"""
    src_path = Path(sample.path)
    with Image.open(src_path) as img:
        image = img.convert("RGB")
    img_w, img_h = image.size

    crop_box = _random_crop_box(img_w, img_h, rng)
    if crop_box is None:
        return None
    corners, width_orig, height_orig = crop_box

    cropped = image.transform(
        (width_orig, height_orig),
        Image.Transform.QUAD,
        corners.as_quad(),
        Image.Resampling.BICUBIC,
    )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=src_path.suffix)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        cropped.save(tmp_path)
        result = ponychart_classifier.predict(str(tmp_path))
        hits = _misclassified_characters(result, thresholds, sample.labels)
        if not hits:
            tmp_path.unlink()
            return None
        return tmp_path, hits
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def mine(count: int, max_failures: int, rng: random.Random) -> int:
    pool = find_candidate_pool()
    thresholds = ponychart_classifier.get_thresholds()

    successes = 0
    consecutive_failures = 0
    while successes < count:
        if not pool:
            print("沒有更多候選原圖可挖掘。")
            break
        if consecutive_failures >= max_failures:
            print(f"連續失敗達 {max_failures} 次，停止挖掘。")
            break

        sample = rng.choice(pool)
        hit = try_one_crop(sample, thresholds, rng)
        if hit is None:
            consecutive_failures += 1
            continue

        tmp_path, hits = hit
        stem = extract_source_stem(Path(sample.path).name)
        assert stem is not None
        dest = next_crop_name(IMAGE_DIR / UNLABELED_SUBDIR, stem, tmp_path.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(dest)

        hits_desc = ", ".join(f"{name}={score:.2f}" for name, score in hits)
        print(f"命中：{stem} -> {dest.relative_to(IMAGE_DIR)}（誤判：{hits_desc}）")

        pool.remove(sample)
        successes += 1
        consecutive_failures = 0

    print(f"\n共產生 {successes} 張裁切圖。請在 app.label_images 複核並標籤。")
    return successes


def main() -> None:
    parser = argparse.ArgumentParser(description="自動挖掘會誤導模型的裁切圖")
    parser.add_argument(
        "--count", "-n", type=int, required=True, help="目標累積成功張數"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=50,
        help="連續失敗（含找不到候選原圖）達此數即停止（預設 50）",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="隨機種子（用於重現結果）"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mine(args.count, args.max_failures, rng)


if __name__ == "__main__":
    main()

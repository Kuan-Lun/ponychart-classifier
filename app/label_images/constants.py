"""共用常數：路徑、標籤對應、顯示設定。"""

from pathlib import Path

from ponychart_classifier.model_spec import (
    DISPLAY_NAME_BY_CLASS,
    INPUT_SIZE,
    PONY_CLASSES,
)

# 所有路徑以 repo root 為基準
REPO_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_SUBDIR = "rawimage"
IMAGE_DIR = REPO_DIR / "rawimage"
LABEL_FILE = IMAGE_DIR / "labels.json"
ANALYSIS_CACHE_FILE = IMAGE_DIR / "analysis_cache.json"
MAX_SIZE = 1004

# 裁切框：長寬比鎖定為 INPUT_SIZE，確保訓練時 resize 不會造成比例失真
CROP_ASPECT_RATIO = INPUT_SIZE.width / INPUT_SIZE.height
# 主視窗圖片顯示區域的固定尺寸；圖片超過則等比縮小，
# 確保切換圖片時視窗大小不會改變
DISPLAY_WIDTH = MAX_SIZE
DISPLAY_HEIGHT = round(MAX_SIZE / CROP_ASPECT_RATIO)
# 裁切寬度（原圖像素）下限，避免訓練 resize 時過度放大造成模糊
CROP_MIN_WIDTH_ORIG = INPUT_SIZE.width
# 進入裁切模式時，預設裁切框最大可佔畫面比例
CROP_INITIAL_FRACTION = 0.6
# WASD 每次移動的畫面像素
CROP_MOVE_STEP = 10
# QE 每次旋轉的角度（度）
CROP_ROTATE_STEP_DEG = 1.0
# RF 每次縮放的倍率
CROP_SCALE_STEP = 1.05
# 按住 Shift 時，WASD 每次移動的畫面像素
CROP_MOVE_STEP_FAST = 30
# 按住 Shift 時，QE 每次旋轉的角度（度）
CROP_ROTATE_STEP_DEG_FAST = 5.0
# 按住 Shift 時，RF 每次縮放的倍率
CROP_SCALE_STEP_FAST = 1.20

LABEL_MAP: dict[int, str] = {
    index: DISPLAY_NAME_BY_CLASS[pony_class]
    for index, pony_class in enumerate(PONY_CLASSES, start=1)
}

LABEL_DIR_NAMES: dict[int, str] = {
    1: "twilight",
    2: "rarity",
    3: "fluttershy",
    4: "rainbow_dash",
    5: "pinkie_pie",
    6: "applejack",
}

UNLABELED_SUBDIR = "unlabeled"
CONFLICT_SUBDIR = "_conflicts"

CLASS_NAMES_LIST: list[str] = [LABEL_MAP[i] for i in range(1, 7)]

# Model analysis: |prob - threshold| below this is considered ambiguous
SUSPICIOUS_MARGIN = 0.15

# Near-duplicate detection: pixel-level thresholds
NEAR_DUP_MAX_DIFF = 10  # 單一像素最大容許差異 (0-255)
NEAR_DUP_MAX_RATIO = 0.01  # 不同像素佔比上限 (1%)

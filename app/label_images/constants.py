"""共用常數：路徑、標籤對應、顯示設定。"""

from pathlib import Path

from ponychart_classifier.model_spec import DISPLAY_NAME_BY_CLASS, PONY_CLASSES

# 所有路徑以 repo root 為基準
REPO_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_SUBDIR = "rawimage"
IMAGE_DIR = REPO_DIR / "rawimage"
LABEL_FILE = IMAGE_DIR / "labels.json"
ANALYSIS_CACHE_FILE = IMAGE_DIR / "analysis_cache.json"
MAX_SIZE = 1004

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

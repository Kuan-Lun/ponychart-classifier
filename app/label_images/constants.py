"""共用常數：路徑、標籤對應、顯示設定。"""

from pathlib import Path

# 所有路徑以 repo root 為基準
REPO_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_SUBDIR = "rawimage"
IMAGE_DIR = REPO_DIR / "rawimage"
LABEL_FILE = IMAGE_DIR / "labels.json"
MAX_SIZE = 800

LABEL_MAP: dict[int, str] = {
    1: "Twilight Sparkle",
    2: "Rarity",
    3: "Fluttershy",
    4: "Rainbow Dash",
    5: "Pinkie Pie",
    6: "Applejack",
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

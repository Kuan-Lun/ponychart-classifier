"""Training-specific constants for PonyChart ML scripts."""

from __future__ import annotations

from pathlib import Path

from ponychart_classifier.model_spec import (  # noqa: F401
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    ImageSize,
)

__all__ = [
    "ARTIFACTS_DIR",
    "CLASS_NAMES",
    "GOF_RECENT_ORIG_LIMIT",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_SIZE",
    "ImageSize",
    "NUM_CLASSES",
]

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/ponychart_classifier/
REPO_DIR = PACKAGE_DIR.parent.parent  # repo root
ARTIFACTS_DIR = REPO_DIR / "artifacts"

RAWIMAGE_DIR = REPO_DIR / "rawimage"
LABELS_FILE = RAWIMAGE_DIR / "labels.json"
OUTPUT_CHECKPOINT = RAWIMAGE_DIR / "checkpoint.pt"

OUTPUT_ONNX = ARTIFACTS_DIR / "model.onnx"
OUTPUT_THRESHOLDS = ARTIFACTS_DIR / "thresholds.json"

# Training hyperparameters (single source of truth)
BACKBONE = "efficientnet_b0"
BATCH_SIZE = 64
SEED = 42
PHASE1_EPOCHS = 30
PHASE1_PATIENCE = 5
PHASE2_EPOCHS = 100
PHASE2_PATIENCE = 12
MIN_DELTA_LOSS = 0.005
MIN_DELTA_F1 = 0.005
LR_HEAD = 4e-3
LR_FEATURES = 1.2e-4
LR_CLASSIFIER = 1.2e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-7

# Data split ratios
HOLDOUT_TEST_SIZE = 0.20
VAL_SIZE = 0.15

# Balancing: goodness-of-fit significance level
GOF_ALPHA = 0.05
GOF_RECENT_ORIG_LIMIT = 3000

# Expected label-size ratios (single / double / triple)
# Automatically normalized to probabilities; values only need to be proportional.
_LABEL_SIZE_RATIOS: list[float] = [2631.71, 2738.32, 1212.97]
if any(v < 0 for v in _LABEL_SIZE_RATIOS):
    raise ValueError(
        f"_LABEL_SIZE_RATIOS contains negative values: {_LABEL_SIZE_RATIOS}"
    )
_ratio_sum = sum(_LABEL_SIZE_RATIOS)
LABEL_SIZE_PROBS: list[float] = [v / _ratio_sum for v in _LABEL_SIZE_RATIOS]

# Resume vs from-scratch threshold (updated by compare_resume_scratch analysis)
# 以「上次 from-scratch 訓練時的樣本數」為基準計算累積 ratio：
#   cumulative_ratio = (目前樣本數 - n_samples_at_full_train) / n_samples_at_full_train
# 超過此值時自動切換為 from-scratch 訓練，避免多次增量訓練累積偏差。
# 例：full_train 用 1000 張，現在有 1040 張 → ratio=0.04 < 0.05 → resume
# 例：full_train 用 1000 張，現在有 1060 張 → ratio=0.06 > 0.05 → from-scratch
RETRAIN_NEW_DATA_RATIO = 0.05

# Reduced settings for hyperparameter search (derived from main settings)
SEARCH_PHASE1_EPOCHS = PHASE1_EPOCHS
SEARCH_PHASE2_EPOCHS = PHASE2_EPOCHS
SEARCH_PATIENCE = PHASE2_PATIENCE

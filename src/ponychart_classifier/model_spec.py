"""Model specification constants shared by inference and training."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

NUM_CLASSES = 6
CLASS_NAMES = [
    "Twilight Sparkle",
    "Rarity",
    "Fluttershy",
    "Rainbow Dash",
    "Pinkie Pie",
    "Applejack",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 320
PRE_RESIZE = 384

MAX_K = 3


@dataclasses.dataclass(frozen=True)
class ClassThresholds:
    """Per-class sigmoid thresholds for multi-label prediction."""

    twilight_sparkle: float
    rarity: float
    fluttershy: float
    rainbow_dash: float
    pinkie_pie: float
    applejack: float

    def as_list(self) -> list[float]:
        """Return thresholds as a list ordered by :data:`CLASS_NAMES`."""
        return [
            self.twilight_sparkle,
            self.rarity,
            self.fluttershy,
            self.rainbow_dash,
            self.pinkie_pie,
            self.applejack,
        ]


@dataclasses.dataclass(frozen=True)
class PredictionResult:
    """Inference result with per-character scores and selected labels."""

    twilight_sparkle: float
    rarity: float
    fluttershy: float
    rainbow_dash: float
    pinkie_pie: float
    applejack: float
    labels: frozenset[str]


def select_predictions(
    probs: Sequence[float],
    thresholds: Sequence[float],
    *,
    min_k: int = 1,
    max_k: int = MAX_K,
) -> list[int]:
    """Return 0-based class indices selected by thresholds with min/max-k capping."""
    picked = [i for i, (p, t) in enumerate(zip(probs, thresholds)) if p >= t]
    if len(picked) < min_k:
        picked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:max_k]
    elif len(picked) > max_k:
        picked = sorted(picked, key=lambda i: probs[i], reverse=True)[:max_k]
    return picked

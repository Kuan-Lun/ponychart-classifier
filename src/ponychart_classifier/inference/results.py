"""Inference result data models."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

from ponychart_classifier.model_spec import PONY_CLASSES, PonyClass


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
        """Return thresholds as a list ordered by the model's class order."""
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


def build_thresholds(scores: Mapping[PonyClass, float]) -> ClassThresholds:
    """Build :class:`ClassThresholds` from a stable class-id keyed mapping."""
    values = {pony_class.value: scores[pony_class] for pony_class in PONY_CLASSES}
    return ClassThresholds(**values)


def build_prediction_result(
    scores: Mapping[PonyClass, float],
    labels: Iterable[str],
) -> PredictionResult:
    """Build :class:`PredictionResult` from a stable class-id keyed mapping."""
    values = {pony_class.value: scores[pony_class] for pony_class in PONY_CLASSES}
    return PredictionResult(**values, labels=frozenset(labels))

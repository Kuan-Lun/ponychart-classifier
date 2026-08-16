"""Backbone experiment configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ponychart_classifier.model_spec import INPUT_SIZE, ORIGINAL_IMAGE_SIZE, ImageSize
from ponychart_classifier.training import BACKBONE_REGISTRY

InputSizeMode = Literal["fixed", "matched"]

# Backbones covered by the comparison, in canonical report order.
# (MobileNetV3 variants are excluded — this sweep targets EfficientNet
# scaling. Add a key here to include another registered backbone.)
BACKBONE_KEYS: list[str] = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
]


@dataclass(frozen=True)
class BackboneExperimentConfig:
    """Per-backbone training/eval settings for the comparison run.

    *batch_size* is per-backbone because larger backbones may need a
    smaller batch to fit in GPU memory; the comparison therefore is not
    strictly batch-equivalent, which mirrors the real deployment trade-off.
    """

    input_size: ImageSize
    batch_size: int = 64


def _matched_input_size(backbone: str) -> ImageSize:
    """Scale ``ORIGINAL_IMAGE_SIZE`` to roughly the pixel count of
    *backbone*'s native ImageNet input, preserving the original aspect
    ratio (see the orig-divisor table in ``training/model.py``).

    divisor = sqrt((orig_h * orig_w) / (imagenet_h * imagenet_w))
    """
    imagenet_size = BACKBONE_REGISTRY[backbone].imagenet_size
    divisor = math.sqrt(
        (ORIGINAL_IMAGE_SIZE.height * ORIGINAL_IMAGE_SIZE.width)
        / (imagenet_size.height * imagenet_size.width)
    )
    return ImageSize(
        round(ORIGINAL_IMAGE_SIZE.height / divisor),
        round(ORIGINAL_IMAGE_SIZE.width / divisor),
    )


def backbone_config(backbone: str, mode: InputSizeMode) -> BackboneExperimentConfig:
    """Build the experiment config for *backbone* under *mode*.

    - ``"fixed"``: every backbone trains at the shared production
      ``INPUT_SIZE`` — isolates the effect of architecture choice alone
      from resolution.
    - ``"matched"``: each backbone trains at its own orig-divisor-scaled
      resolution, i.e. the resolution it would realistically ship at.
    """
    if mode == "fixed":
        return BackboneExperimentConfig(input_size=INPUT_SIZE)
    return BackboneExperimentConfig(input_size=_matched_input_size(backbone))

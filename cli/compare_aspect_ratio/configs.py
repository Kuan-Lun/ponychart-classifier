"""Aspect-ratio experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass

from ponychart_classifier.model_spec import ImageSize

# PonyChart original image dimensions: 1004 x 554 -> ratio ~1.813:1


@dataclass(frozen=True)
class AspectRatioConfig:
    """Per-aspect-ratio training/eval settings."""

    input_size: ImageSize
    description: str


# Ordered from highest total pixels to lowest so OOM surfaces early.
#
# "square_*" configs use a square crop.
# "rect_*" configs preserve the native 1004:554 aspect ratio (w:h ~1.81:1,
# landscape orientation — width > height).
#
# Each rect config is pixel-matched with a square config so the comparison
# isolates the effect of aspect ratio, not total information.
#
# Native ratio: h/w = 554/1004 ≈ 0.5518
#   Given target pixels P:  w = sqrt(P / 0.5518),  h = w * 0.5518
ASPECT_RATIO_CONFIGS: dict[str, AspectRatioConfig] = {
    "square_320": AspectRatioConfig(
        input_size=ImageSize(320, 320),
        description="Square 320x320 (~102K px)",
    ),
    "rect_238x431": AspectRatioConfig(
        input_size=ImageSize(238, 431),
        description="Rect 238x431 (native ratio, ~103K px, matched with square_320)",
    ),
    "square_224": AspectRatioConfig(
        input_size=ImageSize(224, 224),
        description="Square 224x224 (~50K px)",
    ),
    "rect_166x301": AspectRatioConfig(
        input_size=ImageSize(166, 301),
        description="Rect 166x301 (native ratio, ~50K px, matched with square_224)",
    ),
}

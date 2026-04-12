"""Resolution experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass

from ponychart_classifier.model_spec import ImageSize


@dataclass(frozen=True)
class ResolutionConfig:
    """Per-resolution training/eval settings."""

    pre_resize: ImageSize
    input_size: ImageSize


# Ordered from highest to lowest so OOM surfaces early.
RESOLUTION_CONFIGS: dict[str, ResolutionConfig] = {
    "448": ResolutionConfig(
        pre_resize=ImageSize(512, 512), input_size=ImageSize(448, 448)
    ),
    "380": ResolutionConfig(
        pre_resize=ImageSize(448, 448), input_size=ImageSize(380, 380)
    ),
    "320": ResolutionConfig(
        pre_resize=ImageSize(384, 384), input_size=ImageSize(320, 320)
    ),
    "288": ResolutionConfig(
        pre_resize=ImageSize(320, 320), input_size=ImageSize(288, 288)
    ),
    "224": ResolutionConfig(
        pre_resize=ImageSize(256, 256), input_size=ImageSize(224, 224)
    ),
}

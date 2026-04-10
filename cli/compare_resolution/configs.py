"""Resolution experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionConfig:
    """Per-resolution training/eval settings."""

    pre_resize: int
    input_size: int


# Ordered from highest to lowest so OOM surfaces early.
RESOLUTION_CONFIGS: dict[str, ResolutionConfig] = {
    "448": ResolutionConfig(pre_resize=512, input_size=448),
    "380": ResolutionConfig(pre_resize=448, input_size=380),
    "320": ResolutionConfig(pre_resize=384, input_size=320),
    "288": ResolutionConfig(pre_resize=320, input_size=288),
    "224": ResolutionConfig(pre_resize=256, input_size=224),
}

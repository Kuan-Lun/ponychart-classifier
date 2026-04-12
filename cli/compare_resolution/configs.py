"""Resolution experiment configurations.

Configs are defined as scale factors relative to the production
``INPUT_SIZE`` in :mod:`ponychart_classifier.model_spec`.
Changing the production constant automatically updates what each scale
means — no hardcoded pixel sizes to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass

from ponychart_classifier.model_spec import INPUT_SIZE, ImageSize


@dataclass(frozen=True)
class ResolutionConfig:
    """Per-resolution training/eval settings."""

    scale: float
    input_size: ImageSize


def _scaled(base: ImageSize, scale: float) -> ImageSize:
    """Scale an ``ImageSize`` by *scale*, rounding to the nearest int."""
    return ImageSize(round(base.height * scale), round(base.width * scale))


def _build_configs(scales: list[float]) -> dict[str, ResolutionConfig]:
    """Build configs from a list of scale factors (highest first for OOM)."""
    configs: dict[str, ResolutionConfig] = {}
    for s in scales:
        label = f"{s:.2f}x"
        configs[label] = ResolutionConfig(
            scale=s,
            input_size=_scaled(INPUT_SIZE, s),
        )
    return configs


# Ordered from highest to lowest so OOM surfaces early.
RESOLUTION_CONFIGS = _build_configs([1.40, 1.20, 1.00, 0.85, 0.70])

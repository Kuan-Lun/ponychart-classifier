"""Backbone experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneExperimentConfig:
    """Per-backbone training/eval settings for the comparison run.

    *batch_size* is per-backbone because larger backbones (e.g. EfficientNet-B4
    at 380x380) need a smaller batch to fit in GPU memory; the comparison
    therefore is not strictly batch-equivalent, which mirrors the real
    deployment trade-off.
    """

    input_size: int
    pre_resize: int
    batch_size: int


# Insertion order is preserved and used as the canonical ordering for
# the comparison report.  Every entry must reference a backbone that
# exists in BACKBONE_REGISTRY.
BACKBONE_CONFIGS: dict[str, BackboneExperimentConfig] = {
    "efficientnet_b0": BackboneExperimentConfig(
        input_size=320, pre_resize=384, batch_size=64
    ),
    "efficientnet_b4": BackboneExperimentConfig(
        input_size=380, pre_resize=384, batch_size=64
    ),
}

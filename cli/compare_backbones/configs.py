"""Backbone experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass

from ponychart_classifier.model_spec import INPUT_SIZE, PRE_RESIZE, ImageSize


@dataclass(frozen=True)
class BackboneExperimentConfig:
    """Per-backbone training/eval settings for the comparison run.

    *batch_size* is per-backbone because larger backbones (e.g. EfficientNet-B4)
    may need a smaller batch to fit in GPU memory; the comparison therefore
    is not strictly batch-equivalent, which mirrors the real deployment
    trade-off.

    Defaults for *input_size* and *pre_resize* come from the production
    constants in :mod:`ponychart_classifier.model_spec` so they stay in
    sync automatically.
    """

    input_size: ImageSize = INPUT_SIZE
    pre_resize: ImageSize = PRE_RESIZE
    batch_size: int = 64


# Insertion order is preserved and used as the canonical ordering for
# the comparison report.  Every entry must reference a backbone that
# exists in BACKBONE_REGISTRY.
BACKBONE_CONFIGS: dict[str, BackboneExperimentConfig] = {
    "efficientnet_b0": BackboneExperimentConfig(),
    "efficientnet_b4": BackboneExperimentConfig(),
}

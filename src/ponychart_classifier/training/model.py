"""Model building with backbone registry pattern."""

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import psutil
import torch
import torch.nn as nn
from torchvision import models

from .constants import NUM_CLASSES, ImageSize

logger = logging.getLogger(__name__)


@runtime_checkable
class FeatureClassifierModel(Protocol):
    """Protocol for models with separate feature-extractor and classifier."""

    features: nn.Module
    classifier: nn.Sequential


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for a backbone architecture."""

    name: str
    build_fn: Callable[[bool], nn.Module]
    classifier_layer_index: int
    description: str
    # Native square input size the pretrained ImageNet weights were trained
    # at. Used to derive an orig-divisor-scaled input size for a given
    # backbone — see cli.compare_backbones.
    imagenet_size: ImageSize


def _build_mobilenet_v3_small(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.mobilenet_v3_small(weights=weights))


def _build_mobilenet_v3_large(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.mobilenet_v3_large(weights=weights))


def _build_efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.efficientnet_b0(weights=weights))


def _build_efficientnet_b1(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.efficientnet_b1(weights=weights))


def _build_efficientnet_b2(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.efficientnet_b2(weights=weights))


def _build_efficientnet_b3(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.efficientnet_b3(weights=weights))


def _build_efficientnet_b4(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    return cast(nn.Module, models.efficientnet_b4(weights=weights))


# Backbone reference specs (ImageNet defaults; not used at runtime)
# Train mem is approximate — activation overhead means larger models scale slightly
# higher than the param ratio alone; use as a rough OOM risk guide.
# Orig divisor = sqrt((554 × 1004) / (ImageNet width × height)).  Divide
# both original dimensions by this value to preserve the original aspect ratio
# at roughly the same pixel count as the square ImageNet input.
#
# Backbone            ImageNet size  Orig divisor  Params   ONNX size  Train mem (×B0, approx)
# ------------------  -------------  ------------  -------  ---------  -----------------------
# mobilenet_v3_small  224×224        3.33×         2.5 M    ~4 MB      ~0.5×
# mobilenet_v3_large  224×224        3.33×         5.4 M    ~9 MB      ~1.0×
# efficientnet_b0     224×224        3.33×         5.3 M    ~11 MB     1.0× (baseline)
# efficientnet_b1     240×240        3.11×         7.8 M    ~15 MB     ~1.5×
# efficientnet_b2     260×260        2.87×         9.1 M    ~18 MB     ~1.7×
# efficientnet_b3     300×300        2.49×         12 M     ~24 MB     ~2.3×
# efficientnet_b4     380×380        1.96×         19 M     ~67 MB     ~3.6×
BACKBONE_REGISTRY: dict[str, BackboneConfig] = {
    "mobilenet_v3_small": BackboneConfig(
        name="mobilenet_v3_small",
        build_fn=_build_mobilenet_v3_small,
        classifier_layer_index=3,
        description="MobileNetV3-Small (2.5M params, ~4MB ONNX)",
        imagenet_size=ImageSize(224, 224),
    ),
    "mobilenet_v3_large": BackboneConfig(
        name="mobilenet_v3_large",
        build_fn=_build_mobilenet_v3_large,
        classifier_layer_index=3,
        description="MobileNetV3-Large (5.4M params, ~9MB ONNX)",
        imagenet_size=ImageSize(224, 224),
    ),
    "efficientnet_b0": BackboneConfig(
        name="efficientnet_b0",
        build_fn=_build_efficientnet_b0,
        classifier_layer_index=1,
        description="EfficientNet-B0 (5.3M params, ~11MB ONNX)",
        imagenet_size=ImageSize(224, 224),
    ),
    "efficientnet_b1": BackboneConfig(
        name="efficientnet_b1",
        build_fn=_build_efficientnet_b1,
        classifier_layer_index=1,
        description="EfficientNet-B1 (7.8M params, ~15MB ONNX)",
        imagenet_size=ImageSize(240, 240),
    ),
    "efficientnet_b2": BackboneConfig(
        name="efficientnet_b2",
        build_fn=_build_efficientnet_b2,
        classifier_layer_index=1,
        description="EfficientNet-B2 (9.1M params, ~18MB ONNX)",
        imagenet_size=ImageSize(260, 260),
    ),
    "efficientnet_b3": BackboneConfig(
        name="efficientnet_b3",
        build_fn=_build_efficientnet_b3,
        classifier_layer_index=1,
        description="EfficientNet-B3 (12M params, ~24MB ONNX)",
        imagenet_size=ImageSize(300, 300),
    ),
    "efficientnet_b4": BackboneConfig(
        name="efficientnet_b4",
        build_fn=_build_efficientnet_b4,
        classifier_layer_index=1,
        description="EfficientNet-B4 (19M params, ~67MB ONNX)",
        imagenet_size=ImageSize(380, 380),
    ),
}


def _extract_submodules(
    model: nn.Module,
) -> tuple[nn.Module, nn.Sequential]:
    """Extract ``features`` and ``classifier`` sub-modules.

    Raises :class:`TypeError` if *model* does not satisfy
    :class:`FeatureClassifierModel`.
    """
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(features, nn.Module) or not isinstance(classifier, nn.Sequential):
        raise TypeError(
            f"{type(model).__name__} does not satisfy FeatureClassifierModel "
            "(missing .features or .classifier)"
        )
    return features, classifier


def build_model(
    backbone: str = "mobilenet_v3_large",
    pretrained: bool = True,
) -> nn.Module:
    """Build a model with the specified backbone.

    Replaces the final classification layer for NUM_CLASSES output.
    All supported backbones satisfy :class:`FeatureClassifierModel`.
    """
    if backbone not in BACKBONE_REGISTRY:
        available = ", ".join(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {available}")

    config = BACKBONE_REGISTRY[backbone]
    model = config.build_fn(pretrained)
    _, classifier = _extract_submodules(model)

    layer_idx = config.classifier_layer_index
    in_features: int = cast(nn.Linear, classifier[layer_idx]).in_features
    classifier[layer_idx] = nn.Linear(in_features, NUM_CLASSES)

    return model


def _get_rss_bytes() -> int:
    """Return current process RSS in bytes."""
    return int(psutil.Process(os.getpid()).memory_info().rss)


def measure_training_memory(
    backbone: str,
    batch_size: int,
    input_size: ImageSize,
    device: torch.device,
) -> int:
    """Measure system RAM needed for training via a dry-run forward+backward.

    Returns 0 for CUDA (GPU VRAM is separate from system RAM).
    For MPS/CPU, performs a single training step and measures the RSS delta.
    """
    if device.type == "cuda":
        return 0

    gc.collect()
    rss_before = _get_rss_bytes()

    model = build_model(backbone=backbone, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    dummy = torch.randn(batch_size, 3, *input_size.hw(), device=device)
    target = torch.zeros(batch_size, NUM_CLASSES, device=device)

    # Measure after forward (peak: weights + optimizer states + activations)
    logits = model(dummy)
    loss = criterion(logits, target)
    rss_peak = _get_rss_bytes()

    loss.backward()
    optimizer.step()

    gc.collect()
    rss_after = _get_rss_bytes()

    # Use the higher of peak (with activations) and post-step (with gradients)
    total = max(rss_peak - rss_before, rss_after - rss_before, 0)

    # Release MPS allocator cache back to OS so that subsequent
    # psutil.virtual_memory().available readings are accurate.
    del model, optimizer, criterion, dummy, target, logits, loss
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    logger.info(
        "Measured training memory: %s MB (device=%s)",
        f"{total / 1024 / 1024:,.0f}",
        device.type,
    )
    return total

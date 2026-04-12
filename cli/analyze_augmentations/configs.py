"""Augmentation experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchvision import transforms

from ponychart_classifier.training import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
)


@dataclass(frozen=True)
class AugConfig:
    """描述一組空間增強設定。"""

    hflip: bool = False
    vflip: bool = False
    degrees: float = 0

    @property
    def description(self) -> str:
        parts: list[str] = []
        if self.hflip:
            parts.append("HFlip")
        if self.vflip:
            parts.append("VFlip")
        if self.degrees > 0:
            parts.append(f"Rot({self.degrees:.0f})")
        return " + ".join(parts) if parts else "(no spatial aug)"


def build_train_transform(cfg: AugConfig) -> transforms.Compose:
    """根據 AugConfig 建立訓練用 transform pipeline。

    非空間增強（ColorJitter, GaussianBlur, RandomErasing）皆保持一致，
    僅變動翻轉與旋轉，以確保 ablation 公平比較。
    """
    spatial: list[Any] = []
    if cfg.hflip:
        spatial.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.vflip:
        spatial.append(transforms.RandomVerticalFlip(p=0.5))
    spatial.append(
        transforms.RandomAffine(
            degrees=cfg.degrees,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        )
    )
    return transforms.Compose(
        [
            *spatial,
            transforms.RandomCrop(INPUT_SIZE.hw()),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )


# Insertion order is the canonical ordering for the comparison report.
AUG_CONFIGS: dict[str, AugConfig] = {
    "none": AugConfig(),
    "hflip": AugConfig(hflip=True),
    "vflip": AugConfig(vflip=True),
    "rot15": AugConfig(degrees=15),
    "rot45": AugConfig(degrees=45),
    "rot90": AugConfig(degrees=90),
    "current": AugConfig(hflip=True, vflip=True, degrees=90),
}

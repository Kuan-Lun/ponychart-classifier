"""Checkpoint val_f1 recomputation utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ponychart_classifier.model_spec import ImageSize

from .constants import (
    BACKBONE,
    BATCH_SIZE,
    SEED,
    VAL_SIZE,
)
from .dataset import build_cached_dataset, make_dataloader
from .device import get_device, get_performance_cpu_count
from .model import build_model
from .sampling import is_original, load_samples, prepare_balanced_samples
from .splitting import group_hash_split
from .training import evaluate

torch.serialization.add_safe_globals([ImageSize])

logger = logging.getLogger(__name__)


def recompute_checkpoint_val_f1(
    checkpoint_path: Path,
    device: torch.device | None = None,
    num_workers: int | None = None,
) -> float:
    """Recompute val_f1 on the current dataset and update the checkpoint.

    Loads the model and thresholds from the checkpoint, builds the
    validation set using the same split logic as training, evaluates,
    and writes the updated val_f1 back to the checkpoint file.

    Returns the computed val_f1.
    """
    if device is None:
        device = get_device()
    if num_workers is None:
        num_workers = get_performance_cpu_count()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Build model and load weights
    backbone: str = ckpt.get("backbone", BACKBONE)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    thresholds: list[float] = ckpt["thresholds"]

    # Build validation set (same logic as train.py)
    samples = load_samples()
    rng = np.random.RandomState(SEED)
    balanced_samples = prepare_balanced_samples(samples, rng)
    _, val_idx = group_hash_split(balanced_samples, test_size=VAL_SIZE)
    val_samples = [
        s
        for s in (balanced_samples[i] for i in val_idx)
        if is_original(os.path.basename(s[0]))
    ]

    # Evaluate
    val_dataset = build_cached_dataset(val_samples, is_train=False)
    val_loader = make_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss()
    result = evaluate(model, val_loader, criterion, device, thresholds=thresholds)
    val_f1 = result.macro_f1

    # Update checkpoint
    ckpt["val_f1"] = val_f1
    ckpt["per_class_f1"] = result.per_class_f1
    torch.save(ckpt, checkpoint_path)
    logger.info("Updated checkpoint val_f1: %.4f (%s)", val_f1, checkpoint_path)

    return val_f1

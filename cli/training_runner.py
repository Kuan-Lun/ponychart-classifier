"""Composable training-experiment pipeline.

Provides the shared infrastructure used by training-based comparison CLIs
(``compare_resolution``, ``compare_backbones``).  The benchmark CLI does
**not** use this module — it has its own lighter-weight flow.

Design: pure functions + frozen dataclasses, no inheritance.  CLIs
compose with these by calling :func:`prepare_training_setup` and
:func:`run_training_experiment`.
"""

from __future__ import annotations

import logging
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from ponychart_classifier.training import (
    SEED,
    EvalResult,
    HoldoutExperimentSetup,
    ImageSize,
    Sample,
    build_cached_dataset,
    describe_device,
    evaluate,
    export_onnx,
    make_dataloader,
    prepare_holdout_setup_logged,
    seed_all,
    train_model,
)

# ------------------------------------------------------------------
# Utility functions (previously duplicated in both compare scripts)
# ------------------------------------------------------------------


def get_onnx_size_mb(model: nn.Module, input_size: ImageSize) -> float:
    """Export *model* to a temp ONNX file and return its size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_onnx(model, tmp_path, input_size=input_size)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    finally:
        tmp_path.unlink(missing_ok=True)
        data_path = Path(str(tmp_path) + ".data")
        data_path.unlink(missing_ok=True)
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


TrainingSetup = HoldoutExperimentSetup


def prepare_training_setup(logger: logging.Logger) -> TrainingSetup:
    """Build the standard logged holdout setup for experiment CLIs."""
    return prepare_holdout_setup_logged(logger)


# ------------------------------------------------------------------
# Training experiment execution
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingMeasurements:
    """Raw measurements from a single training experiment.

    The caller maps these into its own experiment-specific result type
    (which may carry additional config fields like ``backbone_name``).
    """

    test_result: EvalResult
    thresholds: list[float]
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    hostname: str
    device_label: str


def run_training_experiment(
    *,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    device: torch.device,
    num_workers: int,
    run_label: str,
    backbone: str,
    input_size: ImageSize,
    batch_size: int,
    train_transform: transforms.Compose,
) -> TrainingMeasurements:
    """Train a model, evaluate on test set, and measure ONNX export.

    The caller is responsible for experiment-specific logging (section
    headers, config details) **before** calling this function.

    Parameters match :func:`~ponychart_classifier.training.train_model`
    plus the test-set evaluation and ONNX measurement steps.
    """
    t0 = time.monotonic()
    seed_all(SEED)
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        run_label,
        backbone=backbone,
        train_transform=train_transform,
        input_size=input_size,
        batch_size=batch_size,
    )
    train_time = time.monotonic() - t0

    model, thresholds = result.model, result.thresholds
    criterion = nn.BCEWithLogitsLoss()
    test_ds = build_cached_dataset(
        test_samples,
        is_train=False,
        image_size=input_size,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    test_result = evaluate(model, test_loader, criterion, device, thresholds)

    param_count = count_parameters(model)
    onnx_size = get_onnx_size_mb(model, input_size=input_size)

    return TrainingMeasurements(
        test_result=test_result,
        thresholds=list(thresholds),
        param_count=param_count,
        onnx_size_mb=onnx_size,
        train_time_s=train_time,
        hostname=socket.gethostname(),
        device_label=describe_device(device),
    )

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
from typing import TypedDict

import torch
import torch.nn as nn
from torchvision import transforms

from ponychart_classifier.training import (
    HASH_PREFIX_LEN,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    EvalResult,
    HoldoutSplit,
    ImageSize,
    Sample,
    build_cached_dataset,
    describe_device,
    evaluate,
    export_onnx,
    hash_samples,
    load_samples_logged,
    make_dataloader,
    prepare_holdout_split,
    seed_all,
    setup_device_and_workers,
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


# ------------------------------------------------------------------
# Shared TypedDicts for JSON serialization
# ------------------------------------------------------------------


class TestResultDict(TypedDict):
    """JSON shape of an :class:`~ponychart_classifier.training.EvalResult`."""

    loss: float
    macro_f1: float
    per_class_f1: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]


class SplitDict(TypedDict):
    """JSON shape of the holdout split sizes."""

    train_size: int
    val_size: int
    test_size: int


def eval_result_to_dict(tr: EvalResult) -> TestResultDict:
    """Serialize an :class:`EvalResult` into a :class:`TestResultDict`."""
    return TestResultDict(
        loss=tr.loss,
        macro_f1=tr.macro_f1,
        per_class_f1=list(tr.per_class_f1),
        per_class_precision=list(tr.per_class_precision),
        per_class_recall=list(tr.per_class_recall),
    )


def eval_result_from_dict(d: TestResultDict) -> EvalResult:
    """Deserialize a :class:`TestResultDict` into an :class:`EvalResult`."""
    return EvalResult(
        loss=d["loss"],
        macro_f1=d["macro_f1"],
        per_class_f1=list(d["per_class_f1"]),
        per_class_precision=list(d["per_class_precision"]),
        per_class_recall=list(d["per_class_recall"]),
    )


# ------------------------------------------------------------------
# Training setup
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSetup:
    """Everything needed before running a training experiment."""

    device: torch.device
    num_workers: int
    split: HoldoutSplit
    data_hash: str


def prepare_training_setup(logger: logging.Logger) -> TrainingSetup:
    """Seed -> device -> load samples -> hash -> holdout split.

    This is the shared pipeline that was duplicated in both compare
    scripts' ``cmd_run`` functions.
    """
    rng = seed_all(SEED)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_logged(logger)
    data_hash = hash_samples(all_samples)
    logger.info(
        "Data fingerprint: %s (full=%s)", data_hash[:HASH_PREFIX_LEN], data_hash
    )
    hs = prepare_holdout_split(
        all_samples, rng, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )
    return TrainingSetup(
        device=device, num_workers=num_workers, split=hs, data_hash=data_hash
    )


# ------------------------------------------------------------------
# Training experiment execution
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingMeasurements:
    """Raw measurements from a single training experiment.

    The caller maps these into its own experiment-specific result type
    (which may carry additional config fields like ``pre_resize`` or
    ``backbone_name``).
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
    pre_resize: ImageSize,
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
        pre_resize=pre_resize,
        batch_size=batch_size,
    )
    train_time = time.monotonic() - t0

    model, thresholds = result.model, result.thresholds
    criterion = nn.BCEWithLogitsLoss()
    test_ds = build_cached_dataset(
        test_samples,
        is_train=False,
        input_size=input_size,
        pre_resize=pre_resize,
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

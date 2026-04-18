"""Shared JSON schema helpers for experiment CLI result files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict, cast

from cli.training_runner import TrainingMeasurements, TrainingSetup
from ponychart_classifier.training import EvalResult


class HoldoutSplitDict(TypedDict):
    """JSON shape for train/val/test split sizes."""

    train_size: int
    val_size: int
    test_size: int


class TrainValSplitDict(TypedDict):
    """JSON shape for train/val split sizes."""

    train_size: int
    val_size: int


class TestResultDict(TypedDict):
    """JSON shape of an evaluated holdout test result."""

    loss: float
    macro_f1: float
    per_class_f1: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]


@dataclass(frozen=True)
class HoldoutResultMetadata:
    """Shared metadata carried by one holdout-style CLI result."""

    thresholds: list[float]
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    train_size: int
    val_size: int
    test_size: int
    seed: int
    data_hash: str
    hostname: str
    device: str


def eval_result_to_dict(tr: EvalResult) -> TestResultDict:
    """Serialize an ``EvalResult`` into a JSON-friendly dict."""
    return TestResultDict(
        loss=tr.loss,
        macro_f1=tr.macro_f1,
        per_class_f1=list(tr.per_class_f1),
        per_class_precision=list(tr.per_class_precision),
        per_class_recall=list(tr.per_class_recall),
    )


def eval_result_from_dict(d: TestResultDict) -> EvalResult:
    """Deserialize a test-result dict into an ``EvalResult``."""
    return EvalResult(
        loss=d["loss"],
        macro_f1=d["macro_f1"],
        per_class_f1=list(d["per_class_f1"]),
        per_class_precision=list(d["per_class_precision"]),
        per_class_recall=list(d["per_class_recall"]),
    )


def parse_json_object(raw: str) -> dict[str, object]:
    """Parse one JSON object and reject any non-object top-level values."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(dict[str, object], parsed)


def build_holdout_metadata(
    measurements: TrainingMeasurements,
    setup: TrainingSetup,
    *,
    seed: int,
) -> HoldoutResultMetadata:
    """Build common metadata from shared training measurements."""
    return HoldoutResultMetadata(
        thresholds=list(measurements.thresholds),
        param_count=measurements.param_count,
        onnx_size_mb=measurements.onnx_size_mb,
        train_time_s=measurements.train_time_s,
        train_size=len(setup.split.train),
        val_size=len(setup.split.val),
        test_size=len(setup.split.test),
        seed=seed,
        data_hash=setup.data_hash,
        hostname=measurements.hostname,
        device=measurements.device_label,
    )

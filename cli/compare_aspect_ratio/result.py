"""ExperimentResult type and JSON serialization for aspect-ratio experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from cli.result_utils import (
    HoldoutSplitDict,
    TestResultDict,
    build_holdout_metadata,
    eval_result_from_dict,
    eval_result_to_dict,
    parse_json_object,
)
from cli.training_runner import TrainingMeasurements, TrainingSetup
from ponychart_classifier.training import (
    HASH_PREFIX_LEN,
    SEED,
    EnvDict,
    EvalResult,
    ImageSize,
)

from .configs import AspectRatioConfig

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single aspect-ratio experiment."""

    label: str
    input_size: ImageSize
    description: str
    test_result: EvalResult
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


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class ExperimentDict(TypedDict):
    label: str
    input_size: list[int]
    description: str
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    thresholds: list[float]
    test_result: TestResultDict
    split: HoldoutSplitDict
    seed: int
    data_hash: str
    env: EnvDict


def experiment_to_dict(exp: ExperimentResult) -> ExperimentDict:
    return ExperimentDict(
        label=exp.label,
        input_size=list(exp.input_size.hw()),
        description=exp.description,
        param_count=exp.param_count,
        onnx_size_mb=exp.onnx_size_mb,
        train_time_s=exp.train_time_s,
        thresholds=list(exp.thresholds),
        test_result=eval_result_to_dict(exp.test_result),
        split=HoldoutSplitDict(
            train_size=exp.train_size,
            val_size=exp.val_size,
            test_size=exp.test_size,
        ),
        seed=exp.seed,
        data_hash=exp.data_hash,
        env=EnvDict(hostname=exp.hostname, device=exp.device),
    )


def _to_image_size(v: list[int]) -> ImageSize:
    return ImageSize(v[0], v[1])


def experiment_from_dict(data: ExperimentDict) -> ExperimentResult:
    split = data["split"]
    env = data["env"]
    return ExperimentResult(
        label=data["label"],
        input_size=_to_image_size(data["input_size"]),
        description=data["description"],
        test_result=eval_result_from_dict(data["test_result"]),
        thresholds=list(data["thresholds"]),
        param_count=data["param_count"],
        onnx_size_mb=data["onnx_size_mb"],
        train_time_s=data["train_time_s"],
        train_size=split["train_size"],
        val_size=split["val_size"],
        test_size=split["test_size"],
        seed=data["seed"],
        data_hash=data["data_hash"],
        hostname=env["hostname"],
        device=env["device"],
    )


def _parse_experiment_json(raw: str) -> ExperimentDict:
    return cast(ExperimentDict, parse_json_object(raw))


def result_filename(label: str, data_hash: str) -> str:
    return f"{label}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(exp: ExperimentResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(exp.label, exp.data_hash)
    out_path.write_text(json.dumps(experiment_to_dict(exp), indent=2))
    return out_path


def parse_result_file(raw: str) -> ExperimentResult:
    return experiment_from_dict(_parse_experiment_json(raw))


def measurements_to_result(
    label: str,
    config: AspectRatioConfig,
    m: TrainingMeasurements,
    setup: TrainingSetup,
) -> ExperimentResult:
    metadata = build_holdout_metadata(m, setup, seed=SEED)
    return ExperimentResult(
        label=label,
        input_size=config.input_size,
        description=config.description,
        test_result=m.test_result,
        thresholds=metadata.thresholds,
        param_count=metadata.param_count,
        onnx_size_mb=metadata.onnx_size_mb,
        train_time_s=metadata.train_time_s,
        train_size=metadata.train_size,
        val_size=metadata.val_size,
        test_size=metadata.test_size,
        seed=metadata.seed,
        data_hash=metadata.data_hash,
        hostname=metadata.hostname,
        device=metadata.device,
    )

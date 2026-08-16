"""ExperimentResult type and JSON serialization for backbone experiments."""

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
    BACKBONE_REGISTRY,
    HASH_PREFIX_LEN,
    SEED,
    EnvDict,
    EvalResult,
    ImageSize,
)

from .configs import BackboneExperimentConfig, InputSizeMode

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single backbone experiment."""

    backbone_name: str
    description: str
    test_result: EvalResult
    thresholds: list[float]
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    input_size: ImageSize
    input_size_mode: InputSizeMode
    batch_size: int
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
    backbone_name: str
    description: str
    input_size: list[int]
    input_size_mode: InputSizeMode
    batch_size: int
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    thresholds: list[float]
    test_result: TestResultDict
    split: HoldoutSplitDict
    seed: int
    data_hash: str
    env: EnvDict


def experiment_to_dict(experiment: ExperimentResult) -> ExperimentDict:
    return ExperimentDict(
        backbone_name=experiment.backbone_name,
        description=experiment.description,
        input_size=list(experiment.input_size.hw()),
        input_size_mode=experiment.input_size_mode,
        batch_size=experiment.batch_size,
        param_count=experiment.param_count,
        onnx_size_mb=experiment.onnx_size_mb,
        train_time_s=experiment.train_time_s,
        thresholds=list(experiment.thresholds),
        test_result=eval_result_to_dict(experiment.test_result),
        split=HoldoutSplitDict(
            train_size=experiment.train_size,
            val_size=experiment.val_size,
            test_size=experiment.test_size,
        ),
        seed=experiment.seed,
        data_hash=experiment.data_hash,
        env=EnvDict(hostname=experiment.hostname, device=experiment.device),
    )


def _to_image_size(v: list[int]) -> ImageSize:
    return ImageSize(v[0], v[1])


def experiment_from_dict(data: ExperimentDict) -> ExperimentResult:
    split = data["split"]
    env = data["env"]
    return ExperimentResult(
        backbone_name=data["backbone_name"],
        description=data["description"],
        test_result=eval_result_from_dict(data["test_result"]),
        thresholds=list(data["thresholds"]),
        param_count=data["param_count"],
        onnx_size_mb=data["onnx_size_mb"],
        train_time_s=data["train_time_s"],
        input_size=_to_image_size(data["input_size"]),
        input_size_mode=data["input_size_mode"],
        batch_size=data["batch_size"],
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


def result_filename(backbone_name: str, mode: InputSizeMode, data_hash: str) -> str:
    return f"{backbone_name}__{mode}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(experiment: ExperimentResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(
        experiment.backbone_name, experiment.input_size_mode, experiment.data_hash
    )
    out_path.write_text(json.dumps(experiment_to_dict(experiment), indent=2))
    return out_path


def parse_result_file(raw: str) -> ExperimentResult:
    return experiment_from_dict(_parse_experiment_json(raw))


def measurements_to_result(
    backbone_name: str,
    mode: InputSizeMode,
    config: BackboneExperimentConfig,
    m: TrainingMeasurements,
    setup: TrainingSetup,
) -> ExperimentResult:
    backbone_meta = BACKBONE_REGISTRY[backbone_name]
    metadata = build_holdout_metadata(m, setup, seed=SEED)
    return ExperimentResult(
        backbone_name=backbone_name,
        description=backbone_meta.description,
        test_result=m.test_result,
        thresholds=metadata.thresholds,
        param_count=metadata.param_count,
        onnx_size_mb=metadata.onnx_size_mb,
        train_time_s=metadata.train_time_s,
        input_size=config.input_size,
        input_size_mode=mode,
        batch_size=config.batch_size,
        train_size=metadata.train_size,
        val_size=metadata.val_size,
        test_size=metadata.test_size,
        seed=metadata.seed,
        data_hash=metadata.data_hash,
        hostname=metadata.hostname,
        device=metadata.device,
    )

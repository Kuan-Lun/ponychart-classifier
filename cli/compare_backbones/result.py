"""ExperimentResult type and JSON serialization for backbone experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from cli.training_runner import (
    SplitDict,
    TestResultDict,
    TrainingMeasurements,
    TrainingSetup,
    eval_result_from_dict,
    eval_result_to_dict,
)
from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    HASH_PREFIX_LEN,
    SEED,
    EnvDict,
    EvalResult,
)

from .configs import BackboneExperimentConfig

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
    input_size: int
    pre_resize: int
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
    input_size: int
    pre_resize: int
    batch_size: int
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    thresholds: list[float]
    test_result: TestResultDict
    split: SplitDict
    seed: int
    data_hash: str
    env: EnvDict


def experiment_to_dict(experiment: ExperimentResult) -> ExperimentDict:
    return ExperimentDict(
        backbone_name=experiment.backbone_name,
        description=experiment.description,
        input_size=experiment.input_size,
        pre_resize=experiment.pre_resize,
        batch_size=experiment.batch_size,
        param_count=experiment.param_count,
        onnx_size_mb=experiment.onnx_size_mb,
        train_time_s=experiment.train_time_s,
        thresholds=list(experiment.thresholds),
        test_result=eval_result_to_dict(experiment.test_result),
        split=SplitDict(
            train_size=experiment.train_size,
            val_size=experiment.val_size,
            test_size=experiment.test_size,
        ),
        seed=experiment.seed,
        data_hash=experiment.data_hash,
        env=EnvDict(hostname=experiment.hostname, device=experiment.device),
    )


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
        input_size=data["input_size"],
        pre_resize=data["pre_resize"],
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
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(ExperimentDict, parsed)


def result_filename(backbone_name: str, data_hash: str) -> str:
    return f"{backbone_name}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(experiment: ExperimentResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(
        experiment.backbone_name, experiment.data_hash
    )
    out_path.write_text(json.dumps(experiment_to_dict(experiment), indent=2))
    return out_path


def parse_result_file(raw: str) -> ExperimentResult:
    return experiment_from_dict(_parse_experiment_json(raw))


def measurements_to_result(
    backbone_name: str,
    config: BackboneExperimentConfig,
    m: TrainingMeasurements,
    setup: TrainingSetup,
) -> ExperimentResult:
    backbone_meta = BACKBONE_REGISTRY[backbone_name]
    return ExperimentResult(
        backbone_name=backbone_name,
        description=backbone_meta.description,
        test_result=m.test_result,
        thresholds=m.thresholds,
        param_count=m.param_count,
        onnx_size_mb=m.onnx_size_mb,
        train_time_s=m.train_time_s,
        input_size=config.input_size,
        pre_resize=config.pre_resize,
        batch_size=config.batch_size,
        train_size=len(setup.split.train),
        val_size=len(setup.split.val),
        test_size=len(setup.split.test),
        seed=SEED,
        data_hash=setup.data_hash,
        hostname=m.hostname,
        device=m.device_label,
    )

"""ExperimentResult type and JSON serialization for augmentation experiments."""

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
    HASH_PREFIX_LEN,
    SEED,
    EnvDict,
    EvalResult,
)

from .configs import AugConfig

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single augmentation experiment."""

    label: str
    hflip: bool
    vflip: bool
    degrees: float
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


class AugDict(TypedDict):
    hflip: bool
    vflip: bool
    degrees: float


class ExperimentDict(TypedDict):
    label: str
    augmentation: AugDict
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    thresholds: list[float]
    test_result: TestResultDict
    split: SplitDict
    seed: int
    data_hash: str
    env: EnvDict


def experiment_to_dict(exp: ExperimentResult) -> ExperimentDict:
    return ExperimentDict(
        label=exp.label,
        augmentation=AugDict(
            hflip=exp.hflip,
            vflip=exp.vflip,
            degrees=exp.degrees,
        ),
        param_count=exp.param_count,
        onnx_size_mb=exp.onnx_size_mb,
        train_time_s=exp.train_time_s,
        thresholds=list(exp.thresholds),
        test_result=eval_result_to_dict(exp.test_result),
        split=SplitDict(
            train_size=exp.train_size,
            val_size=exp.val_size,
            test_size=exp.test_size,
        ),
        seed=exp.seed,
        data_hash=exp.data_hash,
        env=EnvDict(hostname=exp.hostname, device=exp.device),
    )


def experiment_from_dict(data: ExperimentDict) -> ExperimentResult:
    split = data["split"]
    env = data["env"]
    aug = data["augmentation"]
    return ExperimentResult(
        label=data["label"],
        hflip=aug["hflip"],
        vflip=aug["vflip"],
        degrees=aug["degrees"],
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
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(ExperimentDict, parsed)


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
    config: AugConfig,
    m: TrainingMeasurements,
    setup: TrainingSetup,
) -> ExperimentResult:
    return ExperimentResult(
        label=label,
        hflip=config.hflip,
        vflip=config.vflip,
        degrees=config.degrees,
        test_result=m.test_result,
        thresholds=m.thresholds,
        param_count=m.param_count,
        onnx_size_mb=m.onnx_size_mb,
        train_time_s=m.train_time_s,
        train_size=len(setup.split.train),
        val_size=len(setup.split.val),
        test_size=len(setup.split.test),
        seed=SEED,
        data_hash=setup.data_hash,
        hostname=m.hostname,
        device=m.device_label,
    )

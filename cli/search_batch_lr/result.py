"""SearchResult type and JSON serialization for batch-LR search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from cli.result_utils import TrainValSplitDict, parse_json_object
from ponychart_classifier.training import HASH_PREFIX_LEN, EnvDict

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Result of a single batch-size search experiment."""

    batch_size: int
    lr_scale: float
    lr_head: float
    lr_features: float
    lr_classifier: float
    best_f1: float
    per_class_f1: list[float]
    phase1_stopped_epoch: int
    stopped_epoch: int
    time_s: float
    train_size: int
    val_size: int
    seed: int
    backbone: str
    data_hash: str
    hostname: str
    device: str


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class SearchResultDict(TypedDict):
    batch_size: int
    lr_scale: float
    lr_head: float
    lr_features: float
    lr_classifier: float
    best_f1: float
    per_class_f1: list[float]
    phase1_stopped_epoch: int
    stopped_epoch: int
    time_s: float
    split: TrainValSplitDict
    seed: int
    backbone: str
    data_hash: str
    env: EnvDict


def result_to_dict(result: SearchResult) -> SearchResultDict:
    return SearchResultDict(
        batch_size=result.batch_size,
        lr_scale=result.lr_scale,
        lr_head=result.lr_head,
        lr_features=result.lr_features,
        lr_classifier=result.lr_classifier,
        best_f1=result.best_f1,
        per_class_f1=list(result.per_class_f1),
        phase1_stopped_epoch=result.phase1_stopped_epoch,
        stopped_epoch=result.stopped_epoch,
        time_s=result.time_s,
        split=TrainValSplitDict(
            train_size=result.train_size,
            val_size=result.val_size,
        ),
        seed=result.seed,
        backbone=result.backbone,
        data_hash=result.data_hash,
        env=EnvDict(hostname=result.hostname, device=result.device),
    )


def result_from_dict(data: SearchResultDict) -> SearchResult:
    split = data["split"]
    env = data["env"]
    return SearchResult(
        batch_size=data["batch_size"],
        lr_scale=data["lr_scale"],
        lr_head=data["lr_head"],
        lr_features=data["lr_features"],
        lr_classifier=data["lr_classifier"],
        best_f1=data["best_f1"],
        per_class_f1=list(data["per_class_f1"]),
        phase1_stopped_epoch=data["phase1_stopped_epoch"],
        stopped_epoch=data["stopped_epoch"],
        time_s=data["time_s"],
        train_size=split["train_size"],
        val_size=split["val_size"],
        seed=data["seed"],
        backbone=data["backbone"],
        data_hash=data["data_hash"],
        hostname=env["hostname"],
        device=env["device"],
    )


def _parse_result_json(raw: str) -> SearchResultDict:
    return cast(SearchResultDict, parse_json_object(raw))


def result_filename(batch_size: int, data_hash: str) -> str:
    """Return the canonical JSON filename for a (batch_size, data) pair."""
    return f"batch{batch_size:03d}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(result: SearchResult, results_dir: Path) -> Path:
    """Write *result* to ``<results_dir>/batch{bs}__{hash12}.json``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(result.batch_size, result.data_hash)
    out_path.write_text(json.dumps(result_to_dict(result), indent=2))
    return out_path


def parse_result_file(raw: str) -> SearchResult:
    """Parse one JSON document into a :class:`SearchResult`."""
    return result_from_dict(_parse_result_json(raw))

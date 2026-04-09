"""
比較不同 backbone 架構對 PonyChart 分類效果的影響。

此腳本拆成兩種模式，方便把不同 backbone 派到不同設備上跑：

1. 單跑一個 backbone 並把結果寫成 JSON：
     uv run python scripts/compare_backbones.py --run efficientnet_b0
     uv run python scripts/compare_backbones.py --run efficientnet_b4

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run python scripts/compare_backbones.py --report

預設 results dir 是 ``scripts/backbone_results/``，可用 ``--results-dir``
指定其他路徑。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``<backbone>__<hash12>.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。

要加新的 backbone，只要在 ``BACKBONE_CONFIGS`` 補一筆即可。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    EvalResult,
    HoldoutSplit,
    Sample,
    build_cached_dataset,
    evaluate,
    export_onnx,
    load_samples_logged,
    log_section,
    make_dataloader,
    prepare_holdout_split,
    seed_all,
    setup_device_and_workers,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "backbone_results"
HASH_PREFIX_LEN = 12  # short hash for filenames; 48 bits, ample for our scale


# ---------------------------------------------------------------------------
# Backbone configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneExperimentConfig:
    """Per-backbone training/eval settings for the comparison run.

    *batch_size* is per-backbone because larger backbones (e.g. EfficientNet-B4
    at 380×380) need a smaller batch to fit in GPU memory; the comparison
    therefore is not strictly batch-equivalent, which mirrors the real
    deployment trade-off.
    """

    input_size: int
    pre_resize: int
    batch_size: int


# Insertion order is preserved and used as the canonical ordering for
# the comparison report. Add backbones here (and only here) to enable them.
# Every entry must reference a backbone that exists in BACKBONE_REGISTRY.
BACKBONE_CONFIGS: dict[str, BackboneExperimentConfig] = {
    "efficientnet_b0": BackboneExperimentConfig(
        input_size=320, pre_resize=384, batch_size=64
    ),
    "efficientnet_b4": BackboneExperimentConfig(
        input_size=380, pre_resize=384, batch_size=64
    ),
}


# ---------------------------------------------------------------------------
# Data fingerprinting
# ---------------------------------------------------------------------------
def hash_samples(samples: list[Sample]) -> str:
    """Return a stable SHA-256 hex digest of the sample set.

    Two machines that load the exact same ``rawimage`` + ``labels.json``
    will produce the same hash. Any addition, deletion, or label change
    flips the hash, which makes it safe to use as a "data snapshot ID"
    for grouping experiment results.
    """
    h = hashlib.sha256()
    for path, labels in sorted(samples):
        h.update(path.encode("utf-8"))
        h.update(b"\x00")
        h.update(",".join(str(label) for label in labels).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Result types
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
class TestResultDict(TypedDict):
    """JSON shape of an :class:`EvalResult`."""

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


class EnvDict(TypedDict):
    """JSON shape of the run environment metadata."""

    hostname: str
    device: str


class ExperimentDict(TypedDict):
    """JSON shape of an :class:`ExperimentResult`."""

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
    """Convert an :class:`ExperimentResult` to a JSON-serializable dict."""
    tr = experiment.test_result
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
        test_result=TestResultDict(
            loss=tr.loss,
            macro_f1=tr.macro_f1,
            per_class_f1=list(tr.per_class_f1),
            per_class_precision=list(tr.per_class_precision),
            per_class_recall=list(tr.per_class_recall),
        ),
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
    """Reconstruct an :class:`ExperimentResult` from an :class:`ExperimentDict`."""
    tr_dict = data["test_result"]
    split = data["split"]
    env = data["env"]
    test_result = EvalResult(
        loss=tr_dict["loss"],
        macro_f1=tr_dict["macro_f1"],
        per_class_f1=list(tr_dict["per_class_f1"]),
        per_class_precision=list(tr_dict["per_class_precision"]),
        per_class_recall=list(tr_dict["per_class_recall"]),
    )
    return ExperimentResult(
        backbone_name=data["backbone_name"],
        description=data["description"],
        test_result=test_result,
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
    """Parse a JSON document into a validated :class:`ExperimentDict`.

    This is the only place where untyped JSON crosses into typed code.
    Any structural mismatch raises ``ValueError``/``KeyError``/``TypeError``,
    which the caller is expected to handle.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(ExperimentDict, parsed)


def result_filename(backbone_name: str, data_hash: str) -> str:
    """Return the canonical JSON filename for a (backbone, data) pair."""
    return f"{backbone_name}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(experiment: ExperimentResult, results_dir: Path) -> Path:
    """Write *experiment* to ``<results_dir>/<backbone>__<hash12>.json``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(
        experiment.backbone_name, experiment.data_hash
    )
    out_path.write_text(json.dumps(experiment_to_dict(experiment), indent=2))
    return out_path


def load_all_results(results_dir: Path) -> list[ExperimentResult]:
    """Load every ``*.json`` in *results_dir* into ``ExperimentResult``s.

    Raises :class:`FileNotFoundError` if *results_dir* is missing, or
    :class:`RuntimeError` if any result file is unparseable. We deliberately
    fail fast rather than skipping bad files: a partial report would silently
    omit results and might push the reader to a wrong conclusion.
    """
    if not results_dir.is_dir():
        msg = f"Results directory does not exist: {results_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    results: list[ExperimentResult] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = _parse_experiment_json(path.read_text())
            experiment = experiment_from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            msg = (
                f"Failed to load result file {path}: {exc}\n"
                "  This may be a stale result from an older script version. "
                "Delete it (or move it elsewhere) and re-run the corresponding "
                "--run, then try --report again."
            )
            for line in msg.splitlines():
                logger.error(line)
            raise RuntimeError(msg) from exc
        results.append(experiment)
    return results


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def get_onnx_size_mb(model: nn.Module, input_size: int) -> float:
    """Export model to a temp ONNX file and return its size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_onnx(model, tmp_path, input_size=input_size)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    finally:
        tmp_path.unlink(missing_ok=True)
        # Clean up potential external data file
        data_path = Path(str(tmp_path) + ".data")
        data_path.unlink(missing_ok=True)
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def describe_device(device: torch.device) -> str:
    """Return a human-readable device label, e.g. ``cuda:0 (NVIDIA H100)``."""
    device_type = str(device.type)
    if device_type == "cuda":
        idx = device.index if device.index is not None else 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return device_type


def run_experiment(
    backbone_name: str,
    config: BackboneExperimentConfig,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    device: torch.device,
    num_workers: int,
    *,
    data_hash: str,
) -> ExperimentResult:
    """Train one backbone and evaluate on the test set.

    Heavy objects (model, dataset, DataLoader workers) are freed
    automatically when this function returns.
    """
    backbone_meta = BACKBONE_REGISTRY[backbone_name]
    log_section(logger, "BACKBONE: %s", backbone_meta.description, width=70)
    logger.info(
        "  Resolution: input=%d  pre_resize=%d  batch_size=%d",
        config.input_size,
        config.pre_resize,
        config.batch_size,
    )
    logger.info("  Data hash: %s", data_hash[:HASH_PREFIX_LEN])

    t0 = time.monotonic()
    seed_all(SEED)
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        backbone_name,
        backbone=backbone_name,
        input_size=config.input_size,
        pre_resize=config.pre_resize,
        batch_size=config.batch_size,
    )
    train_time = time.monotonic() - t0

    # Evaluate on test set
    model, thresholds = result.model, result.thresholds
    criterion = nn.BCEWithLogitsLoss()
    test_ds = build_cached_dataset(
        test_samples,
        is_train=False,
        input_size=config.input_size,
        pre_resize=config.pre_resize,
    )
    test_loader = make_dataloader(
        test_ds,
        config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    test_result = evaluate(model, test_loader, criterion, device, thresholds)

    # Model stats
    param_count = count_parameters(model)
    onnx_size = get_onnx_size_mb(model, input_size=config.input_size)

    experiment = ExperimentResult(
        backbone_name=backbone_name,
        description=backbone_meta.description,
        test_result=test_result,
        thresholds=thresholds,
        param_count=param_count,
        onnx_size_mb=onnx_size,
        train_time_s=train_time,
        input_size=config.input_size,
        pre_resize=config.pre_resize,
        batch_size=config.batch_size,
        train_size=len(train_samples),
        val_size=len(val_samples),
        test_size=len(test_samples),
        seed=SEED,
        data_hash=data_hash,
        hostname=socket.gethostname(),
        device=describe_device(device),
    )

    logger.info(
        ">> %s @ %d: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
        backbone_name,
        config.input_size,
        test_result.macro_f1,
        param_count // 1000,
        onnx_size,
        train_time,
    )
    return experiment


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_run(backbone_name: str, results_dir: Path) -> None:
    """Train one backbone and persist the result to *results_dir*."""
    if backbone_name not in BACKBONE_CONFIGS:
        available = ", ".join(BACKBONE_CONFIGS.keys())
        msg = (
            f"Backbone {backbone_name!r} not configured. Add it to "
            f"BACKBONE_CONFIGS first. Currently configured: {available}"
        )
        logger.error(msg)
        raise ValueError(msg)
    config = BACKBONE_CONFIGS[backbone_name]

    rng = seed_all(SEED)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_logged(logger)
    data_hash = hash_samples(all_samples)
    logger.info(
        "Data fingerprint: %s (full=%s)", data_hash[:HASH_PREFIX_LEN], data_hash
    )

    hs: HoldoutSplit = prepare_holdout_split(
        all_samples, rng, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )

    experiment = run_experiment(
        backbone_name,
        config,
        hs.train,
        hs.val,
        hs.test,
        device,
        num_workers,
        data_hash=data_hash,
    )

    out_path = save_result(experiment, results_dir)
    logger.info("Saved result to %s", out_path)


def _select_consistent_results(
    results: list[ExperimentResult],
) -> dict[str, ExperimentResult]:
    """Group results by ``backbone_name`` after enforcing a single data hash.

    Aborts the program if results in *results* come from more than one
    data snapshot, or if any (backbone, hash) pair is duplicated.
    """
    hashes_to_backbones: dict[str, list[str]] = {}
    for r in results:
        hashes_to_backbones.setdefault(r.data_hash, []).append(r.backbone_name)

    if len(hashes_to_backbones) > 1:
        lines = [
            f"Loaded results were trained on {len(hashes_to_backbones)} "
            "different data snapshots — refusing to produce a misleading "
            "comparison.",
        ]
        for h, names in sorted(hashes_to_backbones.items()):
            lines.append(f"  {h[:HASH_PREFIX_LEN]}: {', '.join(sorted(names))}")
        lines.append(
            "Resolution: keep only the JSONs from one data snapshot in the "
            "results dir, then re-run --report."
        )
        for line in lines:
            logger.error(line)
        raise RuntimeError("\n".join(lines))

    by_backbone: dict[str, ExperimentResult] = {}
    for r in results:
        if r.backbone_name in by_backbone:
            msg = (
                f"Duplicate result for backbone {r.backbone_name!r} at the "
                "same data hash. This shouldn't happen via --run; remove one "
                "of the JSON files."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        by_backbone[r.backbone_name] = r
    return by_backbone


def _ordered_backbones(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded backbone names in canonical (config) order, then extras."""
    canonical = [name for name in BACKBONE_CONFIGS if name in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


def cmd_report(results_dir: Path) -> None:
    """Print a comparison table from all JSON files in *results_dir*."""
    raw_results = load_all_results(results_dir)
    if not raw_results:
        msg = f"No results found in {results_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    results = _select_consistent_results(raw_results)
    ordered_names = _ordered_backbones(results)
    data_hash = next(iter(results.values())).data_hash

    log_section(logger, "BACKBONE COMPARISON RESULTS")
    logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
    logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

    logger.info("")
    logger.info(
        "  %-22s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Backbone",
        "Input",
        "Macro F1",
        "Params",
        "ONNX Size",
        "Time",
        "Thresholds",
    )
    logger.info("  " + "-" * 92)

    for name in ordered_names:
        r = results[name]
        tr = r.test_result
        thr_str = " ".join(f"{t:.2f}" for t in r.thresholds)
        logger.info(
            "  %-22s  %-8d  %-10.4f  %-10s  %-10s  %-10s  %s",
            name,
            r.input_size,
            tr.macro_f1,
            f"{r.param_count / 1e6:.1f}M",
            f"{r.onnx_size_mb:.1f}MB",
            f"{r.train_time_s:.0f}s",
            thr_str,
        )

    # Per-backbone environment (so cross-machine time numbers are interpretable)
    logger.info("")
    logger.info("Run environment:")
    for name in ordered_names:
        r = results[name]
        logger.info("  %-22s  host=%s  device=%s", name, r.hostname, r.device)

    # Per-class F1 table
    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for name in ordered_names:
        header += f"  {name:<22s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 24 * len(ordered_names)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for name in ordered_names:
            f1 = results[name].test_result.per_class_f1[i]
            row += f"  {f1:<22.4f}"
        logger.info(row)

    # Per-class precision/recall
    logger.info("")
    logger.info("Per-class Precision / Recall:")
    for name in ordered_names:
        logger.info("  %s:", name)
        tr = results[name].test_result
        for i, cls_name in enumerate(CLASS_NAMES):
            logger.info(
                "    %-20s  P=%.4f  R=%.4f  F1=%.4f",
                cls_name,
                tr.per_class_precision[i],
                tr.per_class_recall[i],
                tr.per_class_f1[i],
            )

    # Recommendation
    log_section(logger, "RECOMMENDATION")

    best_name = max(
        ordered_names,
        key=lambda n: results[n].test_result.macro_f1,
    )
    best_f1 = results[best_name].test_result.macro_f1

    logger.info("  Best backbone: %s (Macro F1=%.4f)", best_name, best_f1)
    logger.info("")

    for name in ordered_names:
        r = results[name]
        f1 = r.test_result.macro_f1
        diff = f1 - best_f1
        if name == best_name:
            logger.info("  * %s: F1=%.4f (BEST)", name, f1)
        else:
            logger.info("    %s: F1=%.4f (%+.4f vs best)", name, f1, diff)

    # Efficiency analysis — baseline is the first ordered backbone
    logger.info("")
    logger.info("  Efficiency analysis:")
    baseline_name = ordered_names[0]
    baseline = results[baseline_name]
    baseline_f1 = baseline.test_result.macro_f1
    for name in ordered_names:
        r = results[name]
        f1 = r.test_result.macro_f1
        gain = f1 - baseline_f1
        size_ratio = r.onnx_size_mb / baseline.onnx_size_mb
        time_ratio = r.train_time_s / baseline.train_time_s
        logger.info(
            "    %s: F1 %+.4f vs %s, %.1fx size, %.1fx time",
            name,
            gain,
            baseline_name,
            size_ratio,
            time_ratio,
        )
    logger.info(
        "  (time ratios only meaningful when both runs were on the same device)"
    )

    logger.info("")
    logger.info("  To use the best backbone for production training:")
    logger.info("    uv run python scripts/train.py --backbone %s", best_name)
    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train backbones one-at-a-time and merge results across machines. "
            "Use --run to train one backbone (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run",
        metavar="BACKBONE",
        help="Train one backbone and save its result JSON.",
    )
    mode.add_argument(
        "--report",
        action="store_true",
        help="Print a comparison from all result JSONs in --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Where result JSONs live (default: {DEFAULT_RESULTS_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.run is not None:
            cmd_run(args.run, args.results_dir)
        else:
            cmd_report(args.results_dir)
    except (ValueError, FileNotFoundError, RuntimeError):
        # Library/cmd functions log the error at the raise site;
        # the entry point only translates it to an exit code.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

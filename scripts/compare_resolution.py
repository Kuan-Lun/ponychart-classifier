"""
比較不同輸入解析度對訓練效果的影響。

此腳本拆成兩種模式，方便把不同解析度派到不同設備上跑：

1. 單跑一種解析度並把結果寫成 JSON：
     uv run --extra train python scripts/compare_resolution.py --run 320
     uv run --extra train python scripts/compare_resolution.py --run 448

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python scripts/compare_resolution.py --report

預設 results dir 是 ``scripts/resolution_results/``，可用 ``--results-dir``
指定其他路徑。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``<input_size>px__<hash12>.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。

測試五種 (PRE_RESIZE, INPUT_SIZE) 組合：
  224: PRE_RESIZE=256, INPUT_SIZE=224
  288: PRE_RESIZE=320, INPUT_SIZE=288
  320: PRE_RESIZE=384, INPUT_SIZE=320  (current production)
  380: PRE_RESIZE=448, INPUT_SIZE=380
  448: PRE_RESIZE=512, INPUT_SIZE=448
"""

from __future__ import annotations

import argparse
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
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    EnvDict,
    EvalResult,
    HoldoutSplit,
    Sample,
    build_cached_dataset,
    describe_device,
    evaluate,
    export_onnx,
    hash_samples,
    load_all_json_results,
    load_samples_logged,
    log_section,
    make_dataloader,
    prepare_holdout_split,
    seed_all,
    select_consistent_results,
    setup_device_and_workers,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "resolution_results"


# ---------------------------------------------------------------------------
# Resolution configs — ordered from highest to lowest so OOM surfaces early
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolutionConfig:
    """Per-resolution training/eval settings."""

    pre_resize: int
    input_size: int


RESOLUTION_CONFIGS: dict[str, ResolutionConfig] = {
    "448": ResolutionConfig(pre_resize=512, input_size=448),
    "380": ResolutionConfig(pre_resize=448, input_size=380),
    "320": ResolutionConfig(pre_resize=384, input_size=320),
    "288": ResolutionConfig(pre_resize=320, input_size=288),
    "224": ResolutionConfig(pre_resize=256, input_size=224),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single resolution experiment."""

    label: str
    pre_resize: int
    input_size: int
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
class TestResultDict(TypedDict):
    loss: float
    macro_f1: float
    per_class_f1: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]


class SplitDict(TypedDict):
    train_size: int
    val_size: int
    test_size: int


class ExperimentDict(TypedDict):
    label: str
    pre_resize: int
    input_size: int
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
    tr = exp.test_result
    return ExperimentDict(
        label=exp.label,
        pre_resize=exp.pre_resize,
        input_size=exp.input_size,
        param_count=exp.param_count,
        onnx_size_mb=exp.onnx_size_mb,
        train_time_s=exp.train_time_s,
        thresholds=list(exp.thresholds),
        test_result=TestResultDict(
            loss=tr.loss,
            macro_f1=tr.macro_f1,
            per_class_f1=list(tr.per_class_f1),
            per_class_precision=list(tr.per_class_precision),
            per_class_recall=list(tr.per_class_recall),
        ),
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
        label=data["label"],
        pre_resize=data["pre_resize"],
        input_size=data["input_size"],
        test_result=test_result,
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
    return f"{label}px__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(exp: ExperimentResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(exp.label, exp.data_hash)
    out_path.write_text(json.dumps(experiment_to_dict(exp), indent=2))
    return out_path


def _parse_result_file(raw: str) -> ExperimentResult:
    return experiment_from_dict(_parse_experiment_json(raw))


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def get_onnx_size_mb(model: nn.Module, input_size: int) -> float:
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
    return sum(p.numel() for p in model.parameters())


def run_experiment(
    label: str,
    config: ResolutionConfig,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    device: torch.device,
    num_workers: int,
    *,
    data_hash: str,
) -> ExperimentResult:

    log_section(
        logger,
        "RESOLUTION: %spx  (PRE_RESIZE=%d  INPUT_SIZE=%d)",
        label,
        config.pre_resize,
        config.input_size,
        width=70,
    )
    logger.info("  Data hash: %s", data_hash[:HASH_PREFIX_LEN])

    t0 = time.monotonic()
    seed_all(SEED)
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        f"{label}px",
        backbone=BACKBONE,
        input_size=config.input_size,
        pre_resize=config.pre_resize,
        batch_size=BATCH_SIZE,
    )
    train_time = time.monotonic() - t0

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
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    test_result = evaluate(model, test_loader, criterion, device, thresholds)

    param_count = count_parameters(model)
    onnx_size = get_onnx_size_mb(model, input_size=config.input_size)

    experiment = ExperimentResult(
        label=label,
        pre_resize=config.pre_resize,
        input_size=config.input_size,
        test_result=test_result,
        thresholds=list(thresholds),
        param_count=param_count,
        onnx_size_mb=onnx_size,
        train_time_s=train_time,
        train_size=len(train_samples),
        val_size=len(val_samples),
        test_size=len(test_samples),
        seed=SEED,
        data_hash=data_hash,
        hostname=socket.gethostname(),
        device=describe_device(device),
    )

    logger.info(
        ">> %spx: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
        label,
        test_result.macro_f1,
        param_count // 1000,
        onnx_size,
        train_time,
    )
    return experiment


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_run(label: str, results_dir: Path) -> None:
    if label not in RESOLUTION_CONFIGS:
        available = ", ".join(RESOLUTION_CONFIGS.keys())
        msg = (
            f"Resolution {label!r} not configured. "
            f"Currently configured: {available}"
        )
        logger.error(msg)
        raise ValueError(msg)
    config = RESOLUTION_CONFIGS[label]

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
        label,
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


def _ordered_labels(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded labels in canonical (config) order, then extras."""
    canonical = [lbl for lbl in RESOLUTION_CONFIGS if lbl in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


def cmd_report(results_dir: Path) -> None:
    raw_results = load_all_json_results(results_dir, _parse_result_file, logger)
    if not raw_results:
        msg = f"No results found in {results_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    results = select_consistent_results(
        raw_results,
        key=lambda r: r.label,
        key_label="resolution",
        logger=logger,
    )
    ordered = _ordered_labels(results)
    data_hash = next(iter(results.values())).data_hash

    log_section(logger, "RESOLUTION COMPARISON RESULTS")
    logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
    logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

    logger.info("")
    logger.info(
        "  %-12s  %-12s  %-12s  %-10s  %-10s  %-10s  %-10s",
        "Resolution",
        "PRE_RESIZE",
        "INPUT_SIZE",
        "Macro F1",
        "Params",
        "ONNX Size",
        "Time",
    )
    logger.info("  " + "-" * 82)

    baseline_label = ordered[-1]  # smallest resolution as baseline
    baseline_f1 = results[baseline_label].test_result.macro_f1

    for lbl in ordered:
        r = results[lbl]
        f1 = r.test_result.macro_f1
        logger.info(
            "  %-12s  %-12d  %-12d  %-10.4f  %-10s  %-10s  %s",
            f"{lbl}px",
            r.pre_resize,
            r.input_size,
            f1,
            f"{r.param_count / 1e6:.1f}M",
            f"{r.onnx_size_mb:.1f}MB",
            f"{r.train_time_s:.0f}s",
        )

    # Run environment
    logger.info("")
    logger.info("Run environment:")
    for lbl in ordered:
        r = results[lbl]
        logger.info("  %-12s  host=%s  device=%s", f"{lbl}px", r.hostname, r.device)

    # Per-class F1
    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for lbl in ordered:
        header += f"  {lbl + 'px':<14s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 16 * len(ordered)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for lbl in ordered:
            f1 = results[lbl].test_result.per_class_f1[i]
            row += f"  {f1:<14.4f}"
        logger.info(row)

    # Per-class precision/recall
    logger.info("")
    logger.info("Per-class Precision / Recall:")
    for lbl in ordered:
        logger.info("  %spx:", lbl)
        tr = results[lbl].test_result
        for i, cls_name in enumerate(CLASS_NAMES):
            logger.info(
                "    %-20s  P=%.4f  R=%.4f  F1=%.4f",
                cls_name,
                tr.per_class_precision[i],
                tr.per_class_recall[i],
                tr.per_class_f1[i],
            )

    # Summary
    log_section(logger, "SUMMARY")
    best_lbl = max(ordered, key=lambda lbl: results[lbl].test_result.macro_f1)
    best_f1 = results[best_lbl].test_result.macro_f1
    logger.info("  Best resolution: %spx (Macro F1=%.4f)", best_lbl, best_f1)

    logger.info("")
    for lbl in ordered:
        r = results[lbl]
        f1 = r.test_result.macro_f1
        diff = f1 - best_f1
        if lbl == best_lbl:
            logger.info("  * %spx: F1=%.4f (BEST)", lbl, f1)
        else:
            logger.info("    %spx: F1=%.4f (%+.4f vs best)", lbl, f1, diff)

    delta_vs_baseline = best_f1 - baseline_f1
    logger.info("")
    if best_lbl == baseline_label:
        logger.info("  結論: 提高解析度沒有帶來改善，維持 %spx", baseline_label)
    elif delta_vs_baseline > 0.005:
        logger.info(
            "  結論: %spx 有顯著改善 (%+.4f F1)，建議更新 constants.py",
            best_lbl,
            delta_vs_baseline,
        )
    else:
        logger.info(
            "  結論: %spx 改善有限 (%+.4f F1)，考量訓練時間後不建議更換",
            best_lbl,
            delta_vs_baseline,
        )
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train resolutions one-at-a-time and merge results across machines. "
            "Use --run to train one resolution (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run",
        metavar="INPUT_SIZE",
        help=(
            "Train one resolution and save its result JSON. "
            f"Available: {', '.join(RESOLUTION_CONFIGS.keys())}"
        ),
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


def main() -> None:
    args = parse_args()
    if args.run is not None:
        cmd_run(args.run, args.results_dir)
    else:
        cmd_report(args.results_dir)


if __name__ == "__main__":
    main()

"""
比較不同 backbone 架構對 PonyChart 分類效果的影響。

測試 MobileNetV3-Small、MobileNetV3-Large、EfficientNet-B0、
EfficientNet-B2 四種 backbone，
使用 holdout 評估：80% timestamp groups 用於訓練（原圖 + balanced crops），
20% groups 的原圖作為 holdout test set，模擬實際推論場景。
比較 Macro F1、per-class F1、模型大小和訓練時間。

使用方式：
  uv run python scripts/compare_backbones.py
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    BATCH_SIZE,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    EvalResult,
    HoldoutSplit,
    build_cached_dataset,
    evaluate,
    export_onnx,
    load_samples_or_exit,
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

BACKBONES = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b2",
]


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single backbone experiment."""

    test_result: EvalResult
    thresholds: list[float]
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    description: str


def get_onnx_size_mb(model: nn.Module) -> float:
    """Export model to a temp ONNX file and return its size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_onnx(model, tmp_path)
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


def run_experiment(
    backbone_name: str,
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    test_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
) -> ExperimentResult:
    """Train one backbone and evaluate on the test set.

    Heavy objects (model, dataset, DataLoader workers) are freed
    automatically when this function returns.
    """
    config = BACKBONE_REGISTRY[backbone_name]
    log_section(logger, "BACKBONE: %s", config.description, width=70)

    t0 = time.monotonic()
    seed_all(SEED)
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        backbone_name,
        backbone=backbone_name,
    )
    train_time = time.monotonic() - t0

    # Evaluate on test set
    model, thresholds = result.model, result.thresholds
    criterion = nn.BCEWithLogitsLoss()
    test_ds = build_cached_dataset(test_samples, is_train=False)
    test_loader = make_dataloader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    test_result = evaluate(model, test_loader, criterion, device, thresholds)

    # Model stats
    param_count = count_parameters(model)
    onnx_size = get_onnx_size_mb(model)

    experiment = ExperimentResult(
        test_result=test_result,
        thresholds=thresholds,
        param_count=param_count,
        onnx_size_mb=onnx_size,
        train_time_s=train_time,
        description=config.description,
    )

    logger.info(
        ">> %s: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
        backbone_name,
        test_result.macro_f1,
        param_count // 1000,
        onnx_size,
        train_time,
    )
    return experiment


def main() -> None:
    rng = seed_all(SEED)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_or_exit(logger)

    # ── Split into train / val / test ──
    hs: HoldoutSplit = prepare_holdout_split(
        all_samples, rng, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )
    train_samples, val_samples, test_samples = hs.train, hs.val, hs.test

    # ── Run experiments ──
    results: dict[str, ExperimentResult] = {}

    for backbone_name in BACKBONES:
        results[backbone_name] = run_experiment(
            backbone_name,
            train_samples,
            val_samples,
            test_samples,
            device,
            num_workers,
        )

    # ── Comparison table ──
    log_section(logger, "BACKBONE COMPARISON RESULTS")

    logger.info("")
    logger.info(
        "  %-22s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Backbone",
        "Macro F1",
        "Params",
        "ONNX Size",
        "Time",
        "Thresholds",
    )
    logger.info("  " + "-" * 82)

    for name in BACKBONES:
        r = results[name]
        tr = r.test_result
        thr_str = " ".join(f"{t:.2f}" for t in r.thresholds)
        logger.info(
            "  %-22s  %-10.4f  %-10s  %-10s  %-10s  %s",
            name,
            tr.macro_f1,
            f"{r.param_count / 1e6:.1f}M",
            f"{r.onnx_size_mb:.1f}MB",
            f"{r.train_time_s:.0f}s",
            thr_str,
        )

    # Per-class F1 table
    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for name in BACKBONES:
        header += f"  {name:<22s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 24 * len(BACKBONES)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for name in BACKBONES:
            f1 = results[name].test_result.per_class_f1[i]
            row += f"  {f1:<22.4f}"
        logger.info(row)

    # Per-class precision/recall
    logger.info("")
    logger.info("Per-class Precision / Recall:")
    for name in BACKBONES:
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

    # ── Recommendation ──
    log_section(logger, "RECOMMENDATION")

    best_name = max(
        BACKBONES,
        key=lambda n: results[n].test_result.macro_f1,
    )
    best_r = results[best_name]
    best_f1 = best_r.test_result.macro_f1

    logger.info("  Best backbone: %s (Macro F1=%.4f)", best_name, best_f1)
    logger.info("")

    # Compare each to best
    for name in BACKBONES:
        r = results[name]
        f1 = r.test_result.macro_f1
        diff = f1 - best_f1
        if name == best_name:
            logger.info("  * %s: F1=%.4f (BEST)", name, f1)
        else:
            logger.info("    %s: F1=%.4f (%+.4f vs best)", name, f1, diff)

    # Efficiency analysis
    logger.info("")
    logger.info("  Efficiency analysis:")
    small_f1 = results["mobilenet_v3_small"].test_result.macro_f1
    for name in BACKBONES:
        r = results[name]
        f1 = r.test_result.macro_f1
        gain = f1 - small_f1
        size_ratio = r.onnx_size_mb / results["mobilenet_v3_small"].onnx_size_mb
        time_ratio = r.train_time_s / results["mobilenet_v3_small"].train_time_s
        logger.info(
            "    %s: F1 %+.4f vs small, %.1fx size, %.1fx time",
            name,
            gain,
            size_ratio,
            time_ratio,
        )

    logger.info("")
    logger.info("  To use the best backbone for production training:")
    logger.info(
        "    uv run python scripts/train.py" " --backbone %s",
        best_name,
    )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()

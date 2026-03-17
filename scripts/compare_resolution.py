"""
比較不同輸入解析度對訓練效果的影響。

測試三種 (PRE_RESIZE, INPUT_SIZE) 組合：
  224: PRE_RESIZE=256, INPUT_SIZE=224  (current baseline)
  288: PRE_RESIZE=320, INPUT_SIZE=288
  320: PRE_RESIZE=384, INPUT_SIZE=320

共用 20% groups 原圖作為 holdout test set（各解析度使用各自的 test transforms）。

使用方式：
  uv run python scripts/compare_resolution.py
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    evaluate,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    load_samples,
    log_section,
    make_dataloader,
    prepare_holdout_split,
    train_model,
)
from ponychart_classifier.training.dataset import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESOLUTIONS: list[tuple[int, int]] = [
    (256, 224),
    (320, 288),
    (384, 320),
]


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # ── Load all samples ──
    all_samples = load_samples()
    if not all_samples:
        logger.error("No samples found. Check rawimage/ and rawimage/labels.json.")
        return
    logger.info("Total samples loaded: %d", len(all_samples))

    # ── Split groups: test / val / train ──
    split = prepare_holdout_split(
        all_samples, rng, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )
    train_samples = split.train
    val_samples = split.val
    test_samples = split.test
    logger.info("Test set (originals only): %d images", len(test_samples))
    logger.info(
        "Train: %s  Val: %s", f"{len(train_samples):,}", f"{len(val_samples):,}"
    )

    criterion = nn.BCEWithLogitsLoss()

    # ── Run experiments ──
    results: dict[str, dict[str, Any]] = {}

    for pre_resize, input_size in RESOLUTIONS:
        label = f"{input_size}px"
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_section(
            logger,
            "RESOLUTION: PRE_RESIZE=%d  INPUT_SIZE=%d",
            pre_resize,
            input_size,
            width=60,
        )

        t0 = time.monotonic()
        result = train_model(
            train_samples,
            val_samples,
            device,
            num_workers,
            label,
            backbone=BACKBONE,
            pre_resize=pre_resize,
            input_size=input_size,
        )
        model, thresholds = result.model, result.thresholds
        train_time = time.monotonic() - t0

        # Each resolution needs its own test transforms and dataset
        test_tf = get_transforms(is_train=False, input_size=input_size)
        test_ds = PonyChartDataset(test_samples, test_tf, pre_resize=pre_resize)
        test_loader = make_dataloader(
            test_ds,
            BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        )

        test_result = evaluate(model, test_loader, criterion, device, thresholds)

        results[label] = {
            "pre_resize": pre_resize,
            "input_size": input_size,
            "test_result": test_result,
            "thresholds": thresholds,
            "train_time_s": train_time,
        }

        logger.info(
            ">> %s: test Macro F1=%.4f  time=%.0fs",
            label,
            test_result["macro_f1"],
            train_time,
        )

    # ── Comparison table ──
    labels = [f"{s}px" for _, s in RESOLUTIONS]

    log_section(logger, "RESOLUTION COMPARISON RESULTS", width=80)
    logger.info("")

    baseline_f1 = results[labels[0]]["test_result"]["macro_f1"]
    logger.info(
        "  %-10s  %-12s  %-12s  %-10s  %-10s",
        "Resolution",
        "PRE_RESIZE",
        "INPUT_SIZE",
        "Macro F1",
        "Delta",
    )
    logger.info("  " + "-" * 58)
    for lbl in labels:
        r = results[lbl]
        f1 = r["test_result"]["macro_f1"]
        delta = f1 - baseline_f1
        marker = " (baseline)" if lbl == labels[0] else f"  {delta:+.4f}"
        logger.info(
            "  %-10s  %-12d  %-12d  %-10.4f  %s",
            lbl,
            r["pre_resize"],
            r["input_size"],
            f1,
            marker,
        )

    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for lbl in labels:
        header += f"  {lbl:<12s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 14 * len(labels)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for lbl in labels:
            f1 = results[lbl]["test_result"]["per_class_f1"][i]
            row += f"  {f1:<12.4f}"
        logger.info(row)

    logger.info("")
    logger.info("Training time:")
    for lbl in labels:
        r = results[lbl]
        logger.info(
            "  %s: %.0fs (%.1fx baseline)",
            lbl,
            r["train_time_s"],
            r["train_time_s"] / results[labels[0]]["train_time_s"],
        )

    # ── Summary ──
    log_section(logger, "SUMMARY", width=80)
    best_lbl = max(labels, key=lambda lbl: results[lbl]["test_result"]["macro_f1"])
    best_f1 = results[best_lbl]["test_result"]["macro_f1"]
    logger.info("  Best resolution: %s (Macro F1=%.4f)", best_lbl, best_f1)
    delta_vs_baseline = best_f1 - baseline_f1
    if best_lbl == labels[0]:
        logger.info("  結論: 提高解析度沒有帶來改善，維持 %s", labels[0])
    elif delta_vs_baseline > 0.005:
        logger.info(
            "  結論: %s 有顯著改善 (%+.4f F1)，建議更新 constants.py",
            best_lbl,
            delta_vs_baseline,
        )
    else:
        logger.info(
            "  結論: %s 改善有限 (%+.4f F1)，考量訓練時間後不建議更換",
            best_lbl,
            delta_vs_baseline,
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

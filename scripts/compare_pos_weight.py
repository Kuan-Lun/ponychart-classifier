"""
比較有無 pos_weight 對訓練效果的影響。

Experiment A (Baseline): BCEWithLogitsLoss() 無 pos_weight
Experiment B (pos_weight): BCEWithLogitsLoss(pos_weight=w)
  w[cls] = (N - pos_count) / pos_count

共用 20% groups 原圖作為 holdout test set。

使用方式：
  uv run python scripts/compare_pos_weight.py
"""

from __future__ import annotations

import logging

import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    NUM_CLASSES,
    SEED,
    VAL_SIZE,
    compute_pos_weight,
    evaluate,
    load_samples_or_exit,
    log_section,
    make_test_loader,
    prepare_holdout_split_logged,
    seed_all,
    setup_device_and_workers,
    train_with_seed_reset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    rng = seed_all(SEED)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_or_exit(logger)

    # ── Split into train / val / test ──
    split = prepare_holdout_split_logged(
        all_samples, rng, logger, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )
    train_samples, val_samples, test_samples = split.train, split.val, split.test

    # ── Compute pos_weight from training data ──
    pw = compute_pos_weight(train_samples)
    logger.info("pos_weight: %s", dict(zip(CLASS_NAMES, pw.tolist())))

    # ── Experiment A: Baseline (no pos_weight) ──
    result_a = train_with_seed_reset(
        train_samples,
        val_samples,
        device,
        num_workers,
        "A: Baseline (no pos_weight)",
        backbone=BACKBONE,
        verbose=True,
    )
    model_a, thresholds_a = result_a.model, result_a.thresholds

    # ── Experiment B: With pos_weight ──
    train_result_b = train_with_seed_reset(
        train_samples,
        val_samples,
        device,
        num_workers,
        "B: With pos_weight",
        backbone=BACKBONE,
        pos_weight=pw,
        verbose=True,
    )
    model_b, thresholds_b = train_result_b.model, train_result_b.thresholds

    # ── Evaluate both on holdout test set ──
    criterion = nn.BCEWithLogitsLoss()
    test_loader = make_test_loader(test_samples, num_workers=num_workers, device=device)

    result_a = evaluate(model_a, test_loader, criterion, device, thresholds_a)
    result_b = evaluate(model_b, test_loader, criterion, device, thresholds_b)

    # ── Report ──
    log_section(
        logger,
        "HOLDOUT TEST SET EVALUATION (%d original images)",
        len(test_samples),
        width=80,
    )
    logger.info("  A thresholds: %s", dict(zip(CLASS_NAMES, thresholds_a)))
    logger.info("  B thresholds: %s", dict(zip(CLASS_NAMES, thresholds_b)))
    logger.info("")

    logger.info(
        "%-20s  %-18s  %-18s  %-10s",
        "Metric",
        "A (Baseline)",
        "B (pos_weight)",
        "Delta",
    )
    logger.info("-" * 70)
    delta_f1 = result_b.macro_f1 - result_a.macro_f1
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Macro F1",
        result_a.macro_f1,
        result_b.macro_f1,
        delta_f1,
    )
    delta_loss = result_b.loss - result_a.loss
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Loss",
        result_a.loss,
        result_b.loss,
        delta_loss,
    )

    logger.info("")
    logger.info("Per-class F1 comparison:")
    logger.info(
        "  %-20s  %-7s %-7s %-7s | %-7s %-7s %-7s | %-7s",
        "Class",
        "A_P",
        "A_R",
        "A_F1",
        "B_P",
        "B_R",
        "B_F1",
        "Delta",
    )
    logger.info("  " + "-" * 75)
    deltas = []
    for i, name in enumerate(CLASS_NAMES):
        d = result_b.per_class_f1[i] - result_a.per_class_f1[i]
        deltas.append(d)
        logger.info(
            "  %-20s  %-7.4f %-7.4f %-7.4f | %-7.4f %-7.4f %-7.4f | %+.4f",
            name,
            result_a.per_class_precision[i],
            result_a.per_class_recall[i],
            result_a.per_class_f1[i],
            result_b.per_class_precision[i],
            result_b.per_class_recall[i],
            result_b.per_class_f1[i],
            d,
        )

    # ── Summary ──
    log_section(logger, "SUMMARY", width=80)
    logger.info(
        "  Macro F1:  A (Baseline)=%.4f  B (pos_weight)=%.4f  Delta=%+.4f",
        result_a.macro_f1,
        result_b.macro_f1,
        delta_f1,
    )

    improved = sum(1 for d in deltas if d > 0)
    degraded = sum(1 for d in deltas if d < 0)
    logger.info(
        "  Per-class: %d improved, %d degraded (of %d)",
        improved,
        degraded,
        NUM_CLASSES,
    )

    if delta_f1 > 0.005:
        logger.info("  結論: pos_weight 有正面效果 (%+.4f F1)", delta_f1)
    elif delta_f1 < -0.005:
        logger.info("  結論: pos_weight 有負面效果 (%+.4f F1)", delta_f1)
    else:
        logger.info("  結論: pos_weight 影響不大 (delta=%.4f)", delta_f1)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

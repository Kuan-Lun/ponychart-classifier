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
    NUM_CLASSES,
    PerClassMetricsBlock,
    ThresholdRow,
    compute_pos_weight,
    configure_logging,
    evaluate,
    get_transforms,
    is_significant_improvement,
    is_significant_regression,
    log_named_thresholds,
    log_per_class_precision_recall_blocks,
    log_section,
    make_test_loader,
    prepare_holdout_setup_logged,
    train_with_seed_reset,
)

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    setup = prepare_holdout_setup_logged(logger)
    train_samples = setup.split.train
    val_samples = setup.split.val
    test_samples = setup.split.test

    # ── Compute pos_weight from training data ──
    pw = compute_pos_weight(train_samples)
    logger.info("pos_weight: %s", dict(zip(CLASS_NAMES, pw.tolist())))

    # ── Experiment A: Baseline (no pos_weight) ──
    train_result_a = train_with_seed_reset(
        train_samples,
        val_samples,
        setup.device,
        setup.num_workers,
        "A: Baseline (no pos_weight)",
        train_transform=get_transforms(is_train=True),
        backbone=BACKBONE,
        verbose=True,
    )
    model_a, thresholds_a = train_result_a.model, train_result_a.thresholds

    # ── Experiment B: With pos_weight ──
    train_result_b = train_with_seed_reset(
        train_samples,
        val_samples,
        setup.device,
        setup.num_workers,
        "B: With pos_weight",
        train_transform=get_transforms(is_train=True),
        backbone=BACKBONE,
        pos_weight=pw,
        verbose=True,
    )
    model_b, thresholds_b = train_result_b.model, train_result_b.thresholds

    # ── Evaluate both on holdout test set ──
    criterion = nn.BCEWithLogitsLoss()
    test_loader = make_test_loader(
        test_samples,
        num_workers=setup.num_workers,
        device=setup.device,
    )

    eval_a = evaluate(model_a, test_loader, criterion, setup.device, thresholds_a)
    eval_b = evaluate(model_b, test_loader, criterion, setup.device, thresholds_b)

    # ── Report ──
    log_section(
        logger,
        "HOLDOUT TEST SET EVALUATION (%d original images)",
        len(test_samples),
        width=80,
    )
    log_named_thresholds(
        logger,
        list(CLASS_NAMES),
        [
            ThresholdRow(label="A thresholds", thresholds=thresholds_a),
            ThresholdRow(label="B thresholds", thresholds=thresholds_b),
        ],
    )
    logger.info("")

    logger.info(
        "%-20s  %-18s  %-18s  %-10s",
        "Metric",
        "A (Baseline)",
        "B (pos_weight)",
        "Delta",
    )
    logger.info("-" * 70)
    delta_f1 = eval_b.macro_f1 - eval_a.macro_f1
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Macro F1",
        eval_a.macro_f1,
        eval_b.macro_f1,
        delta_f1,
    )
    delta_loss = eval_b.loss - eval_a.loss
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Loss",
        eval_a.loss,
        eval_b.loss,
        delta_loss,
    )

    logger.info("")
    deltas = []
    for i in range(len(CLASS_NAMES)):
        deltas.append(eval_b.per_class_f1[i] - eval_a.per_class_f1[i])
    log_per_class_precision_recall_blocks(
        logger,
        list(CLASS_NAMES),
        [
            PerClassMetricsBlock(
                label="A (Baseline)",
                precision=eval_a.per_class_precision,
                recall=eval_a.per_class_recall,
                f1=eval_a.per_class_f1,
            ),
            PerClassMetricsBlock(
                label="B (pos_weight)",
                precision=eval_b.per_class_precision,
                recall=eval_b.per_class_recall,
                f1=eval_b.per_class_f1,
            ),
        ],
        title="Per-class Precision / Recall:",
    )
    logger.info("")
    logger.info("Per-class F1 delta (B - A):")
    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %-20s  %+.4f", name, deltas[i])

    # ── Summary ──
    log_section(logger, "SUMMARY", width=80)
    logger.info(
        "  Macro F1:  A (Baseline)=%.4f  B (pos_weight)=%.4f  Delta=%+.4f",
        eval_a.macro_f1,
        eval_b.macro_f1,
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

    if is_significant_improvement(delta_f1):
        logger.info("  結論: pos_weight 有正面效果 (%+.4f F1)", delta_f1)
    elif is_significant_regression(delta_f1):
        logger.info("  結論: pos_weight 有負面效果 (%+.4f F1)", delta_f1)
    else:
        logger.info("  結論: pos_weight 影響不大 (delta=%.4f)", delta_f1)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

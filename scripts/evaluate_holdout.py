"""
Holdout 評估：在僅原圖的 test set 上測量真實 F1。

80% timestamp groups 用於訓練（原圖 + balanced crops），
20% groups 的原圖作為 holdout test set，模擬實際推論場景。

Thresholds 在 val set 上 optimize，再套用到 test set 評估。

使用方式：
  uv run python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import argparse
import logging

import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE,
    CLASS_NAMES,
    PerClassMetricsBlock,
    ThresholdRow,
    configure_logging,
    evaluate,
    get_transforms,
    log_named_thresholds,
    log_per_class_precision_recall_blocks,
    log_section,
    make_test_loader,
    prepare_holdout_setup_logged,
    train_model,
)

logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(
        description="Evaluate model F1 on originals-only holdout test set"
    ).parse_args()

    configure_logging(logging.INFO)

    _run()


def _run() -> None:
    setup = prepare_holdout_setup_logged(logger)
    train_samples = setup.split.train
    val_samples = setup.split.val
    test_samples = setup.split.test

    # ── Train from scratch (never resume: different split → data leakage) ──
    train_result = train_model(
        train_samples,
        val_samples,
        setup.device,
        setup.num_workers,
        "Holdout Evaluation",
        train_transform=get_transforms(is_train=True),
        backbone=BACKBONE,
        verbose=True,
    )
    model, thresholds = train_result.model, train_result.thresholds

    # ── Evaluate on holdout test set (originals only) ──
    criterion = nn.BCEWithLogitsLoss()
    test_loader = make_test_loader(
        test_samples,
        num_workers=setup.num_workers,
        device=setup.device,
    )

    eval_result = evaluate(model, test_loader, criterion, setup.device, thresholds)

    # ── Report ──
    log_section(
        logger,
        "HOLDOUT TEST SET EVALUATION (%d original images)",
        len(test_samples),
        width=70,
    )
    log_named_thresholds(
        logger,
        list(CLASS_NAMES),
        [ThresholdRow(label="Thresholds (from val set)", thresholds=thresholds)],
    )
    logger.info("")
    logger.info("  Macro F1: %.4f", eval_result.macro_f1)
    logger.info("  Loss:     %.4f", eval_result.loss)
    log_per_class_precision_recall_blocks(
        logger,
        list(CLASS_NAMES),
        [
            PerClassMetricsBlock(
                label="Holdout test set",
                precision=eval_result.per_class_precision,
                recall=eval_result.per_class_recall,
                f1=eval_result.per_class_f1,
            )
        ],
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

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
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    evaluate,
    load_samples_or_exit,
    log_section,
    make_test_loader,
    prepare_holdout_split_logged,
    seed_all,
    setup_device_and_workers,
    train_model,
)

logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(
        description="Evaluate model F1 on originals-only holdout test set"
    ).parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    rng = seed_all(SEED)
    device, num_workers = setup_device_and_workers(logger)
    all_samples = load_samples_or_exit(logger)

    # ── Split into train / val / test ──
    split = prepare_holdout_split_logged(
        all_samples, rng, logger, test_size=HOLDOUT_TEST_SIZE, val_size=VAL_SIZE
    )
    train_samples, val_samples, test_samples = split.train, split.val, split.test

    # ── Train from scratch (never resume: different split → data leakage) ──
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "Holdout Evaluation",
        backbone=BACKBONE,
        verbose=True,
    )
    model, thresholds = result.model, result.thresholds

    # ── Evaluate on holdout test set (originals only) ──
    criterion = nn.BCEWithLogitsLoss()
    test_loader = make_test_loader(test_samples, num_workers=num_workers, device=device)

    result = evaluate(model, test_loader, criterion, device, thresholds)

    # ── Report ──
    log_section(
        logger,
        "HOLDOUT TEST SET EVALUATION (%d original images)",
        len(test_samples),
        width=70,
    )
    logger.info("Thresholds (from val set): %s", dict(zip(CLASS_NAMES, thresholds)))
    logger.info("")
    logger.info("  Macro F1: %.4f", result.macro_f1)
    logger.info("  Loss:     %.4f", result.loss)
    logger.info("")
    logger.info(
        "  %-20s  %-10s  %-10s  %-10s",
        "Class",
        "Precision",
        "Recall",
        "F1",
    )
    logger.info("  " + "-" * 55)
    for i, name in enumerate(CLASS_NAMES):
        logger.info(
            "  %-20s  %-10.4f  %-10.4f  %-10.4f",
            name,
            result.per_class_precision[i],
            result.per_class_recall[i],
            result.per_class_f1[i],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

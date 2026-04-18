"""Two-phase training experiment for batch-LR search.

Phase 1 freezes the backbone and trains only the head, with val_loss
early stopping (patience ``PHASE1_PATIENCE``) matching ``train.py``.
Phase 2 unfreezes and fine-tunes with ReduceLROnPlateau + val_F1 early
stopping.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from ponychart_classifier.training import (
    MIN_DELTA_F1,
    MIN_DELTA_LOSS,
    PHASE1_PATIENCE,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEARCH_PATIENCE,
    SEARCH_PHASE1_EPOCHS,
    SEARCH_PHASE2_EPOCHS,
    WEIGHT_DECAY,
    build_model,
    evaluate,
    make_dataloader,
    train_one_epoch,
)
from ponychart_classifier.training.dataset import PonyChartDataset
from ponychart_classifier.training.model import _extract_submodules


def run_experiment(
    train_ds: PonyChartDataset,
    val_ds: PonyChartDataset,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    lr_head: float,
    lr_features: float,
    lr_classifier: float,
    backbone: str,
) -> tuple[float, list[float], int, int, float]:
    """Run one two-phase training experiment.

    Returns ``(best_f1, best_per_class_f1, phase1_stopped_epoch,
    stopped_epoch, elapsed_s)``.
    """
    t0 = time.monotonic()
    train_loader = make_dataloader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    model = build_model(backbone=backbone, pretrained=True).to(device)
    features, classifier = _extract_submodules(model)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: Head only (val_loss early stopping, mirrors train.py)
    for param in features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr_head, weight_decay=WEIGHT_DECAY
    )
    best_p1_loss = float("inf")
    p1_patience_counter = 0
    phase1_stopped_epoch = SEARCH_PHASE1_EPOCHS
    for epoch in range(1, SEARCH_PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        p1_val_loss = evaluate(model, val_loader, criterion, device).loss
        if p1_val_loss < best_p1_loss - MIN_DELTA_LOSS:
            best_p1_loss = p1_val_loss
            p1_patience_counter = 0
        else:
            p1_patience_counter += 1
        if p1_patience_counter >= PHASE1_PATIENCE:
            phase1_stopped_epoch = epoch
            break

    # Phase 2: Full fine-tuning
    for param in features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": features.parameters(), "lr": lr_features},
            {"params": classifier.parameters(), "lr": lr_classifier},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    best_f1 = 0.0
    best_per_class: list[float] = []
    patience_counter = 0
    stopped_epoch = SEARCH_PHASE2_EPOCHS

    for epoch in range(1, SEARCH_PHASE2_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = val_result.macro_f1
        scheduler.step(val_f1)

        if val_f1 > best_f1 + MIN_DELTA_F1:
            best_f1 = val_f1
            best_per_class = list(val_result.per_class_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= SEARCH_PATIENCE:
            stopped_epoch = epoch
            break

    return (
        best_f1,
        best_per_class,
        phase1_stopped_epoch,
        stopped_epoch,
        time.monotonic() - t0,
    )

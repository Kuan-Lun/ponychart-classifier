"""Logging helpers for structured report output."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def log_section(
    logger: logging.Logger,
    title: str,
    *args: Any,
    width: int = 90,
) -> None:
    """Log a section header: blank line, = separator, title, = separator.

    Usage::

        log_section(logger, "RECOMMENDATION", width=90)
        log_section(logger, "RESULTS (%d images)", n, width=80)
    """
    logger.info("")
    logger.info("=" * width)
    logger.info(title, *args)
    logger.info("=" * width)


@dataclass(frozen=True)
class PerClassColumn:
    """One column in a per-class F1 comparison matrix."""

    label: str
    macro_f1: float
    per_class_f1: Sequence[float]


@dataclass(frozen=True)
class PerClassPrecisionRecallBlock:
    """Named per-class precision/recall metrics for one configuration."""

    label: str
    precision: Sequence[float]
    recall: Sequence[float]


@dataclass(frozen=True)
class RunEnvironmentRow:
    """Host/device metadata for one experiment row."""

    label: str
    hostname: str
    device: str


@dataclass(frozen=True)
class ThresholdRow:
    """Named threshold vector for one configuration."""

    label: str
    thresholds: Sequence[float]


def _validate_per_class_matrix(
    class_names: Sequence[str],
    columns: Sequence[PerClassColumn],
) -> None:
    """Reject malformed matrix data with caller-facing errors."""
    if not class_names:
        raise ValueError("class_names must not be empty")

    for column in columns:
        _validate_same_length(
            class_names,
            column.per_class_f1,
            field_name="per-class F1",
            owner_label=column.label,
        )


def _validate_same_length(
    class_names: Sequence[str],
    values: Sequence[float],
    *,
    field_name: str,
    owner_label: str,
) -> None:
    """Reject malformed per-class sequences with caller-facing errors."""
    expected_count = len(class_names)
    actual_count = len(values)
    if actual_count != expected_count:
        raise ValueError(
            f"{field_name} count mismatch for {owner_label!r}: "
            f"expected {expected_count}, got {actual_count}"
        )


def log_per_class_f1_matrix(
    logger: logging.Logger,
    class_names: Sequence[str],
    columns: Sequence[PerClassColumn],
    *,
    title: str = ("Per-class F1 (* = best per class; ties give each column a win):"),
) -> None:
    """Log a class-by-column F1 matrix with winner markers + macro + wins."""
    if not columns:
        return
    _validate_per_class_matrix(class_names, columns)

    cell_width = max(7, max(len(c.label) for c in columns))
    class_width = max(
        len("Class"),
        max(len(n) for n in class_names),
        len("Macro F1"),
        len("Wins (ties count)"),
    )
    sep_width = class_width + (cell_width + 2) * len(columns)

    logger.info("")
    logger.info("%s", title)

    header = f"  {'Class':<{class_width}s}"
    for col in columns:
        header += f"  {col.label:<{cell_width}s}"
    logger.info("%s", header)
    logger.info("  %s", "-" * sep_width)

    wins = [0] * len(columns)
    for i, name in enumerate(class_names):
        values = [c.per_class_f1[i] for c in columns]
        best_val = max(values)
        row = f"  {name:<{class_width}s}"
        for j, value in enumerate(values):
            marker = "*" if value == best_val else " "
            cell = f"{value:.4f}{marker}"
            row += f"  {cell:<{cell_width}s}"
            if value == best_val:
                wins[j] += 1
        logger.info("%s", row)

    logger.info("  %s", "-" * sep_width)

    macros = [c.macro_f1 for c in columns]
    best_macro = max(macros)
    macro_row = f"  {'Macro F1':<{class_width}s}"
    for macro in macros:
        marker = "*" if macro == best_macro else " "
        cell = f"{macro:.4f}{marker}"
        macro_row += f"  {cell:<{cell_width}s}"
    logger.info("%s", macro_row)

    wins_row = f"  {'Wins (ties count)':<{class_width}s}"
    for win_count in wins:
        wins_row += f"  {str(win_count):<{cell_width}s}"
    logger.info("%s", wins_row)


def log_per_class_precision_recall(
    logger: logging.Logger,
    class_names: Sequence[str],
    blocks: Sequence[PerClassPrecisionRecallBlock],
    *,
    title: str = "Per-class Precision / Recall:",
) -> None:
    """Log one or more named precision/recall blocks."""
    if not blocks:
        return
    if not class_names:
        raise ValueError("class_names must not be empty")

    for block in blocks:
        _validate_same_length(
            class_names,
            block.precision,
            field_name="precision",
            owner_label=block.label,
        )
        _validate_same_length(
            class_names,
            block.recall,
            field_name="recall",
            owner_label=block.label,
        )

    logger.info("")
    logger.info("%s", title)
    for block in blocks:
        logger.info("  %s:", block.label)
        for i, cls_name in enumerate(class_names):
            logger.info(
                "    %-20s  P=%.4f  R=%.4f",
                cls_name,
                block.precision[i],
                block.recall[i],
            )


def log_run_environments(
    logger: logging.Logger,
    rows: Sequence[RunEnvironmentRow],
    *,
    title: str = "Run environment:",
) -> None:
    """Log host/device rows with a shared heading."""
    if not rows:
        return

    label_width = max(len(row.label) for row in rows)
    logger.info("")
    logger.info("%s", title)
    for row in rows:
        logger.info(
            "  %-*s  host=%s  device=%s",
            label_width,
            row.label,
            row.hostname,
            row.device,
        )


def format_thresholds(thresholds: Sequence[float]) -> str:
    """Format per-class thresholds in a compact space-separated form."""
    return " ".join(f"{threshold:.2f}" for threshold in thresholds)


def log_named_thresholds(
    logger: logging.Logger,
    class_names: Sequence[str],
    rows: Sequence[ThresholdRow],
    *,
    title: str = "Thresholds:",
) -> None:
    """Log one or more named threshold vectors."""
    if not rows:
        return
    if not class_names:
        raise ValueError("class_names must not be empty")

    logger.info("")
    logger.info("%s", title)
    for row in rows:
        _validate_same_length(
            class_names,
            row.thresholds,
            field_name="threshold",
            owner_label=row.label,
        )
        logger.info("  %s:", row.label)
        for cls_name, threshold in zip(class_names, row.thresholds, strict=True):
            logger.info("    %-20s  %.2f", cls_name, threshold)

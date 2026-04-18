from __future__ import annotations

import logging

import pytest

from ponychart_classifier.training import PerClassColumn, log_per_class_f1_matrix


def test_log_per_class_f1_matrix_marks_ties_and_appends_summary_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.cli.report_utils")
    columns = [
        PerClassColumn(label="cfg-a", macro_f1=0.7000, per_class_f1=[0.8000, 0.6000]),
        PerClassColumn(label="cfg-b", macro_f1=0.7000, per_class_f1=[0.8000, 0.5000]),
    ]

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_per_class_f1_matrix(logger, ["Twilight", "Rainbow"], columns)

    lines = caplog.messages
    assert "Per-class F1 (* = best per class; ties give each column a win):" in lines
    assert any("Twilight" in line and "0.8000*" in line for line in lines)
    assert any(
        "Rainbow" in line and "0.6000*" in line and "0.5000 " in line for line in lines
    )
    assert any("Macro F1" in line and line.count("0.7000*") == 2 for line in lines)
    assert any(
        "Wins (ties count)" in line and "2" in line and "1" in line for line in lines
    )


def test_log_per_class_f1_matrix_rejects_empty_class_names() -> None:
    logger = logging.getLogger("tests.cli.report_utils.empty")
    columns = [PerClassColumn(label="cfg", macro_f1=0.8, per_class_f1=[0.8])]

    with pytest.raises(ValueError, match="class_names must not be empty"):
        log_per_class_f1_matrix(logger, [], columns)


def test_log_per_class_f1_matrix_rejects_mismatched_per_class_lengths() -> None:
    logger = logging.getLogger("tests.cli.report_utils.mismatch")
    columns = [PerClassColumn(label="cfg", macro_f1=0.8, per_class_f1=[0.8])]

    with pytest.raises(
        ValueError,
        match="per-class F1 count mismatch for 'cfg': expected 2, got 1",
    ):
        log_per_class_f1_matrix(logger, ["Twilight", "Rainbow"], columns)

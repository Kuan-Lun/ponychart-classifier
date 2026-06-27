from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from scripts import prune_small_confident_crops as script

# SUSPICIOUS_MARGIN = 0.15; select_predictions returns classes where prob >= threshold.
# These probe fixtures use threshold=0.5 throughout.
_THR = [0.5] * 6
_CONFIDENT_PROB = 0.9  # |0.9 - 0.5| = 0.4 >= 0.15 → confident
_BORDERLINE_PROB = 0.55  # |0.55 - 0.5| = 0.05 < 0.15  → not confident
_BELOW_THR = 0.1  # 0.1 < 0.5 → not predicted; |0.1 - 0.5| = 0.4 → confident miss


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size).save(path)


def test_find_candidates_returns_small_crops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240101_000000.png"
    small_crop_name = "pony_chart_20240101_000000_1.png"
    large_crop_name = "pony_chart_20240101_000000_2.png"

    _make_image(tmp_path / orig_name, (1004, 1004))
    _make_image(tmp_path / small_crop_name, (200, 100))  # min < 277 -> too small
    _make_image(tmp_path / large_crop_name, (300, 600))  # min >= 277 -> ok

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)

    candidates = script.find_candidates()
    assert candidates == [tmp_path / small_crop_name]


def test_find_candidates_excludes_originals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240202_000000.png"
    _make_image(tmp_path / orig_name, (200, 100))  # small but is an original

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)

    assert script.find_candidates() == []


# ---------------------------------------------------------------------------
# _all_states_acceptable unit tests
# ---------------------------------------------------------------------------


def test_all_states_acceptable_confident_correct_match() -> None:
    # Class 0: labeled (label 1 → 0-indexed 0), predicted (prob > thr), confident → "=="
    # Classes 1–5: not labeled, not predicted → ""
    probs = [_CONFIDENT_PROB] + [_BELOW_THR] * 5
    assert script._all_states_acceptable(probs, _THR, [1], frozenset({"=="}))
    assert script._all_states_acceptable(probs, _THR, [1], frozenset({"==", "="}))


def test_all_states_acceptable_borderline_correct_match() -> None:
    # Class 0: labeled, predicted, not confident → "="
    probs = [_BORDERLINE_PROB] + [_BELOW_THR] * 5
    assert not script._all_states_acceptable(probs, _THR, [1], frozenset({"=="}))
    assert script._all_states_acceptable(probs, _THR, [1], frozenset({"="}))
    assert script._all_states_acceptable(probs, _THR, [1], frozenset({"==", "="}))


def test_all_states_acceptable_false_negative_excluded() -> None:
    # select_predictions has min_k=1, so we need class 1 to be the forced prediction
    # to keep class 0 (labeled) unpredicted.
    # Class 0: labeled (labels=[1] → label_set={0}), not predicted, confident → "--"
    # Class 1: not labeled, predicted (prob 0.9 ≥ 0.5), confident → "++"
    probs = [_BELOW_THR, _CONFIDENT_PROB] + [_BELOW_THR] * 4
    assert not script._all_states_acceptable(probs, _THR, [1], frozenset({"==", "="}))
    assert script._all_states_acceptable(probs, _THR, [1], frozenset({"--", "++"}))


def test_all_states_acceptable_false_positive_excluded() -> None:
    # Class 0: not labeled but predicted with high confidence → "++"
    # Class 1: labeled but not predicted, confident → "--"
    probs = [_CONFIDENT_PROB] + [_BELOW_THR] * 5
    assert not script._all_states_acceptable(probs, _THR, [2], frozenset({"=="}))
    assert script._all_states_acceptable(probs, _THR, [2], frozenset({"++", "--"}))


# ---------------------------------------------------------------------------
# find_candidates with require
# ---------------------------------------------------------------------------


def _make_analysis_mock(probs: list[float], thresholds: list[float]) -> MagicMock:
    m = MagicMock()
    m.model_probs = {"key": probs}
    m.model_thresholds = thresholds
    m.get_image_probs.return_value = probs
    return m


def _make_store_mock(labels: list[int]) -> MagicMock:
    m = MagicMock()
    m.path_to_key.return_value = "20240101_000000"
    m.get.return_value = labels
    return m


def test_find_candidates_with_require_includes_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240101_000000.png"
    small_crop_name = "pony_chart_20240101_000000_1.png"

    _make_image(tmp_path / orig_name, (1004, 1004))
    _make_image(tmp_path / small_crop_name, (200, 100))

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(script, "IMAGE_SUBDIR", tmp_path)
    monkeypatch.setattr(script, "LABEL_FILE", tmp_path / "labels.json")

    # Class 0: labeled (label=1), predicted, confident → "=="
    probs = [_CONFIDENT_PROB] + [_BELOW_THR] * 5
    monkeypatch.setattr(
        script, "AnalysisManager", lambda: _make_analysis_mock(probs, _THR)
    )
    monkeypatch.setattr(script, "LabelStore", lambda *a: _make_store_mock([1]))

    assert script.find_candidates(frozenset({"=="})) == [tmp_path / small_crop_name]


def test_find_candidates_with_require_excludes_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240101_000000.png"
    small_crop_name = "pony_chart_20240101_000000_1.png"

    _make_image(tmp_path / orig_name, (1004, 1004))
    _make_image(tmp_path / small_crop_name, (200, 100))

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(script, "IMAGE_SUBDIR", tmp_path)
    monkeypatch.setattr(script, "LABEL_FILE", tmp_path / "labels.json")

    # Class 0: labeled, predicted, not confident → "=" (not in {"=="})
    probs = [_BORDERLINE_PROB] + [_BELOW_THR] * 5
    monkeypatch.setattr(
        script, "AnalysisManager", lambda: _make_analysis_mock(probs, _THR)
    )
    monkeypatch.setattr(script, "LabelStore", lambda *a: _make_store_mock([1]))

    assert script.find_candidates(frozenset({"=="})) == []
    assert script.find_candidates(frozenset({"==", "="})) == [
        tmp_path / small_crop_name
    ]

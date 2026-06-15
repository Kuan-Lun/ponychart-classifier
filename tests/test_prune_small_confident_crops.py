from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import prune_small_confident_crops as script


class _FakeAnalysis:
    def __init__(
        self, model_probs: dict[str, list[float]], model_thresholds: list[float]
    ) -> None:
        self.model_probs = model_probs
        self.model_thresholds = model_thresholds

    def get_image_probs(self, key: str) -> list[float] | None:
        return self.model_probs.get(key)


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size).save(path)


def test_find_candidates_filters_by_size_and_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240101_000000.png"
    small_crop_name = "pony_chart_20240101_000000_1.png"
    large_crop_name = "pony_chart_20240101_000000_2.png"

    _make_image(tmp_path / orig_name, (1004, 1004))
    _make_image(tmp_path / small_crop_name, (200, 100))  # min < 277 -> too small
    _make_image(tmp_path / large_crop_name, (300, 600))  # min >= 277 -> ok

    (tmp_path / "labels.json").write_text(
        json.dumps({orig_name: [1, 2]}), encoding="utf-8"
    )

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(script, "LABEL_FILE", tmp_path / "labels.json")
    monkeypatch.setattr("app.label_images.label_store.IMAGE_DIR", tmp_path)

    probs = {orig_name: [0.9, 0.9, 0.1, 0.1, 0.1, 0.1]}
    thresholds = [0.5] * 6
    monkeypatch.setattr(
        script, "AnalysisManager", lambda: _FakeAnalysis(probs, thresholds)
    )

    candidates = script.find_candidates()
    assert candidates == [tmp_path / small_crop_name]


def test_find_candidates_skips_when_original_prediction_is_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orig_name = "pony_chart_20240202_000000.png"
    small_crop_name = "pony_chart_20240202_000000_1.png"

    _make_image(tmp_path / orig_name, (1004, 1004))
    _make_image(tmp_path / small_crop_name, (200, 100))  # min < 277 -> too small

    (tmp_path / "labels.json").write_text(
        json.dumps({orig_name: [1, 2]}), encoding="utf-8"
    )

    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(script, "LABEL_FILE", tmp_path / "labels.json")
    monkeypatch.setattr("app.label_images.label_store.IMAGE_DIR", tmp_path)

    # Model predicts class 3 instead of class 2 -> not a 100% match.
    probs = {orig_name: [0.9, 0.1, 0.9, 0.1, 0.1, 0.1]}
    thresholds = [0.5] * 6
    monkeypatch.setattr(
        script, "AnalysisManager", lambda: _FakeAnalysis(probs, thresholds)
    )

    assert script.find_candidates() == []

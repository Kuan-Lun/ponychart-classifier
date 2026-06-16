from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts import prune_small_confident_crops as script


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

from __future__ import annotations

from pathlib import Path

import pytest

from app.label_images.analysis import AnalysisManager


def test_seed_prediction_from_labels_writes_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """以標籤合成的機率應同時更新記憶體中的 model_probs 並寫入快取。"""
    analysis = AnalysisManager()
    analysis.model_probs = {"existing.png": [0.1] * 6}

    saved: list[dict[str, list[float]]] = []
    monkeypatch.setattr(
        "app.label_images.analysis.prob_cache.save",
        lambda cache_path, model_path, probs: saved.append(dict(probs)),
    )

    analysis.seed_prediction_from_labels("crop.png", [1, 3])

    assert analysis.model_probs["crop.png"] == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert saved and saved[-1]["crop.png"] == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_seed_prediction_from_labels_noop_without_existing_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """從未執行過自動標註（無快取）時，不應憑空建立快取。"""
    analysis = AnalysisManager()
    analysis.model_probs = None

    called = False

    def fake_save(cache_path: Path, model_path: Path, probs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("app.label_images.analysis.prob_cache.save", fake_save)

    analysis.seed_prediction_from_labels("crop.png", [1])

    assert analysis.model_probs is None
    assert not called


def test_seed_prediction_from_labels_noop_for_empty_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """來源圖未標註時不寫入建議，避免 min_k fallback 選出無意義標籤。"""
    analysis = AnalysisManager()
    analysis.model_probs = {"existing.png": [0.1] * 6}

    called = False

    def fake_save(cache_path: Path, model_path: Path, probs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("app.label_images.analysis.prob_cache.save", fake_save)

    analysis.seed_prediction_from_labels("crop.png", [])

    assert "crop.png" not in analysis.model_probs
    assert not called

from pathlib import Path

import pytest

from ponychart_classifier.inference import PonyChartClassifier, artifacts


def _make_classifier(tmp_path: Path) -> PonyChartClassifier:
    model_path = tmp_path / "model.onnx"
    thresholds_path = tmp_path / "thresholds.json"
    model_path.write_bytes(b"model")
    thresholds_path.write_text("{}", encoding="utf-8")
    artifacts.save_etag(model_path, "model-etag")
    artifacts.save_etag(thresholds_path, "thresholds-etag")
    return PonyChartClassifier(model_path=model_path, thresholds_path=thresholds_path)


def test_has_pending_update_false_when_etags_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clf = _make_classifier(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "remote_etag",
        lambda filename: (
            "model-etag" if filename == artifacts.MODEL_FILENAME else "thresholds-etag"
        ),
    )

    assert clf.has_pending_update() is False


def test_has_pending_update_true_when_model_etag_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clf = _make_classifier(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "remote_etag",
        lambda filename: (
            "new-model-etag"
            if filename == artifacts.MODEL_FILENAME
            else "thresholds-etag"
        ),
    )

    assert clf.has_pending_update() is True


def test_has_pending_update_true_when_thresholds_etag_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clf = _make_classifier(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "remote_etag",
        lambda filename: (
            "model-etag"
            if filename == artifacts.MODEL_FILENAME
            else "new-thresholds-etag"
        ),
    )

    assert clf.has_pending_update() is True


def test_has_pending_update_false_when_remote_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clf = _make_classifier(tmp_path)
    monkeypatch.setattr(artifacts, "remote_etag", lambda filename: None)

    assert clf.has_pending_update() is False

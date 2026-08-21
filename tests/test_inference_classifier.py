from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pytest

import ponychart_classifier
from ponychart_classifier.inference import (
    ImageDecodeError,
    PonyChartClassifier,
    artifacts,
)
from ponychart_classifier.model_spec import PONY_CLASSES, ImageSize


class _FakeInput:
    name = "image"


class _FakeSession:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray[Any, Any]] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(
        self,
        output_names: object,
        inputs: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        del output_names
        self.inputs.append(inputs["image"])
        return [
            np.array(
                [[8.0, -8.0, -8.0, -8.0, -8.0, -8.0]],
                dtype=np.float32,
            )
        ]


def _make_loaded_classifier() -> tuple[PonyChartClassifier, _FakeSession]:
    classifier = PonyChartClassifier()
    session = _FakeSession()
    classifier._loaded = True
    classifier._session = session
    classifier._input_size = ImageSize(height=2, width=3)
    classifier._thresholds = {pony_class: 0.5 for pony_class in PONY_CLASSES}
    return classifier, session


def _encode_png(image: np.ndarray[Any, Any]) -> bytes:
    success, encoded = cv.imencode(".png", image)
    assert success
    return bytes(encoded)


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


@pytest.mark.parametrize(
    "image",
    [
        np.array([[0, 64, 128], [192, 224, 255]], dtype=np.uint8),
        np.array(
            [
                [[0, 10, 20], [30, 40, 50], [60, 70, 80]],
                [[90, 100, 110], [120, 130, 140], [150, 160, 170]],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [[0, 10, 20, 30], [40, 50, 60, 70], [80, 90, 100, 110]],
                [
                    [120, 130, 140, 150],
                    [160, 170, 180, 190],
                    [200, 210, 220, 230],
                ],
            ],
            dtype=np.uint8,
        ),
    ],
    ids=("grayscale", "bgr", "bgra"),
)
def test_predict_bytes_matches_path_prediction(
    image: np.ndarray[Any, Any],
    tmp_path: Path,
) -> None:
    encoded = _encode_png(image)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(encoded)
    classifier, session = _make_loaded_classifier()

    path_result = classifier.predict(image_path)
    bytes_result = classifier.predict_bytes(encoded)

    assert bytes_result == path_result
    assert bytes_result.labels == frozenset({"Twilight Sparkle"})
    assert len(session.inputs) == 2
    np.testing.assert_array_equal(session.inputs[1], session.inputs[0])


@pytest.mark.parametrize("encoded", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_predict_bytes_rejects_invalid_encoded_data(encoded: bytes) -> None:
    classifier = PonyChartClassifier()

    with pytest.raises(ImageDecodeError, match="Encoded image"):
        classifier.predict_bytes(encoded)


def test_predict_bytes_requires_immutable_bytes() -> None:
    classifier = PonyChartClassifier()

    with pytest.raises(TypeError, match="image must be bytes"):
        classifier.predict_bytes(bytearray(b"image"))  # type: ignore[arg-type]


def test_predict_bytes_normalizes_opencv_decode_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = PonyChartClassifier()

    def fail_decode(encoded: object, flags: int) -> None:
        del encoded, flags
        raise cv.error("decode failed")

    monkeypatch.setattr(
        "ponychart_classifier.inference.image_decoding.cv.imdecode",
        fail_decode,
    )

    with pytest.raises(ImageDecodeError, match="could not be decoded") as raised:
        classifier.predict_bytes(b"encoded image")

    assert isinstance(raised.value.__cause__, cv.error)


def test_predict_bytes_is_exported_from_package_root() -> None:
    assert callable(ponychart_classifier.predict_bytes)

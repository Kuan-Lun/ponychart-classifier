"""High-level ONNX classifier orchestration."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ponychart_classifier.model_spec import (
    DISPLAY_NAME_BY_CLASS,
    PONY_CLASSES,
    ImageSize,
    PonyClass,
    parse_class_key,
)

from . import artifacts
from .image_decoding import decode_image_bytes, read_image_path
from .label_selection import select_predictions
from .preprocessing import preprocess_bgr
from .results import (
    ClassThresholds,
    PredictionResult,
    build_prediction_result,
    build_thresholds,
)


def _select_providers() -> list[str]:
    """Return the best available ONNX Runtime execution providers."""
    available = set(ort.get_available_providers())
    for preferred in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
        if preferred in available:
            return [preferred, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _read_input_size(session: Any) -> ImageSize:
    """Recover ``(H, W)`` from the ONNX session's input shape ``[N, C, H, W]``."""
    shape = session.get_inputs()[0].shape
    if (
        len(shape) != 4
        or not isinstance(shape[2], int)
        or not isinstance(shape[3], int)
    ):
        raise RuntimeError(
            f"ONNX input must have fixed H and W (shape [N, C, H, W]); got {shape!r}."
        )
    return ImageSize(height=shape[2], width=shape[3])


class PonyChartClassifier:
    """Lazy-loading ONNX classifier for PonyChart images."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        thresholds_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path or artifacts.DEFAULT_MODEL_PATH)
        self._thresholds_path = Path(
            thresholds_path or artifacts.DEFAULT_THRESHOLDS_PATH
        )
        self._loaded = False
        self._session: Any = None
        self._classes: list[PonyClass] = list(PONY_CLASSES)
        self._thresholds: dict[PonyClass, float] = {}
        self._input_size: ImageSize | None = None

    @property
    def thresholds(self) -> ClassThresholds:
        """Per-class thresholds. Triggers model loading if needed."""
        self.load()
        scores = {
            pony_class: self._thresholds.get(pony_class, 0.5)
            for pony_class in self._classes
        }
        return build_thresholds(scores)

    def clear_artifacts(self) -> None:
        """Clear default classifier-owned artifacts and reset in-memory state."""
        artifacts.clear_artifacts()
        self._loaded = False
        self._session = None
        self._thresholds = {}
        self._input_size = None

    def load(self) -> None:
        """Load the ONNX model and thresholds. Safe to call multiple times."""
        if self._loaded:
            return
        artifacts.ensure_artifact(self._model_path, artifacts.MODEL_FILENAME)
        artifacts.ensure_artifact(self._thresholds_path, artifacts.THRESHOLDS_FILENAME)
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=_select_providers(),
        )
        self._input_size = _read_input_size(self._session)
        with self._thresholds_path.open(encoding="utf-8") as f:
            raw_thresholds = json.load(f)
        if not isinstance(raw_thresholds, dict):
            raise RuntimeError(
                f"Thresholds file must contain a JSON object: {self._thresholds_path}"
            )
        self._thresholds = {
            parse_class_key(str(key)): float(value)
            for key, value in raw_thresholds.items()
        }
        self._loaded = True

    def update(self) -> bool:
        """Check for newer artifacts and download them if available."""
        updated = False
        updated |= artifacts.update_artifact(self._model_path, artifacts.MODEL_FILENAME)
        updated |= artifacts.update_artifact(
            self._thresholds_path,
            artifacts.THRESHOLDS_FILENAME,
        )
        if updated:
            self._loaded = False
            self._session = None
            self._thresholds = {}
            self._input_size = None
        return updated

    def has_pending_update(self) -> bool:
        """Check, without downloading, whether newer artifacts exist remotely."""
        for path, filename in (
            (self._model_path, artifacts.MODEL_FILENAME),
            (self._thresholds_path, artifacts.THRESHOLDS_FILENAME),
        ):
            remote = artifacts.remote_etag(filename)
            if remote is not None and remote != artifacts.local_etag(path):
                return True
        return False

    def predict(
        self,
        img_path: str | Path,
        min_k: int = 1,
        max_k: int = 3,
    ) -> PredictionResult:
        """Predict characters in an image read from a filesystem path."""
        image = read_image_path(img_path)
        return self._predict_bgr(image, min_k=min_k, max_k=max_k)

    def predict_bytes(
        self,
        image: bytes,
        min_k: int = 1,
        max_k: int = 3,
    ) -> PredictionResult:
        """Predict characters directly from encoded in-memory image bytes."""
        decoded = decode_image_bytes(image)
        return self._predict_bgr(decoded, min_k=min_k, max_k=max_k)

    def _predict_bgr(
        self,
        image: np.ndarray[Any, Any],
        *,
        min_k: int,
        max_k: int,
    ) -> PredictionResult:
        """Run prediction for an already decoded OpenCV BGR image."""
        self.load()
        if self._input_size is None:
            raise RuntimeError("Classifier input size is unavailable after loading.")

        input_tensor = preprocess_bgr(image, input_size=self._input_size)
        input_name: str = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: input_tensor})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))

        scores = {self._classes[i]: float(probs[i]) for i in range(len(self._classes))}
        thresholds = [
            self._thresholds.get(pony_class, 0.5) for pony_class in self._classes
        ]
        indices = select_predictions(list(probs), thresholds, min_k=min_k, max_k=max_k)
        picked = [DISPLAY_NAME_BY_CLASS[self._classes[i]] for i in indices]
        return build_prediction_result(scores, picked)

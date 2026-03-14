"""High-level inference API for PonyChart character classification."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import cv2 as cv
import numpy as np
import onnxruntime as ort

from .model_spec import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PRE_RESIZE,
    select_predictions,
)

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)


def _package_dir() -> str:
    return os.path.dirname(__file__)


class PonyChartClassifier:
    """Lazy-loading ONNX classifier for PonyChart images."""

    def __init__(self) -> None:
        self._loaded = False
        self._session: Any = None
        self._classes: list[str] = list(CLASS_NAMES)
        self._thresholds: dict[str, float] = {}

    def load(self) -> None:
        """Load the ONNX model and thresholds. Safe to call multiple times."""
        if self._loaded:
            return

        d = _package_dir()
        model_path = os.path.join(d, "model.onnx")
        th_path = os.path.join(d, "thresholds.json")
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        with open(th_path, encoding="utf-8") as f:
            self._thresholds = json.load(f)
        self._loaded = True

    def _preprocess(self, bgr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """BGR image -> NCHW float32 tensor (matching training transforms)."""
        resized = cv.resize(bgr, (PRE_RESIZE, PRE_RESIZE), interpolation=cv.INTER_AREA)
        offset = (PRE_RESIZE - INPUT_SIZE) // 2
        cropped = resized[offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
        rgb = cv.cvtColor(cropped, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW -> NCHW
        return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(
        self, img_path: str, min_k: int = 1, max_k: int = 3
    ) -> tuple[list[str], dict[str, float]]:
        """Predict characters in a PonyChart image.

        Returns ``(picked_names, scores)`` where *picked_names* is a list of
        selected character names and *scores* maps every class name to its
        sigmoid probability.
        """
        self.load()
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        input_tensor = self._preprocess(img)
        input_name: str = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: input_tensor})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))

        scores = {self._classes[i]: float(probs[i]) for i in range(len(self._classes))}
        thresholds = [self._thresholds.get(c, 0.5) for c in self._classes]
        indices = select_predictions(list(probs), thresholds, min_k=min_k, max_k=max_k)
        picked = [self._classes[i] for i in indices]
        return picked, scores


_default_classifier = PonyChartClassifier()


def predict(
    img_path: str, min_k: int = 1, max_k: int = 3
) -> tuple[list[str], dict[str, float]]:
    """Predict characters using the default classifier instance."""
    return _default_classifier.predict(img_path, min_k=min_k, max_k=max_k)


def preload() -> None:
    """Pre-load the ONNX model to catch dependency issues early."""
    try:
        _default_classifier.load()
    except ImportError as e:
        msg = "onnxruntime failed to load."
        if sys.platform == "win32" and "DLL load failed" in str(e):
            msg += (
                "\nPossible cause: missing Microsoft Visual C++ Redistributable."
                "\nDownload from https://aka.ms/vs/17/release/vc_redist.x64.exe"
            )
        else:
            msg += "\nPlease install: pip install onnxruntime"
        raise RuntimeError(msg) from e

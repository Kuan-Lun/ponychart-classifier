"""PonyChart classifier -- inference constants and prediction utilities."""

import sys

from .inference import ClassThresholds, PonyChartClassifier, PredictionResult

_classifier = PonyChartClassifier()

predict = _classifier.predict
update = _classifier.update
has_pending_update = _classifier.has_pending_update


def clear_artifacts() -> None:
    """Delete the runtime artifact cache directory and reset the default classifier."""
    _classifier.clear_artifacts()


def get_thresholds() -> ClassThresholds:
    """Return per-class thresholds, loading/downloading the model if needed."""
    return _classifier.thresholds


def preload() -> None:
    """Pre-load the ONNX model to catch dependency issues early."""
    try:
        _classifier.load()
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


__all__ = [
    "ClassThresholds",
    "PonyChartClassifier",
    "PredictionResult",
    "clear_artifacts",
    "get_thresholds",
    "has_pending_update",
    "predict",
    "preload",
    "update",
]

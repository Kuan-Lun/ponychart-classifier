"""PonyChart classifier -- inference constants and prediction utilities."""

import os
import sys

from .inference import PonyChartClassifier
from .model_spec import PredictionResult

_pkg_dir = os.path.dirname(__file__)
_classifier = PonyChartClassifier(
    model_path=os.path.join(_pkg_dir, "model.onnx"),
    thresholds_path=os.path.join(_pkg_dir, "thresholds.json"),
)

predict = _classifier.predict


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
    "PonyChartClassifier",
    "PredictionResult",
    "predict",
    "preload",
]

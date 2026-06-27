"""Image preprocessing for ONNX inference."""

from typing import Any, cast

import cv2 as cv
import numpy as np

from ponychart_classifier.model_spec import IMAGENET_MEAN, IMAGENET_STD, ImageSize

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)


def preprocess_bgr(
    image: np.ndarray[Any, Any],
    *,
    input_size: ImageSize,
) -> np.ndarray[Any, Any]:
    """Convert a BGR image into an NCHW float32 tensor."""
    resized = cv.resize(image, input_size.wh(), interpolation=cv.INTER_AREA)
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return cast(
        np.ndarray[Any, Any],
        normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32),
    )

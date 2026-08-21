"""Decode inference images into OpenCV's BGR pixel layout."""

from pathlib import Path
from typing import Any, cast

import cv2 as cv
import numpy as np


class ImageDecodeError(ValueError):
    """Raised when encoded in-memory image data cannot be decoded."""


def read_image_path(path: str | Path) -> np.ndarray[Any, Any]:
    """Read an image path as a three-channel BGR array."""
    image = cast(np.ndarray[Any, Any] | None, cv.imread(str(path), cv.IMREAD_COLOR))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def decode_image_bytes(image: bytes) -> np.ndarray[Any, Any]:
    """Decode immutable encoded image bytes as a three-channel BGR array."""
    if not isinstance(image, bytes):
        raise TypeError("image must be bytes")
    if not image:
        raise ImageDecodeError("Encoded image is empty.")

    encoded: np.ndarray[Any, Any] = np.frombuffer(image, dtype=np.uint8)
    try:
        decoded = cast(
            np.ndarray[Any, Any] | None,
            cv.imdecode(encoded, cv.IMREAD_COLOR),
        )
    except cv.error as error:
        raise ImageDecodeError("Encoded image could not be decoded.") from error
    if decoded is None:
        raise ImageDecodeError("Encoded image could not be decoded.")
    return decoded

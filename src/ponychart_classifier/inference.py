"""High-level inference API for PonyChart character classification."""

from __future__ import annotations

import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import cv2 as cv
import numpy as np
import onnxruntime as ort

from .model_spec import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PRE_RESIZE,
    PredictionResult,
    select_predictions,
)

_BASE_URL = "https://www.csie.ntu.edu.tw/~d06922002/ponychart_classifier"
_logger = logging.getLogger(__name__)


def _make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context, preferring *certifi*'s CA bundle if available."""
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _make_unverified_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _urlopen(req: urllib.request.Request) -> Any:
    """Open *req*, falling back to unverified SSL on certificate errors."""
    try:
        return urllib.request.urlopen(req, context=_make_ssl_context())  # noqa: S310
    except urllib.error.URLError as first:
        if not isinstance(first.reason, ssl.SSLError):
            raise
        _logger.warning(
            "SSL verification failed (%s); retrying without verification.",
            first.reason,
        )
        return urllib.request.urlopen(  # noqa: S310
            req, context=_make_unverified_ssl_context()
        )


_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)


class PonyChartClassifier:
    """Lazy-loading ONNX classifier for PonyChart images."""

    def __init__(self, model_path: str, thresholds_path: str) -> None:
        self._model_path = model_path
        self._thresholds_path = thresholds_path
        self._loaded = False
        self._session: Any = None
        self._classes: list[str] = list(CLASS_NAMES)
        self._thresholds: dict[str, float] = {}

    @staticmethod
    def _download(path: str, filename: str) -> None:
        """Download *filename* from the remote host."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE_URL}/{filename}"
        _logger.info("Downloading %s -> %s", url, p)
        req = urllib.request.Request(url)
        try:
            with _urlopen(req) as resp:
                p.write_bytes(resp.read())
        except HTTPError as e:
            raise HTTPError(
                url,
                e.code,
                f"Failed to download {filename} (HTTP {e.code}).\n"
                f"You can download it manually:\n  {url} -> {p}\n"
                f"If the file no longer exists, please contact the author.",
                e.headers,
                e.fp,
            ) from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to download {filename}: {e.reason}\n"
                f"You can download it manually:\n  {url} -> {p}\n"
                f"If this is an SSL error, try: pip install certifi",
            ) from None

    @staticmethod
    def _ensure_file(path: str, filename: str) -> None:
        """Download *filename* from the remote host if it does not exist locally."""
        if not Path(path).exists():
            PonyChartClassifier._download(path, filename)
            etag = PonyChartClassifier._remote_etag(filename)
            if etag is not None:
                PonyChartClassifier._save_etag(path, etag)

    @staticmethod
    def _remote_etag(filename: str) -> str | None:
        """Return the ETag of *filename* on the remote host, or *None* on failure."""
        url = f"{_BASE_URL}/{filename}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with _urlopen(req) as resp:
                etag: str | None = resp.headers.get("ETag")
                return etag
        except (HTTPError, urllib.error.URLError):
            return None

    @staticmethod
    def _local_etag(path: str) -> str | None:
        """Read the locally stored ETag for *path*, or *None* if absent."""
        etag_path = Path(path).with_suffix(Path(path).suffix + ".etag")
        if etag_path.exists():
            return etag_path.read_text(encoding="utf-8").strip()
        return None

    @staticmethod
    def _save_etag(path: str, etag: str) -> None:
        """Persist *etag* alongside *path*."""
        etag_path = Path(path).with_suffix(Path(path).suffix + ".etag")
        etag_path.write_text(etag, encoding="utf-8")

    def update(self) -> bool:
        """Check for a newer model and download it if available.

        Returns *True* if the model was updated, *False* otherwise.
        """
        updated = False
        for path, filename in [
            (self._model_path, "model.onnx"),
            (self._thresholds_path, "thresholds.json"),
        ]:
            remote = self._remote_etag(filename)
            if remote is None:
                if not Path(path).exists():
                    self._download(path, filename)
                _logger.warning(
                    "Could not check for updates: %s is unreachable.",
                    filename,
                )
                continue
            if remote != self._local_etag(path):
                self._download(path, filename)
                self._save_etag(path, remote)
                updated = True
        if updated:
            self._loaded = False
        return updated

    def load(self) -> None:
        """Load the ONNX model and thresholds. Safe to call multiple times."""
        if not self._loaded:
            self._ensure_file(self._model_path, "model.onnx")
            self._ensure_file(self._thresholds_path, "thresholds.json")
            self._session = ort.InferenceSession(
                self._model_path, providers=["CPUExecutionProvider"]
            )
            with open(self._thresholds_path, encoding="utf-8") as f:
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
    ) -> PredictionResult:
        """Predict characters in a PonyChart image."""
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
        return PredictionResult(
            twilight_sparkle=scores["Twilight Sparkle"],
            rarity=scores["Rarity"],
            fluttershy=scores["Fluttershy"],
            rainbow_dash=scores["Rainbow Dash"],
            pinkie_pie=scores["Pinkie Pie"],
            applejack=scores["Applejack"],
            labels=frozenset(picked),
        )

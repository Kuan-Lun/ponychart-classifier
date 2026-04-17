"""Runtime artifact path resolution and download helpers."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

_BASE_URL = "https://www.csie.ntu.edu.tw/~d06922002/ponychart_classifier"
MODEL_FILENAME = "model.onnx"
THRESHOLDS_FILENAME = "thresholds.json"
_logger = logging.getLogger(__name__)


def _default_artifact_dir() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "ponychart-classifier" / "Cache"
        return Path.home() / "AppData" / "Local" / "ponychart-classifier" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ponychart-classifier"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "ponychart-classifier"
    return Path.home() / ".cache" / "ponychart-classifier"


DEFAULT_ARTIFACT_DIR = _default_artifact_dir()
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / MODEL_FILENAME
DEFAULT_THRESHOLDS_PATH = DEFAULT_ARTIFACT_DIR / THRESHOLDS_FILENAME


class _SSLOpener:
    """URL opener that falls back to unverified SSL once on certificate errors."""

    def __init__(self) -> None:
        self._ctx: ssl.SSLContext | None = None

    @staticmethod
    def _verified_context() -> ssl.SSLContext:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    @staticmethod
    def _unverified_context() -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def urlopen(self, req: urllib.request.Request) -> Any:
        """Open *req*, falling back to unverified SSL on certificate errors."""
        if self._ctx is not None:
            return urllib.request.urlopen(req, context=self._ctx)  # noqa: S310
        try:
            ctx = self._verified_context()
            resp = urllib.request.urlopen(req, context=ctx)  # noqa: S310
            self._ctx = ctx
            return resp
        except URLError as first:
            if not isinstance(first.reason, ssl.SSLError):
                raise
            _logger.warning(
                "SSL verification failed (%s); retrying without verification.",
                first.reason,
            )
            self._ctx = self._unverified_context()
            return urllib.request.urlopen(req, context=self._ctx)  # noqa: S310


@functools.lru_cache(maxsize=1)
def _opener() -> _SSLOpener:
    return _SSLOpener()


def _etag_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".etag")


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    _write_bytes_atomic(path, content.encode("utf-8"))


def remote_etag(filename: str) -> str | None:
    """Return the remote ETag for *filename*, or *None* on failure."""
    url = f"{_BASE_URL}/{filename}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with _opener().urlopen(req) as resp:
            etag: str | None = resp.headers.get("ETag")
            return etag
    except (HTTPError, URLError):
        return None


def local_etag(path: Path) -> str | None:
    """Return the locally cached ETag for *path*, or *None* if absent."""
    etag_path = _etag_path(path)
    if not etag_path.exists():
        return None
    return etag_path.read_text(encoding="utf-8").strip()


def save_etag(path: Path, etag: str) -> None:
    """Persist *etag* next to *path*."""
    _write_text_atomic(_etag_path(path), etag)


def download_artifact(path: Path, filename: str) -> None:
    """Download *filename* into *path* using atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE_URL}/{filename}"
    _logger.info("Downloading %s -> %s", url, path)
    req = urllib.request.Request(url)
    try:
        with _opener().urlopen(req) as resp:
            _write_bytes_atomic(path, resp.read())
    except HTTPError as e:
        raise HTTPError(
            url,
            e.code,
            f"Failed to download {filename} (HTTP {e.code}).\n"
            f"You can download it manually:\n  {url} -> {path}\n"
            f"If the file no longer exists, please contact the author.",
            e.headers,
            e.fp,
        ) from None
    except URLError as e:
        raise RuntimeError(
            f"Failed to download {filename}: {e.reason}\n"
            f"You can download it manually:\n  {url} -> {path}\n"
            f"If this is an SSL error, try: pip install certifi",
        ) from None


def ensure_artifact(path: Path, filename: str) -> None:
    """Ensure *path* exists, downloading it from the remote host if needed."""
    if path.exists():
        return
    download_artifact(path, filename)
    etag = remote_etag(filename)
    if etag is not None:
        save_etag(path, etag)


def update_artifact(path: Path, filename: str) -> bool:
    """Refresh *path* from the remote host when its ETag changes."""
    remote = remote_etag(filename)
    if remote is None:
        if not path.exists():
            download_artifact(path, filename)
        _logger.warning("Could not check for updates: %s is unreachable.", filename)
        return False
    if remote == local_etag(path):
        return False
    download_artifact(path, filename)
    save_etag(path, remote)
    return True


def clear_artifacts() -> None:
    """Delete the entire runtime artifact cache directory if it exists."""
    if not DEFAULT_ARTIFACT_DIR.exists():
        return
    try:
        shutil.rmtree(DEFAULT_ARTIFACT_DIR)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to clear runtime artifacts at {DEFAULT_ARTIFACT_DIR}: {exc}"
        ) from exc

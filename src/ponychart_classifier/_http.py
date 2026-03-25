"""SSL-aware URL opener with automatic fallback to unverified connections."""

from __future__ import annotations

import functools
import logging
import ssl
import urllib.request
from typing import Any

_logger = logging.getLogger(__name__)


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
        except urllib.error.URLError as first:
            if not isinstance(first.reason, ssl.SSLError):
                raise
            _logger.warning(
                "SSL verification failed (%s); retrying without verification.",
                first.reason,
            )
            self._ctx = self._unverified_context()
            return urllib.request.urlopen(req, context=self._ctx)  # noqa: S310


@functools.lru_cache(maxsize=1)
def opener() -> _SSLOpener:
    """Return the singleton SSL-aware URL opener (created on first call)."""
    return _SSLOpener()

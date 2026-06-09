"""Probability cache：跨 session 持久化模型預測結果。

以 model.onnx 的 SHA-256 作為 invalidation key；
模型更新後 cache 自動失效，下次分析會重新計算。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load(cache_path: Path, model_path: Path) -> dict[str, list[float]] | None:
    """讀取 cache。若 cache 不存在、格式錯誤、或 model hash 不符則回傳 None。"""
    if not cache_path.exists() or not model_path.exists():
        return None
    try:
        with cache_path.open(encoding="utf-8") as f:
            data: object = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("model_sha256") != _sha256(model_path):
            return None
        probs = data.get("probs")
        if not isinstance(probs, dict):
            return None
        return probs
    except Exception:
        return None


def save(
    cache_path: Path,
    model_path: Path,
    probs: dict[str, list[float]],
) -> None:
    """將 probs 與當前 model hash 寫入 cache（atomic write）。"""
    data = {
        "model_sha256": _sha256(model_path),
        "probs": probs,
    }
    tmp_fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        Path(tmp_name).replace(cache_path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

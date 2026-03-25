"""標籤資料的載入、正規化、查詢與持久化。"""

import json
from pathlib import Path

from .constants import IMAGE_DIR


class LabelStore:
    """標籤資料的載入、正規化、查詢與持久化。"""

    def __init__(self, label_file: Path, image_subdir: str):
        self._file = label_file
        self._subdir = image_subdir
        self._data: dict[str, list[int]] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            self._data = self._normalize(raw)
        except Exception:
            self._data = {}

    def _normalize(self, raw: dict[str, list[int]]) -> dict[str, list[int]]:
        """正規化舊的 key 格式。

        新格式 key 相對於 IMAGE_DIR（例如 1/twilight/xxx.png）。
        舊格式 rawimage/... 會自動去掉前綴。
        """
        norm: dict[str, list[int]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            kk = k.replace("\\", "/")
            if kk.startswith("data/"):
                kk = kk[len("data/") :]
            if kk.startswith(self._subdir + "/"):
                kk = kk[len(self._subdir) + 1 :]
            if kk:
                norm[kk] = v
        return norm

    def get(self, key: str) -> list[int]:
        return list(self._data.get(key, []))

    def set(self, key: str, labels: list[int]) -> None:
        if labels:
            self._data[key] = labels
        elif key in self._data:
            del self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str) -> None:
        """刪除指定 key 的標籤。"""
        self._data.pop(key, None)

    def rename_key(self, old_key: str, new_key: str) -> None:
        """將 key 更名（搬移檔案後用）。"""
        if old_key in self._data:
            self._data[new_key] = self._data.pop(old_key)

    def all_keys(self) -> list[str]:
        """回傳所有 key。"""
        return list(self._data)

    def purge_orphans(self, base_dir: Path) -> list[str]:
        """移除檔案不存在的孤兒 entries，回傳被清除的 keys。"""
        orphans = [k for k in self._data if not (base_dir / k).is_file()]
        for k in orphans:
            del self._data[k]
        return orphans

    def save(self) -> None:
        self._file.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def path_to_key(self, p: Path) -> str:
        """將絕對路徑轉為 labels.json 的 key（相對於 IMAGE_DIR）。

        支援子資料夾結構，例如 1/twilight/xxx.png。
        """
        try:
            return str(p.relative_to(IMAGE_DIR))
        except ValueError:
            return p.name

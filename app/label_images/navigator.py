"""圖片列表管理、過濾與導航。"""

from collections.abc import Callable
from pathlib import Path

from .label_store import LabelStore


class ImageNavigator:
    """圖片列表管理、過濾與導航。"""

    def __init__(self, all_paths: list[Path], store: LabelStore):
        self._all_paths = all_paths
        self._store = store
        self._paths = list(all_paths)
        self._idx = 0
        self._filter_fn: Callable[[Path], bool] | None = None

    @property
    def total(self) -> int:
        return len(self._paths)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def is_empty(self) -> bool:
        return len(self._paths) == 0

    @property
    def is_filtered(self) -> bool:
        return self._filter_fn is not None

    @property
    def current_path(self) -> Path:
        return self._paths[self._idx]

    @property
    def current_key(self) -> str:
        return self._store.path_to_key(self._paths[self._idx])

    @property
    def all_paths(self) -> list[Path]:
        return self._all_paths

    def go_next(self) -> None:
        self._idx = (self._idx + 1) % len(self._paths)

    def go_prev(self) -> None:
        self._idx = (self._idx - 1) % len(self._paths)

    def go_to(self, n: int) -> bool:
        """跳到第 n 張（1-based）。回傳是否成功。"""
        if 1 <= n <= len(self._paths):
            self._idx = n - 1
            return True
        return False

    def go_to_key(self, key: str) -> None:
        """跳到指定 key 的圖片。"""
        self._idx = self._find_index_by_key(key)

    def apply_filter(self, fn: Callable[[Path], bool] | None) -> bool:
        """套用篩選函式。回傳 False 表示篩選結果為空。"""
        current_key = self.current_key if not self.is_empty else None
        self._filter_fn = fn
        if fn is not None:
            self._paths = [p for p in self._all_paths if fn(p)]
            if not self._paths:
                self._filter_fn = None
                return False
        else:
            self._paths = list(self._all_paths)
        if current_key:
            self._idx = self._find_index_by_key(current_key)
        else:
            self._idx = 0
        return True

    def refresh_filter(self) -> None:
        """重新套用目前的過濾設定（例如 save 後）。"""
        if self._filter_fn is not None:
            self._paths = [p for p in self._all_paths if self._filter_fn(p)]
        else:
            self._paths = list(self._all_paths)
        if self._paths:
            self._idx = min(self._idx, len(self._paths) - 1)
        else:
            self._idx = 0

    def advance_after_label(self, labeled_key: str) -> None:
        """在過濾模式下，從 labeled_key 往後找下一個符合篩選的圖片。"""
        try:
            start = next(
                i
                for i, p in enumerate(self._all_paths)
                if self._store.path_to_key(p) == labeled_key
            )
        except StopIteration:
            self._idx = 0
            return

        all_len = len(self._all_paths)
        for offset in range(1, all_len + 1):
            candidate = self._all_paths[(start + offset) % all_len]
            if self._filter_fn is None or self._filter_fn(candidate):
                try:
                    self._idx = next(
                        i for i, p in enumerate(self._paths) if p == candidate
                    )
                except StopIteration:
                    self._idx = 0
                return
        self._idx = 0

    def add_path(self, path: Path) -> None:
        """加入新圖片（裁切產生）並跳轉到該圖片。"""
        self._all_paths.append(path)
        self._all_paths.sort()
        self.refresh_filter()
        try:
            self._idx = next(i for i, p in enumerate(self._paths) if p == path)
        except StopIteration:
            pass

    def replace_path(self, old_path: Path, new_path: Path) -> None:
        """替換路徑（檔案搬移後用），保持目前索引位置。"""
        for i, p in enumerate(self._all_paths):
            if p == old_path:
                self._all_paths[i] = new_path
                break
        for i, p in enumerate(self._paths):
            if p == old_path:
                self._paths[i] = new_path
                break

    def remove_path(self, path: Path) -> None:
        """移除圖片並調整索引。"""
        self._all_paths = [p for p in self._all_paths if p != path]
        self._paths = [p for p in self._paths if p != path]
        if self._paths:
            self._idx = min(self._idx, len(self._paths) - 1)
        else:
            self._idx = 0

    def contains_key(self, key: str) -> bool:
        """目前篩選後的列表中是否包含指定 key。"""
        return any(self._store.path_to_key(p) == key for p in self._paths)

    def _find_index_by_key(self, key: str) -> int:
        try:
            return next(
                i
                for i, p in enumerate(self._paths)
                if self._store.path_to_key(p) == key
            )
        except StopIteration:
            return min(self._idx, len(self._paths) - 1) if self._paths else 0

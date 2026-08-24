"""模型分析：背景推論與結果表格顯示。"""

import json
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import ponychart_classifier as _pkg
from ponychart_classifier.inference import artifacts
from ponychart_classifier.inference.label_selection import select_predictions
from ponychart_classifier.model_spec import PONY_CLASSES, parse_class_key
from ponychart_classifier.training.sampling import Sample

from . import prob_cache
from .constants import (
    ANALYSIS_CACHE_FILE,
    CLASS_NAMES_LIST,
    LABEL_MAP,
    SUSPICIOUS_MARGIN,
)
from .label_store import LabelStore
from .navigator import ImageNavigator
from .source_identity import parse_key_identity


def _load_thresholds(path: Path) -> list[float] | None:
    """從 thresholds.json 直接讀取 threshold list，不需載入 ONNX model。"""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            raw: object = json.load(f)
        if not isinstance(raw, dict):
            return None
        parsed = {parse_class_key(str(k)): float(v) for k, v in raw.items()}
        return [parsed.get(c, 0.5) for c in PONY_CLASSES]
    except Exception:
        return None


class AnalysisManager:
    """管理模型分析的背景執行緒與結果。"""

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self.model_probs: dict[str, list[float]] | None = None
        self.model_thresholds: list[float] | None = None
        self._thread: threading.Thread | None = None
        self._result: tuple[dict[str, list[float]], list[float]] | None = None
        self._error: str | None = None
        self._progress: tuple[int, int] | None = None
        self._active_keys: set[str] = set()
        self._tombstones: set[str] = set()
        self._cache_path: Path = ANALYSIS_CACHE_FILE
        self._model_path: Path = artifacts.DEFAULT_MODEL_PATH
        self.model_probs = prob_cache.load(self._cache_path, self._model_path)
        if self.model_probs is not None:
            self.model_thresholds = _load_thresholds(artifacts.DEFAULT_THRESHOLDS_PATH)

    def refresh_staleness(self) -> bool:
        """檢查遠端模型/thresholds 是否已更新；若是則捨棄記憶體中的舊預測結果。

        prob_cache 以 model.onnx 的 sha256 作為整包 invalidation key，只有
        在下一次全量分析寫回快取時才會更新；若模型在快取寫回前就已更新，
        既有預測會被誤判為「已分析」而永遠不會重新推論。啟動時檢查一次，
        讓所有圖片重新回到「尚未分析」狀態，交由既有的補跑邏輯全部重算。
        """
        with self._state_lock:
            if self.model_probs is None:
                return False
        if not _pkg.has_pending_update():
            return False
        with self._state_lock:
            self.model_probs = None
            self.model_thresholds = None
        return True

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def has_results(self) -> bool:
        with self._state_lock:
            return self.model_probs is not None

    def collect_samples(
        self,
        nav: ImageNavigator,
        store: LabelStore,
        key_filter: Callable[[str], bool] | None = None,
    ) -> tuple[list[Sample], list[str]]:
        """走訪 ``nav.all_paths`` 並用 ``key_filter`` 篩出要分析的樣本。

        ``key_filter`` 為 None 時不過濾。回傳 ``(samples, keys)``，
        呼叫端可由 ``len(samples)`` 取得待分析張數。
        """
        samples: list[Sample] = []
        keys: list[str] = []
        for p in nav.all_paths:
            key = store.path_to_key(p)
            if key_filter is not None and not key_filter(key):
                continue
            samples.append(Sample(str(p), store.get(key)))
            keys.append(key)
        return samples, keys

    def start(
        self,
        samples: list[Sample],
        keys: list[str],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        root: tk.Tk,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """啟動背景分析。新預測一律合併進 ``model_probs``，不會覆蓋舊資料。"""
        if self.is_running:
            return

        with self._state_lock:
            self._result = None
            self._error = None
            self._progress = None
            self._active_keys = set(keys)
            self._tombstones.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(samples, keys),
                daemon=True,
            )
            self._thread.start()

        def poll() -> None:
            if self.is_running:
                with self._state_lock:
                    progress = self._progress
                if on_progress is not None and progress is not None:
                    done, total = progress
                    on_progress(done, total)
                root.after(200, poll)
                return
            with self._state_lock:
                self._thread = None
                error = self._error
                result = self._result
                self._error = None
                self._result = None
                cache_probs: dict[str, list[float]] | None = None
                if error is None and result is not None:
                    new_probs, new_thresholds = result
                    new_probs = {
                        key: probs
                        for key, probs in new_probs.items()
                        if key not in self._tombstones
                    }
                    if self.model_probs is None:
                        self.model_probs = new_probs
                    else:
                        merged = dict(self.model_probs)
                        merged.update(new_probs)
                        self.model_probs = merged
                    self.model_thresholds = new_thresholds
                    cache_probs = (
                        dict(self.model_probs) if self.model_probs is not None else None
                    )
                self._active_keys.clear()
                self._tombstones.clear()
            if error is not None:
                on_error(error)
                return
            if result is not None:
                if cache_probs is not None:
                    prob_cache.save(
                        self._cache_path,
                        self._model_path,
                        cache_probs,
                    )
                on_complete()

        root.after(200, poll)

    def _run(
        self,
        samples: list[Sample],
        keys: list[str],
    ) -> None:
        try:
            _pkg.update()
            thresholds = _pkg.get_thresholds().as_list()
            result: dict[str, list[float]] = {}
            total = len(samples)
            for i, ((img_path, _labels), key) in enumerate(zip(samples, keys)):
                try:
                    pred = _pkg.predict(img_path)
                except Exception:
                    with self._state_lock:
                        if key in self._tombstones:
                            self._progress = (i + 1, total)
                            continue
                    raise
                probs = [
                    pred.twilight_sparkle,
                    pred.rarity,
                    pred.fluttershy,
                    pred.rainbow_dash,
                    pred.pinkie_pie,
                    pred.applejack,
                ]
                with self._state_lock:
                    if key not in self._tombstones:
                        result[key] = probs
                    self._progress = (i + 1, total)
            with self._state_lock:
                result = {
                    key: probs
                    for key, probs in result.items()
                    if key not in self._tombstones
                }
                self._result = (result, thresholds)
        except Exception as e:
            with self._state_lock:
                self._error = str(e)

    def get_image_probs(self, key: str) -> list[float] | None:
        """取得指定圖片的模型預測機率。"""
        with self._state_lock:
            if self.model_probs is None:
                return None
            return self.model_probs.get(key)

    def rename_key(self, old_key: str, new_key: str) -> None:
        """同步圖片搬移後的 key 變更，避免遺失既有預測結果。"""
        with self._state_lock:
            if old_key in self._active_keys:
                self._tombstones.add(old_key)
            if self.model_probs is not None and old_key in self.model_probs:
                self.model_probs[new_key] = self.model_probs.pop(old_key)
            if self._result is not None:
                pending, thresholds = self._result
                if old_key in pending:
                    pending = dict(pending)
                    pending[new_key] = pending.pop(old_key)
                    self._result = (pending, thresholds)

    def delete_key(self, key: str) -> None:
        """移除已刪除圖片對應的預測結果。"""
        with self._state_lock:
            if key in self._active_keys:
                self._tombstones.add(key)
            if self.model_probs is not None:
                self.model_probs.pop(key, None)
            if self._result is not None and key in self._result[0]:
                pending, thresholds = self._result
                pending = dict(pending)
                pending.pop(key, None)
                self._result = (pending, thresholds)

    def delete_keys(self, keys: list[str]) -> None:
        """批次移除多張已不存在圖片的預測結果。"""
        for key in keys:
            self.delete_key(key)

    def purge_source(self, source_stem: str) -> list[str]:
        """Remove every cached, pending, or active key for one retired source."""
        with self._state_lock:
            keys: set[str] = set(self._active_keys)
            if self.model_probs is not None:
                keys.update(self.model_probs)
            if self._result is not None:
                keys.update(self._result[0])
            matches = {
                key
                for key in keys
                if (parsed := parse_key_identity(key)) is not None
                and parsed.source_stem == source_stem
            }
            self._tombstones.update(matches & self._active_keys)
            if self.model_probs is not None:
                for key in matches:
                    self.model_probs.pop(key, None)
            if self._result is not None:
                pending, thresholds = self._result
                self._result = (
                    {
                        key: value
                        for key, value in pending.items()
                        if key not in matches
                    },
                    thresholds,
                )
            return sorted(matches)

    @staticmethod
    def _safe_existing_cache_path(image_dir: Path, key: str) -> bool:
        normalized = key.replace("\\", "/")
        relative = PurePosixPath(normalized)
        if (
            not normalized
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            return False
        path = image_dir.joinpath(*relative.parts)
        try:
            path.resolve(strict=False).relative_to(image_dir.resolve(strict=True))
        except OSError, ValueError:
            return False
        return path.is_file()

    def purge_missing_files(self, image_dir: Path) -> list[str]:
        """Drop startup cache orphans while retaining any existing image file."""
        with self._state_lock:
            if self.model_probs is None:
                return []
            missing = sorted(
                key
                for key in self.model_probs
                if not self._safe_existing_cache_path(image_dir, key)
            )
        self.delete_keys(missing)
        return missing

    def save_cache(self) -> None:
        """將目前的 model_probs 寫回 analysis_cache.json。"""
        with self._state_lock:
            probs = dict(self.model_probs) if self.model_probs is not None else None
        if probs is not None:
            prob_cache.save(self._cache_path, self._model_path, probs)

    def save_cache_fail_safe(self) -> bool:
        """Persist derived predictions or invalidate them completely on failure.

        Label/file changes are the source of truth and may already be committed when
        this runs. A stale cache is therefore removed and in-memory predictions are
        cleared instead of letting a cache error leave keys pointing at old paths.
        """
        with self._state_lock:
            if self.model_probs is None:
                return True
        try:
            self.save_cache()
        except Exception:
            with self._state_lock:
                self.model_probs = None
                self.model_thresholds = None
            try:
                self._cache_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise RuntimeError(
                    "Prediction cache save failed and stale cache removal also failed"
                ) from cleanup_error
            return False
        return True


class AnalysisTable:
    """模型分析結果的表格 UI 元件。"""

    def __init__(self, parent: tk.Misc, on_confirm_all: Callable[[], None]) -> None:
        self._frame = tk.Frame(parent)
        self._labels: dict[tuple[int, int], tk.Label] = {}
        self._build(on_confirm_all)

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def _build(self, on_confirm_all: Callable[[], None]) -> None:
        row_headers = ["Prob", "Label"]
        col_headers = list(LABEL_MAP.values())

        tk.Button(
            self._frame,
            text="✓ 套用預測",
            width=10,
            command=on_confirm_all,
        ).grid(row=0, column=0)
        for c, name in enumerate(col_headers):
            tk.Label(
                self._frame,
                text=name,
                font=("Consolas", 10, "bold"),
                width=12,
            ).grid(row=0, column=c + 1, padx=2)

        for r, header in enumerate(row_headers):
            tk.Label(
                self._frame,
                text=header,
                font=("Consolas", 10, "bold"),
                width=10,
                anchor="w",
            ).grid(row=r + 1, column=0, padx=(0, 4))
            for c in range(len(col_headers)):
                lbl = tk.Label(
                    self._frame,
                    text="",
                    font=("Consolas", 10),
                    width=12,
                )
                lbl.grid(row=r + 1, column=c + 1, padx=2)
                self._labels[(r, c)] = lbl

    def update(
        self,
        probs: list[float] | None,
        thresholds: list[float] | None,
        current_labels: list[int],
        anchor_widget: tk.Misc,
    ) -> None:
        """更新表格內容。若無資料則隱藏。"""
        if probs is None or thresholds is None:
            self._frame.pack_forget()
            return

        self._frame.pack(before=anchor_widget, pady=(0, 4))
        predicted_set = set(select_predictions(probs, thresholds))

        for c in range(len(CLASS_NAMES_LIST)):
            prob = probs[c]
            thr = thresholds[c]
            has_label = (c + 1) in current_labels
            pred = c in predicted_set

            self._labels[(0, c)].configure(text=f"{prob:.2f}")

            confident = abs(prob - thr) >= SUSPICIOUS_MARGIN
            if has_label and pred:
                text = "==" if confident else "="
            elif has_label and not pred:
                text = "−−" if confident else "−"
            elif not has_label and pred:
                text = "++" if confident else "+"
            else:
                text = ""
            self._labels[(1, c)].configure(text=text)

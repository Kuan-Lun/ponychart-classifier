"""模型分析：背景推論與結果表格顯示。"""

import json
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

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
        self.model_probs: dict[str, list[float]] | None = None
        self.model_thresholds: list[float] | None = None
        self._thread: threading.Thread | None = None
        self._result: tuple[dict[str, list[float]], list[float]] | None = None
        self._error: str | None = None
        self._progress: tuple[int, int] | None = None
        self._cache_path: Path = ANALYSIS_CACHE_FILE
        self._model_path: Path = artifacts.DEFAULT_MODEL_PATH
        self.model_probs = prob_cache.load(self._cache_path, self._model_path)
        if self.model_probs is not None:
            self.model_thresholds = _load_thresholds(artifacts.DEFAULT_THRESHOLDS_PATH)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def has_results(self) -> bool:
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

        self._result = None
        self._error = None
        self._progress = None
        self._thread = threading.Thread(
            target=self._run,
            args=(samples, keys),
            daemon=True,
        )
        self._thread.start()

        def poll() -> None:
            if self.is_running:
                if on_progress is not None and self._progress is not None:
                    done, total = self._progress
                    on_progress(done, total)
                root.after(200, poll)
                return
            self._thread = None
            if self._error is not None:
                err = self._error
                self._error = None
                on_error(err)
                return
            if self._result is not None:
                new_probs, new_thresholds = self._result
                self._result = None
                if self.model_probs is None:
                    self.model_probs = new_probs
                else:
                    merged = dict(self.model_probs)
                    merged.update(new_probs)
                    self.model_probs = merged
                self.model_thresholds = new_thresholds
                if self.model_probs is not None:
                    prob_cache.save(
                        self._cache_path,
                        self._model_path,
                        self.model_probs,
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
                pred = _pkg.predict(img_path)
                result[key] = [
                    pred.twilight_sparkle,
                    pred.rarity,
                    pred.fluttershy,
                    pred.rainbow_dash,
                    pred.pinkie_pie,
                    pred.applejack,
                ]
                self._progress = (i + 1, total)
            self._result = (result, thresholds)
        except Exception as e:
            self._error = str(e)

    def get_image_probs(self, key: str) -> list[float] | None:
        """取得指定圖片的模型預測機率。"""
        if self.model_probs is None:
            return None
        return self.model_probs.get(key)

    def rename_key(self, old_key: str, new_key: str) -> None:
        """同步圖片搬移後的 key 變更，避免遺失既有預測結果。"""
        if self.model_probs is None:
            return
        if old_key in self.model_probs:
            self.model_probs[new_key] = self.model_probs.pop(old_key)

    def delete_key(self, key: str) -> None:
        """移除已刪除圖片對應的預測結果。"""
        if self.model_probs is None:
            return
        self.model_probs.pop(key, None)

    def delete_keys(self, keys: list[str]) -> None:
        """批次移除多張已不存在圖片的預測結果。"""
        for key in keys:
            self.delete_key(key)


class AnalysisTable:
    """模型分析結果的表格 UI 元件。"""

    def __init__(self, parent: tk.Misc) -> None:
        self._frame = tk.Frame(parent)
        self._labels: dict[tuple[int, int], tk.Label] = {}
        self._build()

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def _build(self) -> None:
        row_headers = ["Prob", "Label"]
        col_headers = list(LABEL_MAP.values())

        tk.Label(self._frame, text="", width=10).grid(row=0, column=0)
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

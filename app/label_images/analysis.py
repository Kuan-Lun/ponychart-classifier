"""模型分析：背景推論與結果表格顯示。"""

import threading
import tkinter as tk
from collections.abc import Callable

import ponychart_classifier as _pkg
from ponychart_classifier.model_spec import select_predictions

from .constants import CLASS_NAMES_LIST, LABEL_MAP, SUSPICIOUS_MARGIN
from .label_store import LabelStore
from .navigator import ImageNavigator


class AnalysisManager:
    """管理模型分析的背景執行緒與結果。"""

    def __init__(self) -> None:
        self.model_probs: dict[str, list[float]] | None = None
        self.model_thresholds: list[float] | None = None
        self._thread: threading.Thread | None = None
        self._result: tuple[dict[str, list[float]], list[float]] | None = None
        self._error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def has_results(self) -> bool:
        return self.model_probs is not None

    def start(
        self,
        nav: ImageNavigator,
        store: LabelStore,
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        root: tk.Tk,
    ) -> None:
        """啟動背景分析。"""
        if self.is_running:
            return

        samples: list[tuple[str, list[int]]] = []
        keys: list[str] = []
        for p in nav.all_paths:
            key = store.path_to_key(p)
            samples.append((str(p), store.get(key)))
            keys.append(key)

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(samples, keys),
            daemon=True,
        )
        self._thread.start()

        def poll() -> None:
            if self.is_running:
                root.after(200, poll)
                return
            self._thread = None
            if self._error is not None:
                err = self._error
                self._error = None
                on_error(err)
                return
            if self._result is not None:
                self.model_probs, self.model_thresholds = self._result
                self._result = None
                on_complete()

        root.after(200, poll)

    def _run(
        self,
        samples: list[tuple[str, list[int]]],
        keys: list[str],
    ) -> None:
        try:
            _pkg.update()
            thresholds = _pkg.get_thresholds().as_list()
            result: dict[str, list[float]] = {}
            for (img_path, _labels), key in zip(samples, keys):
                pred = _pkg.predict(img_path)
                result[key] = [
                    pred.twilight_sparkle,
                    pred.rarity,
                    pred.fluttershy,
                    pred.rainbow_dash,
                    pred.pinkie_pie,
                    pred.applejack,
                ]
            self._result = (result, thresholds)
        except Exception as e:
            self._error = str(e)

    def get_image_probs(self, key: str) -> list[float] | None:
        """取得指定圖片的模型預測機率。"""
        if self.model_probs is None:
            return None
        return self.model_probs.get(key)


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

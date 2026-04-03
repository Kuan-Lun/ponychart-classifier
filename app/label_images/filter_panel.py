"""篩選面板 UI 元件：管理所有篩選相關的 checkbox 狀態與 UI。"""

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .constants import CLASS_NAMES_LIST
from .filter_builder import FilterConfig, build_filter_fn
from .label_store import LabelStore
from .navigator import ImageNavigator


class FilterPanel:
    """管理所有篩選 checkbox 的 UI 與狀態。"""

    def __init__(
        self,
        root: tk.Tk,
        *,
        on_filter_changed: Callable[[], None],
    ) -> None:
        self._on_filter_changed = on_filter_changed
        self._build_filter_ui(root)
        self._build_class_filter_ui(root)

    # ── UI 建構 ──────────────────────────────────────────────

    def _build_filter_ui(self, root: tk.Tk) -> None:
        filter_frame = tk.Frame(root)
        filter_frame.pack(pady=(0, 4))

        self.raw_only_var = tk.BooleanVar(value=False)
        self._raw_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示原圖",
            variable=self.raw_only_var,
            command=self._on_raw_toggle,
        )
        self._raw_only_cb.pack(side="left", padx=(0, 8))

        self.uncropped_only_var = tk.BooleanVar(value=False)
        self._uncropped_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示未裁切原圖",
            variable=self.uncropped_only_var,
            command=self._on_uncropped_toggle,
        )
        self._uncropped_only_cb.pack(side="left", padx=(0, 8))

        self.crop_mismatch_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="裁切標籤不符",
            variable=self.crop_mismatch_var,
            command=self._on_filter_changed,
        ).pack(side="left", padx=(0, 8))

        self.filter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="只顯示未標註",
            variable=self.filter_var,
            command=self._on_filter_changed,
        ).pack(side="left", padx=(0, 8))

        self.train_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="只顯示 Train",
            variable=self.train_only_var,
            command=self._on_filter_changed,
        ).pack(side="left")

    def _build_class_filter_ui(self, root: tk.Tk) -> None:
        class_frame = tk.Frame(root)
        class_frame.pack(pady=(0, 4))
        tk.Label(class_frame, text="Class:").pack(side="left")
        self._class_vars: list[tk.BooleanVar] = []
        for short in CLASS_NAMES_LIST:
            var = tk.BooleanVar(value=False)
            self._class_vars.append(var)
            tk.Checkbutton(
                class_frame,
                text=short,
                variable=var,
                command=self._on_filter_changed,
            ).pack(side="left", padx=(2, 2))

    def build_suspicious_checkboxes(self, parent: tk.Frame) -> None:
        """在指定的 frame 中建構 Mislabel / Missing checkbox。"""
        self._mislabel_var = tk.BooleanVar(value=False)
        self._mislabel_cb = tk.Checkbutton(
            parent,
            text="Mislabel (−)",
            variable=self._mislabel_var,
            command=self._on_filter_changed,
        )
        self._mislabel_cb.pack(side="left", padx=(0, 4))

        self._missing_var = tk.BooleanVar(value=False)
        self._missing_cb = tk.Checkbutton(
            parent,
            text="Missing (+)",
            variable=self._missing_var,
            command=self._on_filter_changed,
        )
        self._missing_cb.pack(side="left", padx=(0, 4))

        self.set_suspicious_state("disabled")

    # ── 篩選邏輯 ─────────────────────────────────────────────

    def _on_raw_toggle(self) -> None:
        if not self.raw_only_var.get():
            self.uncropped_only_var.set(False)
            self._uncropped_only_cb.configure(state="normal")
        self._on_filter_changed()

    def _on_uncropped_toggle(self) -> None:
        if self.uncropped_only_var.get():
            self.raw_only_var.set(True)
            self._raw_only_cb.configure(state="disabled")
        else:
            self._raw_only_cb.configure(state="normal")
        self._on_filter_changed()

    def build_filter_fn(
        self,
        nav: ImageNavigator,
        store: LabelStore,
        *,
        model_probs: dict[str, list[float]] | None,
        model_thresholds: list[float] | None,
    ) -> Callable[[Path], bool] | None:
        """根據目前的 checkbox 狀態建構篩選函式。"""
        config = FilterConfig(
            raw_only=self.raw_only_var.get(),
            uncropped_only=self.uncropped_only_var.get(),
            unlabeled_only=self.filter_var.get(),
            crop_mismatch=self.crop_mismatch_var.get(),
            train_only=self.train_only_var.get(),
            selected_classes=[i for i, var in enumerate(self._class_vars) if var.get()],
            show_mislabel=self._mislabel_var.get(),
            show_missing=self._missing_var.get(),
            model_probs=model_probs,
            model_thresholds=model_thresholds,
        )
        return build_filter_fn(config, nav.all_paths, store)

    def reset(self) -> None:
        """重設所有篩選條件。"""
        self.raw_only_var.set(False)
        self.uncropped_only_var.set(False)
        self.crop_mismatch_var.set(False)
        self.filter_var.set(False)
        self.train_only_var.set(False)
        self._raw_only_cb.configure(state="normal")
        self._uncropped_only_cb.configure(state="normal")
        self._mislabel_var.set(False)
        self._missing_var.set(False)
        for var in self._class_vars:
            var.set(False)

    def set_suspicious_state(
        self,
        state: Literal["normal", "disabled"],
    ) -> None:
        """設定 suspicious filter checkbox 的啟用狀態。"""
        self._mislabel_cb.configure(state=state)
        self._missing_cb.configure(state=state)

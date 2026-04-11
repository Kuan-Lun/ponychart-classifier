"""非模態視窗：背景載入、生命週期、section 組裝。

架構說明：
- Section 為獨立的渲染單位，互不相依，可自由組合。
- `_BaseViewer` 負責視窗生命週期、背景載入與輪詢，呼叫子類提供的
  `_load_async` / `_build_sections` hook。
- `_CheckpointViewer` 追加「checkpoint.pt 必須存在」的前置檢查。

新增一個 section 時：實作 `Section.render`，再於對應的 viewer 的
`_build_sections` 中加入即可，不需修改任何現有類別。
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Any

from ..label_store import LabelStore
from .extractors import (
    CHECKPOINT_PATH,
    extract_changes,
    extract_counts,
    extract_hyperparams,
    extract_model,
    extract_split_counts,
    load_checkpoint,
)
from .sections import (
    ChangesSection,
    DistributionTestSection,
    HyperparamsSection,
    ImageCountsSection,
    ModelArchSection,
    NoticeSection,
    Section,
    SplitCountsSection,
    ValF1Section,
)
from .stats import snapshot_orig_samples
from .widgets import FONT


class _BaseViewer:
    """非模態視窗骨架：背景載入 → 渲染 sections → 重新載入。

    子類別實作：
    - `_preflight()`：視窗開啟前的同步檢查（可選）。
    - `_snapshot_ui_state()`：在 UI 執行緒抓取當前 UI/store 狀態（可選）。
    - `_load_async(ui_state)`：背景執行緒載入資料。
    - `_build_sections(data)`：由資料建立 sections 列表。
    """

    _title: str = ""

    def __init__(self, parent: tk.Tk) -> None:
        self._parent = parent
        self._win: tk.Toplevel | None = None
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._loading_label: tk.Label | None = None

    # ── Lifecycle ──────────────────────────────────────────

    def show(self) -> None:
        if not self._preflight():
            return
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            self._win.focus_force()
            return
        self._win = tk.Toplevel(self._parent)
        self._win.title(self._title)
        self._win.resizable(False, False)
        self._start_load()

    def _start_load(self) -> None:
        assert self._win is not None
        self._loading_label = tk.Label(
            self._win, text="載入中...", font=FONT, padx=40, pady=20
        )
        self._loading_label.pack()
        ui_state = self._snapshot_ui_state()
        self._thread = threading.Thread(
            target=self._load, args=(ui_state,), daemon=True
        )
        self._thread.start()
        self._poll()

    def _load(self, ui_state: dict[str, Any]) -> None:
        self._result = None
        self._error = None
        try:
            data = self._load_async(ui_state)
            self._result = {**ui_state, **data}
        except Exception as e:
            self._error = str(e)

    def _poll(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._parent.after(100, self._poll)
            return
        self._thread = None
        if self._win is None or not self._win.winfo_exists():
            return
        if self._loading_label is not None:
            self._loading_label.destroy()
            self._loading_label = None
        if self._error is not None:
            tk.Label(
                self._win,
                text=f"錯誤：{self._error}",
                font=FONT,
                fg="red",
                padx=16,
            ).pack()
            return
        assert self._result is not None
        self._render(self._result)

    def _render(self, data: dict[str, Any]) -> None:
        assert self._win is not None
        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=8, pady=8)
        for section in self._build_sections(data):
            section.render(container)
        tk.Button(container, text="重新載入", command=self._refresh).pack(pady=(8, 0))

    def _refresh(self) -> None:
        if self._win is None or not self._win.winfo_exists():
            return
        for w in self._win.winfo_children():
            w.destroy()
        self._start_load()

    # ── Hooks for subclasses ───────────────────────────────

    def _preflight(self) -> bool:
        return True

    def _snapshot_ui_state(self) -> dict[str, Any]:
        return {}

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        raise NotImplementedError


class _CheckpointViewer(_BaseViewer):
    """需要 checkpoint.pt 存在的 viewer 基底。"""

    def _preflight(self) -> bool:
        if not CHECKPOINT_PATH.exists():
            messagebox.showwarning(
                self._title,
                f"找不到 checkpoint 檔案：\n{CHECKPOINT_PATH}",
                parent=self._parent,
            )
            return False
        return True


class DataOverviewViewer(_BaseViewer):
    """資料概況：圖片數量、變更明細、資料分布檢定。

    分布檢定僅需 LabelStore，因此在沒有 checkpoint 時仍可開啟；
    checkpoint 依賴的 sections 會替換為提示訊息。
    """

    _title = "資料概況"

    def __init__(self, parent: tk.Tk, store: LabelStore) -> None:
        super().__init__(parent)
        self._store = store

    def _snapshot_ui_state(self) -> dict[str, Any]:
        # 在 UI 執行緒抓取 store 快照，避免和標註動作競爭。
        return {"orig_samples": snapshot_orig_samples(self._store)}

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        if not CHECKPOINT_PATH.exists():
            return {"checkpoint": None}
        ckpt = load_checkpoint(CHECKPOINT_PATH)
        return {
            "checkpoint": {
                "created_at": ckpt.get("created_at"),
                "counts": extract_counts(ckpt),
                "changes": extract_changes(ckpt),
            }
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        sections: list[Section] = []
        ckpt = data["checkpoint"]
        if ckpt is None:
            sections.append(
                NoticeSection(
                    f"找不到 checkpoint 檔案，僅顯示資料分布檢定：\n{CHECKPOINT_PATH}"
                )
            )
        else:
            sections.append(ImageCountsSection(ckpt["counts"], ckpt["created_at"]))
            sections.append(ChangesSection(ckpt["changes"]))
        sections.append(DistributionTestSection(data["orig_samples"]))
        return sections


class ModelInfoViewer(_CheckpointViewer):
    """模型資訊：模型架構、訓練超參數。"""

    _title = "模型資訊"

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        ckpt = load_checkpoint(CHECKPOINT_PATH)
        return {
            "file_size_mb": CHECKPOINT_PATH.stat().st_size / 1024 / 1024,
            "model": extract_model(ckpt),
            "hyperparams": extract_hyperparams(ckpt),
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        return [
            ModelArchSection(data["model"], data["file_size_mb"]),
            HyperparamsSection(data["hyperparams"]),
        ]


class ValF1Viewer(_CheckpointViewer):
    """分析結果：資料集分割、驗證集 F1。"""

    _title = "分析結果"

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        ckpt = load_checkpoint(CHECKPOINT_PATH)
        return {
            "model": extract_model(ckpt),
            "split_counts": extract_split_counts(),
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        return [
            SplitCountsSection(data["split_counts"]),
            ValF1Section(data["model"]),
        ]

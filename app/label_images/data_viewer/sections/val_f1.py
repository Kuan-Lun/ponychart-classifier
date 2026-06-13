"""驗證集 F1 section。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ...constants import CLASS_NAMES_LIST
from ..widgets import FONT, FONT_BOLD, grid_cell, section_frame, section_header


class ValF1Section:
    """驗證集 macro F1 與各角色 per-class F1。"""

    _NAME_W = 18
    _F1_W = 10

    def __init__(self, model: dict[str, Any]) -> None:
        self._m = model

    def render(self, parent: tk.Widget) -> None:
        section_header(parent, "驗證集 F1")
        per_class_f1: list[float] | None = self._m.get("per_class_f1")
        if per_class_f1 is None:
            tk.Label(
                parent,
                text="尚無 F1 資料，請按下方「重新載入」計算。",
                font=FONT,
                fg="#999",
                padx=16,
            ).pack(anchor="w")
            return

        frame = section_frame(parent)
        grid_cell(frame, "角色", 0, 0, font=FONT_BOLD, width=self._NAME_W, anchor="w")
        grid_cell(frame, "F1", 0, 1, font=FONT_BOLD, width=self._F1_W)
        tk.Frame(frame, height=1, bg="#999").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )

        for i, name in enumerate(CLASS_NAMES_LIST):
            grid_cell(frame, name, i + 2, 0, width=self._NAME_W, anchor="w")
            grid_cell(frame, f"{per_class_f1[i]:.4f}", i + 2, 1, width=self._F1_W)

        macro_row = len(CLASS_NAMES_LIST) + 2
        grid_cell(frame, "Macro", macro_row, 0, width=self._NAME_W, anchor="w")
        grid_cell(frame, str(self._m["val_f1"]), macro_row, 1, width=self._F1_W)

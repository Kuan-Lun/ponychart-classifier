"""資料集分割 section。"""

import tkinter as tk
from typing import Any

from ..widgets import FONT_BOLD, grid_cell, section_frame, section_header


class SplitCountsSection:
    """訓練／驗證集張數與比例。"""

    _COL_W = 8

    def __init__(self, split_counts: dict[str, Any]) -> None:
        self._sc = split_counts

    def render(self, parent: tk.Widget) -> None:
        if not self._sc:
            return
        section_header(parent, "資料集分割")
        frame = section_frame(parent)

        for col, h in enumerate(["", "比例", "張數"]):
            grid_cell(frame, h, 0, col, font=FONT_BOLD, width=self._COL_W)
        tk.Frame(frame, height=1, bg="#999").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=2
        )

        sc = self._sc
        rows = [
            ("訓練集", sc.get("train_ratio", 0), sc.get("train_size", 0)),
            ("驗證集", sc.get("val_ratio", 0), sc.get("val_size", 0)),
            ("合計", 1.0, sc.get("total_size", 0)),
        ]
        for r, (label, ratio, size) in enumerate(rows, start=2):
            grid_cell(frame, label, r, 0, font=FONT_BOLD, width=self._COL_W, anchor="w")
            grid_cell(frame, f"{ratio:.2f}", r, 1, width=self._COL_W)
            grid_cell(frame, f"{size:,}", r, 2, width=self._COL_W)

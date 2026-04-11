"""圖片數量 section。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import (
    FONT_BOLD,
    fmt_diff,
    grid_cell,
    kv_row,
    section_frame,
    section_header,
)


class ImageCountsSection:
    """圖片數量：原圖／裁切／合計 × 完整訓練／上次存檔／目前。"""

    _HEADERS = ["", "完整訓練", "上次存檔", "目前", "距上次", "距完整訓練"]
    _COL_W = 14

    def __init__(self, counts: dict[str, Any], created_at: Any) -> None:
        self._counts = counts
        self._created_at = created_at

    def render(self, parent: tk.Widget) -> None:
        section_header(parent, "圖片數量")
        info = section_frame(parent)
        kv_row(info, "最新圖片時間", str(self._created_at or "N/A"))

        frame = section_frame(parent)
        c = self._counts
        rows = [
            ("原圖", c["orig_full"], c["orig_last"], c["orig_cur"]),
            ("裁切", c["crop_full"], c["crop_last"], c["crop_cur"]),
            ("合計", c["total_full"], c["total_last"], c["total_cur"]),
        ]

        for col, h in enumerate(self._HEADERS):
            grid_cell(frame, h, 0, col, font=FONT_BOLD, width=self._COL_W)

        for r, (label, full, last, cur) in enumerate(rows, start=1):
            grid_cell(frame, label, r, 0, font=FONT_BOLD, width=self._COL_W, anchor="w")
            grid_cell(frame, f"{full:,}", r, 1, width=self._COL_W)
            grid_cell(
                frame,
                f"{last:,}" if last is not None else "-",
                r,
                2,
                width=self._COL_W,
            )
            grid_cell(frame, f"{cur:,}", r, 3, width=self._COL_W)
            since_last = fmt_diff(cur, last) if last is not None else ""
            since_full = fmt_diff(cur, full)
            grid_cell(frame, since_last, r, 4, width=self._COL_W)
            grid_cell(frame, since_full, r, 5, width=self._COL_W)

        grid_cell(
            frame,
            f"({c['unlabeled']} 未標註)",
            len(rows) + 1,
            3,
            width=self._COL_W,
            fg="#999",
            sticky="e",
        )

"""變更明細 section。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import (
    FONT_BOLD,
    fmt_detail_cell,
    grid_cell,
    section_frame,
    section_header,
)


class ChangesSection:
    """變更明細：新增／移除／重新標註 vs 上次存檔與完整訓練。"""

    _HEADERS = ["", "距完整訓練", "距上次存檔"]
    _LABELS = ["新增", "移除", "重新標註"]
    _COL_W = 24

    def __init__(self, changes: dict[str, Any]) -> None:
        self._changes = changes

    def render(self, parent: tk.Widget) -> None:
        section_header(parent, "變更明細")
        frame = section_frame(parent)

        for col, h in enumerate(self._HEADERS):
            grid_cell(frame, h, 0, col, font=FONT_BOLD, width=self._COL_W)

        full_detail = self._changes["full"]
        last_detail = self._changes["last"]
        for r, (label, fd, ld) in enumerate(
            zip(self._LABELS, full_detail, last_detail), start=1
        ):
            grid_cell(frame, label, r, 0, font=FONT_BOLD, width=self._COL_W, anchor="w")
            grid_cell(frame, fmt_detail_cell(*fd), r, 1, width=self._COL_W)
            grid_cell(frame, fmt_detail_cell(*ld), r, 2, width=self._COL_W)

"""提示訊息 section。"""

import tkinter as tk

from ..widgets import FONT


class NoticeSection:
    """單行提示訊息，常用於資料缺失的佔位。"""

    def __init__(self, message: str, *, fg: str = "#999") -> None:
        self._message = message
        self._fg = fg

    def render(self, parent: tk.Widget) -> None:
        tk.Label(
            parent,
            text=self._message,
            font=FONT,
            fg=self._fg,
            padx=16,
            pady=4,
        ).pack(anchor="w")

"""裁切模式的狀態管理與畫布事件處理。"""

import tkinter as tk
from collections.abc import Callable


class CropHandler:
    """裁切模式的狀態管理與畫布事件處理。"""

    def __init__(
        self,
        canvas: tk.Canvas,
        on_selection_complete: Callable[[], None] | None = None,
    ):
        self._canvas = canvas
        self._on_selection_complete = on_selection_complete
        self.active: bool = False
        self._start: tuple[int, int] | None = None
        self._end: tuple[int, int] | None = None
        self._rect_id: int | None = None

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)

    def enter(self) -> None:
        self.active = True
        self._start = None
        self._end = None
        self._clear_rect()

    def exit(self) -> None:
        self.active = False
        self._start = None
        self._end = None
        self._clear_rect()

    def get_selection(self) -> tuple[int, int, int, int] | None:
        """回傳選取區域 (x1, y1, x2, y2)，若無有效選取則回傳 None。"""
        if self._start is None or self._end is None:
            return None
        sx, sy = self._start
        ex, ey = self._end
        x1, x2 = sorted((sx, ex))
        y1, y2 = sorted((sy, ey))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return x1, y1, x2, y2

    def _clear_rect(self) -> None:
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_press(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active:
            return
        self._start = (e.x, e.y)
        self._end = None
        self._clear_rect()

    def _on_drag(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active or self._start is None:
            return
        self._clear_rect()
        self._rect_id = self._canvas.create_rectangle(
            self._start[0],
            self._start[1],
            e.x,
            e.y,
            outline="red",
            width=2,
            dash=(4, 4),
        )

    def _on_release(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active or self._start is None:
            return
        self._end = (e.x, e.y)
        if self._on_selection_complete:
            self._on_selection_complete()

"""訓練超參數 section。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import kv_row, section_frame, section_header


class HyperparamsSection:
    """訓練超參數。若 hyperparams 為空則不渲染 section。"""

    def __init__(self, hyperparams: dict[str, Any]) -> None:
        self._hp = hyperparams

    def render(self, parent: tk.Widget) -> None:
        if not self._hp:
            return
        section_header(parent, "訓練超參數")
        f = section_frame(parent)
        for label, val in self._hp.items():
            kv_row(f, label, str(val))

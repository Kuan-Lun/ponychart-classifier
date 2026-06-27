"""模型架構 section。"""

import tkinter as tk
from typing import Any

from ..widgets import kv_row, section_frame, section_header


class ModelArchSection:
    """模型架構：backbone、輸入尺寸、參數量等。"""

    def __init__(self, model: dict[str, Any], file_size_mb: float) -> None:
        self._m = model
        self._file_size_mb = file_size_mb

    def render(self, parent: tk.Widget) -> None:
        section_header(parent, "模型架構")
        f = section_frame(parent)
        kv_row(f, "檔案大小", f"{self._file_size_mb:.2f} MB")
        m = self._m
        kv_row(f, "Backbone", str(m["backbone"]))
        kv_row(f, "Input size", str(m["input_size"]))
        kv_row(f, "Classes", str(m["num_classes"]))
        kv_row(f, "Parameters", f"{m['n_params']:,}")
        kv_row(f, "State dict keys", f"{m['n_keys']:,}")
        kv_row(f, "Val size", str(m["val_size"]))

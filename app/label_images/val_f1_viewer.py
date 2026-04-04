"""分析結果視窗：顯示驗證集 macro F1 與各角色 per-class F1。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from .checkpoint_viewer import _FONT, _FONT_BOLD, _BaseViewer
from .constants import CLASS_NAMES_LIST


class ValF1Viewer(_BaseViewer):
    """分析結果視窗：Val F1 + 各角色 F1。"""

    _title = "分析結果"

    def _render(self, data: dict[str, Any]) -> None:
        assert self._win is not None
        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        m = data["model"]

        # --- Macro F1 ---
        self._section(container, "驗證集 F1")
        f = self._pack_section_frame(container)
        self._kv(f, "Val size", str(m["val_size"]))
        self._kv(f, "Macro F1", str(m["val_f1"]))

        # --- Per-class F1 table ---
        per_class_f1: list[float] | None = m.get("per_class_f1")
        if per_class_f1 is not None:
            self._section(container, "各角色 F1")
            table = tk.Frame(container)
            table.pack(anchor="w", padx=16, pady=(0, 4))

            # Header
            tk.Label(table, text="角色", font=_FONT_BOLD, width=18, anchor="w").grid(
                row=0, column=0, padx=2
            )
            tk.Label(table, text="F1", font=_FONT_BOLD, width=10, anchor="e").grid(
                row=0, column=1, padx=2
            )

            # Rows
            for i, name in enumerate(CLASS_NAMES_LIST):
                f1 = per_class_f1[i]
                tk.Label(table, text=name, font=_FONT, width=18, anchor="w").grid(
                    row=i + 1, column=0, padx=2
                )
                tk.Label(
                    table, text=f"{f1:.4f}", font=_FONT, width=10, anchor="e"
                ).grid(row=i + 1, column=1, padx=2)
        else:
            self._section(container, "各角色 F1")
            tk.Label(
                container,
                text="尚無 per-class F1 資料，請重新執行自動標註。",
                font=_FONT,
                fg="#999",
                padx=16,
            ).pack(anchor="w")

        # --- Refresh ---
        tk.Button(container, text="重新載入", command=self._refresh).pack(pady=(8, 0))

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

        # --- 資料集分割 ---
        sc = data.get("split_counts", {})
        if sc:
            self._section(container, "資料集分割")
            table = tk.Frame(container)
            table.pack(anchor="w", padx=16, pady=(0, 4))

            # Header
            for col, h in enumerate(["", "比例", "張數"]):
                tk.Label(table, text=h, font=_FONT_BOLD, width=8, anchor="e").grid(
                    row=0, column=col, padx=2
                )

            # Separator (header / data)
            sep = tk.Frame(table, height=1, bg="#999")
            sep.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)

            # Rows
            rows = [
                ("訓練集", sc.get("train_ratio", 0), sc.get("train_size", 0)),
                ("驗證集", sc.get("val_ratio", 0), sc.get("val_size", 0)),
                ("合計", 1.0, sc.get("total_size", 0)),
            ]
            for r, (label, ratio, size) in enumerate(rows, start=2):
                tk.Label(table, text=label, font=_FONT_BOLD, width=8, anchor="w").grid(
                    row=r, column=0, padx=2
                )
                tk.Label(
                    table, text=f"{ratio:.2f}", font=_FONT, width=8, anchor="e"
                ).grid(row=r, column=1, padx=2)
                tk.Label(table, text=f"{size:,}", font=_FONT, width=8, anchor="e").grid(
                    row=r, column=2, padx=2
                )

        # --- 驗證集 F1 ---
        self._section(container, "驗證集 F1")
        per_class_f1: list[float] | None = m.get("per_class_f1")
        if per_class_f1 is not None:
            f1_table = tk.Frame(container)
            f1_table.pack(anchor="w", padx=16, pady=(0, 4))

            # Header
            tk.Label(f1_table, text="角色", font=_FONT_BOLD, width=18, anchor="w").grid(
                row=0, column=0, padx=2
            )
            tk.Label(f1_table, text="F1", font=_FONT_BOLD, width=10, anchor="e").grid(
                row=0, column=1, padx=2
            )

            # Separator (header / data)
            sep = tk.Frame(f1_table, height=1, bg="#999")
            sep.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

            # Per-class rows
            for i, name in enumerate(CLASS_NAMES_LIST):
                f1 = per_class_f1[i]
                tk.Label(f1_table, text=name, font=_FONT, width=18, anchor="w").grid(
                    row=i + 2, column=0, padx=2
                )
                tk.Label(
                    f1_table, text=f"{f1:.4f}", font=_FONT, width=10, anchor="e"
                ).grid(row=i + 2, column=1, padx=2)

            # Macro F1 row (summary)
            macro_row = len(CLASS_NAMES_LIST) + 2
            tk.Label(f1_table, text="Macro", font=_FONT, width=18, anchor="w").grid(
                row=macro_row, column=0, padx=2
            )
            tk.Label(
                f1_table, text=str(m["val_f1"]), font=_FONT, width=10, anchor="e"
            ).grid(row=macro_row, column=1, padx=2)
        else:
            tk.Label(
                container,
                text="尚無 F1 資料，請重新執行自動標註。",
                font=_FONT,
                fg="#999",
                padx=16,
            ).pack(anchor="w")

        # --- Refresh ---
        tk.Button(container, text="重新載入", command=self._refresh).pack(pady=(8, 0))

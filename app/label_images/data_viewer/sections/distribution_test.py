"""資料分布檢定 section。"""

import tkinter as tk

from ponychart_classifier.stats import GoFTestResult, goodness_of_fit_test
from ponychart_classifier.training.constants import (
    GOF_RECENT_ORIG_LIMIT,
    LABEL_SIZE_PROBS,
)

from ..stats import combo_counts_flat, count_by_label_size, overall_counts
from ..widgets import FONT, FONT_BOLD, grid_cell, section_frame, section_header


class DistributionTestSection:
    """資料分布檢定：三層表頭呈現所有分布 × 所有方法。"""

    _GOF_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Asymptotic",
            [
                ("pearson_asymptotic", "Pearson"),
                ("lr_asymptotic", "LR"),
            ],
        ),
        (
            "Exact",
            [
                ("pearson_exact", "Pearson"),
                ("lr_exact", "LR"),
                ("probability_exact", "Probability"),
            ],
        ),
    ]

    _NAME_W = 10
    _N_W = 7
    _DIST_W = 8
    _STAT_W = 8
    _P_W = 9
    _INFO_COLS = 3  # 名稱, n, 分布

    def __init__(self, orig_samples: dict[str, list[int]]) -> None:
        self._orig = orig_samples
        self._all_methods = [m for _, methods in self._GOF_GROUPS for m in methods]
        self._total_cols = self._INFO_COLS + len(self._all_methods) * 2

    def render(self, parent: tk.Widget) -> None:
        section_header(
            parent,
            f"資料分布檢定（原圖，最近 {GOF_RECENT_ORIG_LIMIT:,} 張）",
        )
        if not self._orig:
            tk.Label(parent, text="尚無原圖標註資料", font=FONT, fg="#999").pack(
                anchor="w", padx=16, pady=(0, 4)
            )
            return
        tk.Label(
            parent,
            text=f"此次檢定使用 {len(self._orig):,} 張原圖",
            font=FONT,
            anchor="w",
        ).pack(anchor="w", padx=16, pady=(0, 4))
        tk.Label(
            parent,
            text=(
                "註：Probability exact 沒有單一統計量；"
                "N/A 代表真正的 exact，數值* 代表 fallback 到 Pearson asymptotic。"
            ),
            font=FONT,
            anchor="w",
            fg="#666",
        ).pack(anchor="w", padx=16, pady=(0, 4))

        frame = section_frame(parent)
        self._render_header(frame)

        uniform_rows, ratio_rows = self._prepare_rows()
        next_row = self._render_data_rows(frame, uniform_rows, 6)
        if ratio_rows:
            self._render_data_rows(frame, ratio_rows, next_row, probs=LABEL_SIZE_PROBS)

    def _prepare_rows(
        self,
    ) -> tuple[
        list[tuple[str, list[int]]],
        list[tuple[str, list[int]]],
    ]:
        by_n = count_by_label_size(self._orig)
        overall = overall_counts(self._orig)

        uniform_rows: list[tuple[str, list[int]]] = []
        if sum(by_n[1]) > 0:
            uniform_rows.append(("單標籤", by_n[1]))
        combo2 = combo_counts_flat(self._orig, 2)
        if sum(combo2) > 0:
            uniform_rows.append(("雙標籤組合", combo2))
        combo3 = combo_counts_flat(self._orig, 3)
        if sum(combo3) > 0:
            uniform_rows.append(("三標籤組合", combo3))
        uniform_rows.append(("整體出現次數", overall))

        ratio_rows: list[tuple[str, list[int]]] = []
        label_size_counts = [
            sum(1 for v in self._orig.values() if len(v) == n) for n in (1, 2, 3)
        ]
        if sum(label_size_counts) > 0:
            ratio_rows.append(("標籤數", label_size_counts))

        return uniform_rows, ratio_rows

    def _render_header(self, frame: tk.Widget) -> None:
        c0 = self._INFO_COLS

        # Row 0: group headers (Asymptotic / Exact)
        for ic in range(self._INFO_COLS):
            grid_cell(frame, "", 0, ic, width=self._NAME_W)
        col = c0
        for grp_label, methods in self._GOF_GROUPS:
            span = len(methods) * 2
            grid_cell(
                frame,
                grp_label,
                0,
                col,
                font=FONT_BOLD,
                anchor="center",
                width=self._STAT_W,
                columnspan=span,
            )
            tk.Frame(frame, height=1, bg="#999").grid(
                row=1, column=col, columnspan=span, sticky="ew", padx=4, pady=0
            )
            col += span

        # Row 2: method names
        for i, (_key, mname) in enumerate(self._all_methods):
            col = c0 + i * 2
            grid_cell(
                frame,
                mname,
                2,
                col,
                font=FONT_BOLD,
                anchor="center",
                width=self._STAT_W + self._P_W,
                columnspan=2,
            )
            tk.Frame(frame, height=1, bg="#999").grid(
                row=3, column=col, columnspan=2, sticky="ew", padx=4, pady=0
            )

        # Row 4: sub-headers
        grid_cell(frame, "名稱", 4, 0, font=FONT_BOLD, width=self._NAME_W, anchor="w")
        grid_cell(frame, "n", 4, 1, font=FONT_BOLD, width=self._N_W)
        grid_cell(frame, "分布", 4, 2, font=FONT_BOLD, width=self._DIST_W)
        for i in range(len(self._all_methods)):
            col = c0 + i * 2
            grid_cell(frame, "統計量", 4, col, font=FONT_BOLD, width=self._STAT_W)
            grid_cell(frame, "p-value", 4, col + 1, font=FONT_BOLD, width=self._P_W)

        # Row 5: separator
        tk.Frame(frame, height=1, bg="#ccc").grid(
            row=5, column=0, columnspan=self._total_cols, sticky="ew", pady=2
        )

    def _render_data_rows(
        self,
        frame: tk.Widget,
        rows: list[tuple[str, list[int]]],
        start_row: int,
        *,
        probs: list[float] | None = None,
    ) -> int:
        c0 = self._INFO_COLS
        dist_text = ":".join(str(round(p * 50)) for p in probs) if probs else "均勻"
        for r_idx, (row_label, counts) in enumerate(rows):
            row = start_row + r_idx
            n = sum(counts)
            grid_cell(
                frame,
                row_label,
                row,
                0,
                font=FONT_BOLD,
                width=self._NAME_W,
                anchor="w",
            )
            grid_cell(frame, str(n), row, 1, width=self._N_W)
            grid_cell(frame, dist_text, row, 2, width=self._DIST_W)
            for i, (method_key, _mname) in enumerate(self._all_methods):
                col = c0 + i * 2
                if n == 0:
                    grid_cell(frame, "—", row, col, width=self._STAT_W)
                    grid_cell(frame, "—", row, col + 1, width=self._P_W)
                    continue
                r = goodness_of_fit_test(
                    counts,
                    probs=probs,
                    method=method_key,  # type: ignore[arg-type]
                )
                grid_cell(
                    frame,
                    self._fmt_stat(method_key, r),
                    row,
                    col,
                    width=self._STAT_W,
                )
                grid_cell(frame, self._fmt_p(r), row, col + 1, width=self._P_W)
        return start_row + len(rows)

    @staticmethod
    def _fmt_stat(method_key: str, r: GoFTestResult) -> str:
        if method_key == "probability_exact":
            if r.exact:
                return "N/A"
            if r.statistic is not None:
                return f"{r.statistic:.2f}*"
        if r.statistic is None:
            return "—"
        return f"{r.statistic:.2f}"

    @staticmethod
    def _fmt_p(r: GoFTestResult) -> str:
        p = r.p_value
        if p < 0.001:
            s = "<.001"
        elif p > 0.999:
            s = ">.999"
        else:
            s = f"{p:.3f}"
        if p <= 0.01:
            return s + "**"
        if p <= 0.05:
            return s + "* "
        return s + "  "

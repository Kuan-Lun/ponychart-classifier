"""角色分布分析：以表格呈現標籤分布統計與均勻性檢定。"""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from itertools import combinations

from ponychart_classifier.stats import GoFTestResult, goodness_of_fit_test

from .constants import LABEL_MAP
from .file_ops import is_raw_image
from .label_store import LabelStore

_NUM_CLASSES = len(LABEL_MAP)
_SHORT_NAMES = ["TS", "RA", "FS", "RD", "PP", "AJ"]

_FONT = ("Consolas", 11)
_FONT_BOLD = ("Consolas", 11, "bold")
_FONT_HEADER = ("Consolas", 12, "bold")
_FONT_SECTION = ("Consolas", 13, "bold")


def _split_samples(
    store: LabelStore,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """將 store 中的標籤資料分為原圖和全部。"""
    from pathlib import Path

    all_samples: dict[str, list[int]] = {}
    orig_samples: dict[str, list[int]] = {}
    for key in store.all_keys():
        labels = store.get(key)
        if not labels:
            continue
        all_samples[key] = labels
        filename = Path(key).name
        if is_raw_image(Path(filename)):
            orig_samples[key] = labels
    return orig_samples, all_samples


def _pct(count: int, total: int) -> str:
    """格式化百分比（不帶 % 符號），total 為 0 時回傳 '—'。"""
    if total == 0:
        return "—"
    return f"{count / total * 100:.1f}"


def _count_by_label_size(
    samples: dict[str, list[int]],
) -> dict[int, list[int]]:
    """按標籤數量分組，回傳每個角色在該組中的出現次數。

    Returns:
        {n_labels: [count_per_class]}  n_labels = 1, 2, 3
    """
    result: dict[int, list[int]] = {}
    for n in (1, 2, 3):
        counts = [0] * _NUM_CLASSES
        group = [v for v in samples.values() if len(v) == n]
        for labels in group:
            for lbl in labels:
                counts[lbl - 1] += 1
        result[n] = counts
    return result


def _overall_counts(samples: dict[str, list[int]]) -> list[int]:
    """各角色整體出現次數。"""
    counts = [0] * _NUM_CLASSES
    for labels in samples.values():
        for lbl in labels:
            counts[lbl - 1] += 1
    return counts


def _combo_counts(
    samples: dict[str, list[int]], size: int
) -> list[tuple[tuple[int, ...], int]]:
    """計算指定大小的標籤組合出現次數，按次數降序排列。

    只回傳 count > 0 的組合。
    """
    combos = [tuple(sorted(v)) for v in samples.values() if len(v) == size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, _NUM_CLASSES + 1), size))
    result = [(c, cnt.get(c, 0)) for c in all_combos if cnt.get(c, 0) > 0]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _combo_counts_flat(samples: dict[str, list[int]], size: int) -> list[int]:
    """計算指定大小的標籤組合出現次數，回傳 flat list。"""
    combos = [tuple(sorted(v)) for v in samples.values() if len(v) == size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, _NUM_CLASSES + 1), size))
    return [cnt.get(c, 0) for c in all_combos]


class DistributionViewer:
    """角色分布分析視窗。"""

    def __init__(self, parent: tk.Tk, store: LabelStore) -> None:
        self._parent = parent
        self._store = store
        self._win: tk.Toplevel | None = None

    def show(self) -> None:
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            self._win.focus_force()
            return

        self._win = tk.Toplevel(self._parent)
        self._win.title("角色分布分析")
        self._win.resizable(False, False)

        orig, all_ = _split_samples(self._store)
        if not all_:
            tk.Label(
                self._win, text="尚無標註資料", font=_FONT, padx=40, pady=20
            ).pack()
            return

        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=12, pady=8)

        self._render_summary(container, orig, all_)
        self._render_char_table(container, orig, all_)
        self._render_double_matrix(container, orig, all_)
        self._render_gof_table(container, orig)

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _section(parent: tk.Widget, title: str) -> None:
        tk.Label(parent, text=title, font=_FONT_SECTION, anchor="w").pack(
            anchor="w", padx=4, pady=(10, 2)
        )
        tk.Frame(parent, height=1, bg="#ccc").pack(fill="x", padx=4, pady=(0, 4))

    @staticmethod
    def _make_cell(
        frame: tk.Widget,
        text: str,
        row: int,
        col: int,
        *,
        font: tuple[str, int] | tuple[str, int, str] = _FONT,
        anchor: str = "e",
        width: int = 10,
        bg: str = "",
        fg: str = "",
    ) -> None:
        kwargs: dict[str, object] = {
            "text": text,
            "font": font,
            "width": width,
            "anchor": anchor,
        }
        if bg:
            kwargs["bg"] = bg
        if fg:
            kwargs["fg"] = fg
        lbl = tk.Label(frame, **kwargs)  # type: ignore[arg-type]
        lbl.grid(row=row, column=col, padx=1, pady=1)

    # ── Sections ─────────────────────────────────────────────

    def _render_summary(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
    ) -> None:
        text = f"原圖: {len(orig)} 張  |  全部（含裁切）: {len(all_)} 張"
        tk.Label(parent, text=text, font=_FONT_HEADER, fg="#666").pack(
            anchor="w", padx=4, pady=(0, 4)
        )

    def _render_char_table(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
    ) -> None:
        """各角色統計表：單標籤分布 + 整體出現次數。"""
        self._section(parent, "各角色統計")
        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        orig_by_n = _count_by_label_size(orig)
        all_by_n = _count_by_label_size(all_)
        orig_overall = _overall_counts(orig)
        all_overall = _overall_counts(all_)

        # Header
        headers = ["原圖/全部"] + _SHORT_NAMES + ["合計"]
        for col, h in enumerate(headers):
            self._make_cell(frame, h, 0, col, font=_FONT_BOLD, width=12)

        row_defs: list[tuple[str, list[int], list[int]]] = [
            ("單標籤", orig_by_n[1], all_by_n[1]),
            ("雙標籤", orig_by_n[2], all_by_n[2]),
            ("三標籤", orig_by_n[3], all_by_n[3]),
            ("出現次數", orig_overall, all_overall),
        ]

        for r, (label, o_counts, a_counts) in enumerate(row_defs, start=1):
            self._make_cell(frame, label, r, 0, font=_FONT_BOLD, anchor="w", width=12)
            o_total = sum(o_counts)
            a_total = sum(a_counts)
            for c in range(_NUM_CLASSES):
                o_s = _pct(o_counts[c], o_total)
                a_s = _pct(a_counts[c], a_total)
                self._make_cell(frame, f"{o_s}/{a_s}", r, c + 1, width=12)
            self._make_cell(
                frame,
                f"{o_total}/{a_total}",
                r,
                _NUM_CLASSES + 1,
                width=12,
            )

    def _render_double_matrix(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
    ) -> None:
        """雙標籤組合：以上三角矩陣呈現，每格原圖/全部。"""
        self._section(parent, "雙標籤組合")

        orig_combos = dict(_combo_counts(orig, 2))
        all_combos = dict(_combo_counts(all_, 2))
        o_total = sum(orig_combos.values())
        a_total = sum(all_combos.values())

        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        # Column headers
        self._make_cell(frame, "原圖/全部", 0, 0, font=_FONT_BOLD, width=12, anchor="w")
        for c, name in enumerate(_SHORT_NAMES):
            self._make_cell(frame, name, 0, c + 1, font=_FONT_BOLD, width=12)

        for i in range(_NUM_CLASSES):
            self._make_cell(
                frame,
                _SHORT_NAMES[i],
                i + 1,
                0,
                font=_FONT_BOLD,
                anchor="w",
                width=6,
            )
            for j in range(_NUM_CLASSES):
                if j <= i:
                    self._make_cell(frame, "—", i + 1, j + 1, width=12)
                else:
                    combo = (i + 1, j + 1)
                    o_s = _pct(orig_combos.get(combo, 0), o_total)
                    a_s = _pct(all_combos.get(combo, 0), a_total)
                    self._make_cell(frame, f"{o_s}/{a_s}", i + 1, j + 1, width=12)

    # ── GoF test table ────────────────────────────────────────

    # ── GoF table: 3-level header ─────────────────────────────
    #
    # Row 0:  [blank]     Asymptotic          Exact
    # Row 1:  [blank]     Pearson    LR       Pearson    LR       Probability
    # Row 2:  分布        統計量 p   統計量 p  統計量 p   統計量 p  統計量 p
    # ------- separator -------
    # Row 4+: data

    # (group_label, colspan_in_methods, methods_in_group)
    # Each method: (api_key, display_name)
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

    @staticmethod
    def _fmt_stat(r: GoFTestResult) -> str:
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

    def _render_gof_table(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
    ) -> None:
        """均勻性檢定：一張表呈現所有分布 × 所有方法（三層表頭）。"""
        self._section(parent, "均勻性檢定（原圖）")

        by_n = _count_by_label_size(orig)
        overall = _overall_counts(orig)
        data_rows: list[tuple[str, list[int]]] = [("整體出現次數", overall)]
        if sum(by_n[1]) > 0:
            data_rows.append(("單標籤", by_n[1]))
        combo2 = _combo_counts_flat(orig, 2)
        if sum(combo2) > 0:
            data_rows.append(("雙標籤組合", combo2))
        combo3 = _combo_counts_flat(orig, 3)
        if sum(combo3) > 0:
            data_rows.append(("三標籤組合", combo3))

        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        label_w = 16
        stat_w = 8
        p_w = 9

        # Flatten methods for column indexing
        all_methods: list[tuple[str, str]] = []
        for _grp, methods in self._GOF_GROUPS:
            all_methods.extend(methods)
        n_methods = len(all_methods)
        total_cols = 1 + n_methods * 2

        # Row 0: group headers (Asymptotic / Exact)
        self._make_cell(frame, "", 0, 0, width=label_w)
        col = 1
        for grp_label, methods in self._GOF_GROUPS:
            span = len(methods) * 2
            lbl = tk.Label(
                frame,
                text=grp_label,
                font=_FONT_BOLD,
                anchor="center",
            )
            lbl.grid(row=0, column=col, columnspan=span, padx=1, pady=1)
            col += span

        # Row 1: method names (Pearson / LR / Probability)
        self._make_cell(frame, "", 1, 0, width=label_w)
        for i, (_key, mname) in enumerate(all_methods):
            col = 1 + i * 2
            lbl = tk.Label(
                frame,
                text=mname,
                font=_FONT_BOLD,
                width=stat_w + p_w,
                anchor="center",
            )
            lbl.grid(row=1, column=col, columnspan=2, padx=1, pady=1)

        # Row 2: sub-headers (統計量 / p-value)
        self._make_cell(
            frame,
            "分布",
            2,
            0,
            font=_FONT_BOLD,
            width=label_w,
            anchor="w",
        )
        for i in range(n_methods):
            col = 1 + i * 2
            self._make_cell(frame, "統計量", 2, col, font=_FONT_BOLD, width=stat_w)
            self._make_cell(frame, "p-value", 2, col + 1, font=_FONT_BOLD, width=p_w)

        # Row 3: separator
        tk.Frame(frame, height=1, bg="#ccc").grid(
            row=3,
            column=0,
            columnspan=total_cols,
            sticky="ew",
            pady=2,
        )

        # Row 4+: data
        for r_idx, (row_label, counts) in enumerate(data_rows):
            row = r_idx + 4
            n = sum(counts)
            self._make_cell(
                frame,
                f"{row_label} (n={n})",
                row,
                0,
                font=_FONT_BOLD,
                width=label_w,
                anchor="w",
            )
            for i, (method_key, _mname) in enumerate(all_methods):
                col = 1 + i * 2
                if n == 0:
                    self._make_cell(frame, "—", row, col, width=stat_w)
                    self._make_cell(frame, "—", row, col + 1, width=p_w)
                else:
                    r = goodness_of_fit_test(
                        counts,
                        method=method_key,  # type: ignore[arg-type]
                    )
                    self._make_cell(
                        frame,
                        self._fmt_stat(r),
                        row,
                        col,
                        width=stat_w,
                    )
                    self._make_cell(
                        frame,
                        self._fmt_p(r),
                        row,
                        col + 1,
                        width=p_w,
                    )

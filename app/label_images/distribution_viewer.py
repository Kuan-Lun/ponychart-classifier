"""角色分布：以表格呈現標籤分布統計與均勻性檢定。"""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from itertools import combinations
from typing import Protocol

from ponychart_classifier.stats import GoFTestResult, goodness_of_fit_test
from ponychart_classifier.training.constants import LABEL_SIZE_PROBS

from .constants import LABEL_MAP
from .file_ops import is_raw_image
from .label_store import LabelStore

_NUM_CLASSES = len(LABEL_MAP)
_SHORT_NAMES = ["TS", "RA", "FS", "RD", "PP", "AJ"]

_FONT = ("Consolas", 11)
_FONT_BOLD = ("Consolas", 11, "bold")
_FONT_HEADER = ("Consolas", 12, "bold")
_FONT_SECTION = ("Consolas", 13, "bold")


# ── Data helpers ────────────────────────────────────────────


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
    """按標籤數量分組，回傳每個角色在該組中的出現次數。"""
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


def _combo_counts_flat(samples: dict[str, list[int]], size: int) -> list[int]:
    """計算指定大小的標籤組合出現次數，回傳 flat list。"""
    combos = [tuple(sorted(v)) for v in samples.values() if len(v) == size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, _NUM_CLASSES + 1), size))
    return [cnt.get(c, 0) for c in all_combos]


# ── UI helpers ──────────────────────────────────────────────


def _section_header(parent: tk.Widget, title: str) -> None:
    tk.Label(parent, text=title, font=_FONT_SECTION, anchor="w").pack(
        anchor="w", padx=4, pady=(10, 2)
    )
    tk.Frame(parent, height=1, bg="#ccc").pack(fill="x", padx=4, pady=(0, 4))


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


# ── Section protocol ───────────────────────────────────────


class SectionRenderer(Protocol):
    def render(self, parent: tk.Widget) -> None: ...


# ── Sections ───────────────────────────────────────────────


class SummarySection:
    """總覽：原圖 / 全部張數。"""

    def __init__(self, orig: dict[str, list[int]], all_: dict[str, list[int]]) -> None:
        self._orig = orig
        self._all = all_

    def render(self, parent: tk.Widget) -> None:
        text = f"原圖: {len(self._orig)} 張  |  全部（含裁切）: {len(self._all)} 張"
        tk.Label(parent, text=text, font=_FONT_HEADER, fg="#666").pack(
            anchor="w", padx=4, pady=(0, 4)
        )


class CharTableSection:
    """各角色統計表：標籤分布百分比 + 整體出現次數。"""

    def __init__(self, orig: dict[str, list[int]], all_: dict[str, list[int]]) -> None:
        self._orig = orig
        self._all = all_

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "各角色統計")
        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        orig_by_n = _count_by_label_size(self._orig)
        all_by_n = _count_by_label_size(self._all)
        orig_overall = _overall_counts(self._orig)
        all_overall = _overall_counts(self._all)

        headers = ["原圖/全部"] + _SHORT_NAMES + ["合計"]
        for col, h in enumerate(headers):
            _make_cell(frame, h, 0, col, font=_FONT_BOLD, width=12)

        row_defs: list[tuple[str, list[int], list[int]]] = [
            ("單標籤", orig_by_n[1], all_by_n[1]),
            ("雙標籤", orig_by_n[2], all_by_n[2]),
            ("三標籤", orig_by_n[3], all_by_n[3]),
            ("出現次數", orig_overall, all_overall),
        ]

        for r, (label, o_counts, a_counts) in enumerate(row_defs, start=1):
            _make_cell(frame, label, r, 0, font=_FONT_BOLD, anchor="w", width=12)
            o_total = sum(o_counts)
            a_total = sum(a_counts)
            for c in range(_NUM_CLASSES):
                o_s = _pct(o_counts[c], o_total)
                a_s = _pct(a_counts[c], a_total)
                _make_cell(frame, f"{o_s}/{a_s}", r, c + 1, width=12)
            _make_cell(frame, f"{o_total}/{a_total}", r, _NUM_CLASSES + 1, width=12)


class GoFTableSection:
    """適合度檢定：三層表頭呈現所有分布 × 所有方法。"""

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

    _LABEL_W = 16
    _STAT_W = 8
    _P_W = 9

    def __init__(self, orig: dict[str, list[int]]) -> None:
        self._orig = orig
        self._all_methods = [m for _, methods in self._GOF_GROUPS for m in methods]
        self._total_cols = 1 + len(self._all_methods) * 2

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "適合度檢定（原圖）")

        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        self._render_header(frame)

        uniform_rows, ratio_rows = self._prepare_rows()

        next_row = self._render_data_rows(frame, uniform_rows, 6)

        if ratio_rows:
            tk.Frame(frame, height=1, bg="#ccc").grid(
                row=next_row,
                column=0,
                columnspan=self._total_cols,
                sticky="ew",
                pady=2,
            )
            self._render_data_rows(
                frame, ratio_rows, next_row + 1, probs=LABEL_SIZE_PROBS
            )

    # ── Private helpers ──

    def _prepare_rows(
        self,
    ) -> tuple[
        list[tuple[str, list[int]]],
        list[tuple[str, list[int]]],
    ]:
        by_n = _count_by_label_size(self._orig)
        overall = _overall_counts(self._orig)

        uniform_rows: list[tuple[str, list[int]]] = [("整體出現次數", overall)]
        if sum(by_n[1]) > 0:
            uniform_rows.append(("單標籤", by_n[1]))
        combo2 = _combo_counts_flat(self._orig, 2)
        if sum(combo2) > 0:
            uniform_rows.append(("雙標籤組合", combo2))
        combo3 = _combo_counts_flat(self._orig, 3)
        if sum(combo3) > 0:
            uniform_rows.append(("三標籤組合", combo3))

        ratio_rows: list[tuple[str, list[int]]] = []
        label_size_counts = [
            sum(1 for v in self._orig.values() if len(v) == n) for n in (1, 2, 3)
        ]
        if sum(label_size_counts) > 0:
            ratio_label = ":".join(str(round(p * 50)) for p in LABEL_SIZE_PROBS)
            ratio_rows.append((f"標籤數 {ratio_label}", label_size_counts))

        return uniform_rows, ratio_rows

    def _render_header(self, frame: tk.Widget) -> None:
        # Row 0: group headers (Asymptotic / Exact)
        _make_cell(frame, "", 0, 0, width=self._LABEL_W)
        col = 1
        for grp_label, methods in self._GOF_GROUPS:
            span = len(methods) * 2
            lbl = tk.Label(frame, text=grp_label, font=_FONT_BOLD, anchor="center")
            lbl.grid(row=0, column=col, columnspan=span, padx=1, pady=1)
            tk.Frame(frame, height=1, bg="#999").grid(
                row=1, column=col, columnspan=span, sticky="ew", padx=4, pady=0
            )
            col += span

        # Row 2: method names
        _make_cell(frame, "", 2, 0, width=self._LABEL_W)
        for i, (_key, mname) in enumerate(self._all_methods):
            col = 1 + i * 2
            lbl = tk.Label(
                frame,
                text=mname,
                font=_FONT_BOLD,
                width=self._STAT_W + self._P_W,
                anchor="center",
            )
            lbl.grid(row=2, column=col, columnspan=2, padx=1, pady=1)
            tk.Frame(frame, height=1, bg="#999").grid(
                row=3, column=col, columnspan=2, sticky="ew", padx=4, pady=0
            )

        # Row 4: sub-headers
        _make_cell(
            frame, "分布", 4, 0, font=_FONT_BOLD, width=self._LABEL_W, anchor="w"
        )
        for i in range(len(self._all_methods)):
            col = 1 + i * 2
            _make_cell(frame, "統計量", 4, col, font=_FONT_BOLD, width=self._STAT_W)
            _make_cell(frame, "p-value", 4, col + 1, font=_FONT_BOLD, width=self._P_W)

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
        for r_idx, (row_label, counts) in enumerate(rows):
            row = start_row + r_idx
            n = sum(counts)
            _make_cell(
                frame,
                f"{row_label} (n={n})",
                row,
                0,
                font=_FONT_BOLD,
                width=self._LABEL_W,
                anchor="w",
            )
            for i, (method_key, _mname) in enumerate(self._all_methods):
                col = 1 + i * 2
                if n == 0:
                    _make_cell(frame, "—", row, col, width=self._STAT_W)
                    _make_cell(frame, "—", row, col + 1, width=self._P_W)
                else:
                    r = goodness_of_fit_test(
                        counts,
                        probs=probs,
                        method=method_key,  # type: ignore[arg-type]
                    )
                    _make_cell(frame, self._fmt_stat(r), row, col, width=self._STAT_W)
                    _make_cell(frame, self._fmt_p(r), row, col + 1, width=self._P_W)
        return start_row + len(rows)

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


# ── Viewer ─────────────────────────────────────────────────


class DistributionViewer:
    """角色分布視窗。"""

    def __init__(self, parent: tk.Tk, store: LabelStore) -> None:
        self._parent = parent
        self._store = store
        self._win: tk.Toplevel | None = None

    def _build_sections(
        self,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
    ) -> list[SectionRenderer]:
        return [
            SummarySection(orig, all_),
            CharTableSection(orig, all_),
            GoFTableSection(orig),
        ]

    def show(self) -> None:
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            self._win.focus_force()
            return

        self._win = tk.Toplevel(self._parent)
        self._win.title("角色分布")
        self._win.resizable(False, False)

        orig, all_ = _split_samples(self._store)
        if not all_:
            tk.Label(
                self._win, text="尚無標註資料", font=_FONT, padx=40, pady=20
            ).pack()
            return

        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=12, pady=8)

        for section in self._build_sections(orig, all_):
            section.render(container)

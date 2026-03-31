"""角色分布分析：以表格呈現標籤分布統計。"""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from itertools import combinations

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
    """格式化百分比，total 為 0 時回傳 '—'。"""
    if total == 0:
        return "—"
    return f"{count / total * 100:.1f}%"


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


def _cooccurrence_matrix(
    samples: dict[str, list[int]],
) -> list[list[int]]:
    """6x6 共現矩陣。"""
    matrix = [[0] * _NUM_CLASSES for _ in range(_NUM_CLASSES)]
    for labels in samples.values():
        for a in labels:
            for b in labels:
                matrix[a - 1][b - 1] += 1
    return matrix


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
        self._render_combo_table(container, orig, all_, size=2, title="雙標籤組合")
        self._render_combo_table(container, orig, all_, size=3, title="三標籤組合")
        self._render_cooccurrence(container, orig, all_)

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
        headers = [""] + _SHORT_NAMES + ["合計"]
        for col, h in enumerate(headers):
            self._make_cell(frame, h, 0, col, font=_FONT_BOLD, width=10)

        row_defs: list[tuple[str, list[int]]] = [
            ("單標籤 (原圖)", orig_by_n[1]),
            ("雙標籤 (原圖)", orig_by_n[2]),
            ("三標籤 (原圖)", orig_by_n[3]),
            ("單標籤 (全部)", all_by_n[1]),
            ("雙標籤 (全部)", all_by_n[2]),
            ("三標籤 (全部)", all_by_n[3]),
            ("出現次數 (原圖)", orig_overall),
            ("出現次數 (全部)", all_overall),
        ]

        for r, (label, counts) in enumerate(row_defs, start=1):
            self._make_cell(frame, label, r, 0, font=_FONT_BOLD, anchor="w", width=16)
            total = sum(counts)
            for c, val in enumerate(counts):
                self._make_cell(frame, _pct(val, total), r, c + 1)
            self._make_cell(frame, str(total), r, _NUM_CLASSES + 1)

    def _render_combo_table(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
        *,
        size: int,
        title: str,
    ) -> None:
        """雙/三標籤組合表。"""
        orig_combos = _combo_counts(orig, size)
        all_combos = _combo_counts(all_, size)
        # Union of non-zero combos, sorted by all_ count descending
        combo_keys: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for c, _ in all_combos:
            if c not in seen:
                combo_keys.append(c)
                seen.add(c)
        for c, _ in orig_combos:
            if c not in seen:
                combo_keys.append(c)
                seen.add(c)

        if not combo_keys:
            return

        self._section(parent, title)
        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=8, pady=(0, 4))

        orig_map = dict(orig_combos)
        all_map = dict(all_combos)

        # Header
        combo_labels = ["+".join(_SHORT_NAMES[i - 1] for i in c) for c in combo_keys]
        headers = [""] + combo_labels + ["合計"]
        for col, h in enumerate(headers):
            w = max(8, len(h) + 2)
            self._make_cell(frame, h, 0, col, font=_FONT_BOLD, width=w)

        rows: list[tuple[str, dict[tuple[int, ...], int]]] = [
            ("原圖", orig_map),
            ("全部", all_map),
        ]
        for r, (label, cmap) in enumerate(rows, start=1):
            self._make_cell(frame, label, r, 0, font=_FONT_BOLD, anchor="w", width=8)
            total = sum(cmap.values())
            for c_idx, combo in enumerate(combo_keys):
                val = cmap.get(combo, 0)
                self._make_cell(
                    frame,
                    _pct(val, total),
                    r,
                    c_idx + 1,
                    width=max(8, len(combo_labels[c_idx]) + 2),
                )
            self._make_cell(
                frame,
                str(total),
                r,
                len(combo_keys) + 1,
                width=max(8, len("合計") + 2),
            )

    def _render_cooccurrence(
        self,
        parent: tk.Widget,
        orig: dict[str, list[int]],
        all_: dict[str, list[int]],
    ) -> None:
        """共現矩陣：原圖 / 全部並列。"""
        self._section(parent, "共現矩陣")

        outer = tk.Frame(parent)
        outer.pack(anchor="w", padx=8, pady=(0, 4))

        orig_matrix = _cooccurrence_matrix(orig)
        all_matrix = _cooccurrence_matrix(all_)

        for side, (label, matrix) in enumerate(
            [("原圖", orig_matrix), ("全部", all_matrix)]
        ):
            if side > 0:
                # spacer
                tk.Frame(outer, width=24).grid(row=0, column=1)

            sub = tk.Frame(outer)
            sub.grid(row=0, column=side * 2, sticky="n")

            tk.Label(sub, text=label, font=_FONT_BOLD).grid(
                row=0, column=0, columnspan=_NUM_CLASSES + 1
            )

            # Column headers
            self._make_cell(sub, "", 1, 0, width=6)
            for c, name in enumerate(_SHORT_NAMES):
                self._make_cell(sub, name, 1, c + 1, font=_FONT_BOLD, width=6)

            # Conditional probability P(col|row)
            cond: list[list[float]] = [
                [0.0] * _NUM_CLASSES for _ in range(_NUM_CLASSES)
            ]
            for i in range(_NUM_CLASSES):
                diag = matrix[i][i]
                for j in range(_NUM_CLASSES):
                    if i != j and diag > 0:
                        cond[i][j] = matrix[i][j] / diag * 100

            flat_cond = [
                cond[i][j]
                for i in range(_NUM_CLASSES)
                for j in range(_NUM_CLASSES)
                if i != j
            ]
            cond_max = max(flat_cond) if flat_cond else 1.0

            for i in range(_NUM_CLASSES):
                self._make_cell(
                    sub,
                    _SHORT_NAMES[i],
                    i + 2,
                    0,
                    font=_FONT_BOLD,
                    anchor="w",
                    width=6,
                )
                for j in range(_NUM_CLASSES):
                    if i == j:
                        self._make_cell(sub, "—", i + 2, j + 1, width=6)
                    else:
                        rate = cond[i][j]
                        if cond_max > 0 and rate > 0:
                            intensity = int(200 * rate / cond_max)
                            r_ = 200 - intensity
                            g_ = 200 - intensity // 2
                            bg = f"#{r_:02x}{g_:02x}{200:02x}"
                        else:
                            bg = ""
                        self._make_cell(
                            sub, f"{rate:.0f}%", i + 2, j + 1, width=6, bg=bg
                        )

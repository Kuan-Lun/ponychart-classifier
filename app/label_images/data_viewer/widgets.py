"""Low-level Tk layout helpers shared by data_viewer sections."""

from __future__ import annotations

import tkinter as tk

FONT = ("Consolas", 11)
FONT_BOLD = ("Consolas", 11, "bold")
FONT_HEADER = ("Consolas", 12, "bold")


def section_header(parent: tk.Widget, title: str) -> None:
    tk.Label(parent, text=title, font=FONT_HEADER, anchor="w").pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    tk.Frame(parent, height=1, bg="#ccc").pack(fill="x", padx=8, pady=(0, 4))


def section_frame(parent: tk.Widget) -> tk.Frame:
    frame = tk.Frame(parent)
    frame.pack(anchor="w", padx=16, pady=(0, 4))
    return frame


def kv_row(parent: tk.Widget, key: str, value: str) -> None:
    row = tk.Frame(parent)
    row.pack(anchor="w")
    tk.Label(row, text=f"{key}:", font=FONT_BOLD, width=18, anchor="w").pack(
        side="left"
    )
    tk.Label(row, text=value, font=FONT, anchor="w").pack(side="left")


def grid_cell(
    frame: tk.Widget,
    text: str,
    row: int,
    col: int,
    *,
    font: tuple[str, int] | tuple[str, int, str] = FONT,
    anchor: str = "e",
    width: int = 10,
    fg: str = "",
    columnspan: int = 1,
    sticky: str = "",
) -> None:
    kwargs: dict[str, object] = {
        "text": text,
        "font": font,
        "width": width,
        "anchor": anchor,
    }
    if fg:
        kwargs["fg"] = fg
    lbl = tk.Label(frame, **kwargs)  # type: ignore[arg-type]
    grid_kwargs: dict[str, object] = {
        "row": row,
        "column": col,
        "padx": 2,
        "pady": 1,
        "columnspan": columnspan,
    }
    if sticky:
        grid_kwargs["sticky"] = sticky
    lbl.grid(**grid_kwargs)  # type: ignore[arg-type]


def fmt_diff(cur: int, base: int) -> str:
    diff = cur - base
    ratio = diff / base if base else 0
    return f"{diff:+,d} ({ratio:+.1%})"


def fmt_detail_cell(total: int, n_o: int, n_c: int) -> str:
    return f"{total} ({n_o} 原圖, {n_c} 裁切)"

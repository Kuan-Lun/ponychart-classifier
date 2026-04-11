"""資料／模型資訊的非模態視窗。

架構說明：
- Section 為獨立的渲染單位，互不相依，可自由組合。
- `_BaseViewer` 負責視窗生命週期、背景載入與輪詢，呼叫子類提供的
  `_load_async` / `_build_sections` hook。
- `_CheckpointViewer` 追加「checkpoint.pt 必須存在」的前置檢查。

新增一個 section 時：實作 `Section.render`，再於對應的 viewer 的
`_build_sections` 中加入即可，不需修改任何現有類別。
"""

from __future__ import annotations

import threading
import tkinter as tk
from collections import Counter
from itertools import combinations
from pathlib import Path
from tkinter import messagebox
from typing import Any, Protocol

from ponychart_classifier.stats import GoFTestResult, goodness_of_fit_test
from ponychart_classifier.training.constants import LABEL_SIZE_PROBS

from .constants import CLASS_NAMES_LIST, IMAGE_DIR, LABEL_MAP
from .file_ops import is_raw_image
from .label_store import LabelStore

_CHECKPOINT_PATH = IMAGE_DIR / "checkpoint.pt"

_FONT = ("Consolas", 11)
_FONT_BOLD = ("Consolas", 11, "bold")
_FONT_HEADER = ("Consolas", 12, "bold")

_NUM_CLASSES = len(LABEL_MAP)


# ── Checkpoint loading ──────────────────────────────────────


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """背景執行緒呼叫：載入 checkpoint 原始 dict。"""
    import torch

    result: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return result


def _extract_counts(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取圖片數量統計。

    三欄資料皆來自 labels.json 快照（完整訓練／上次存檔／當前），
    避免掃描磁碟誤把未標註圖片計入。未標註數另外從 rawimage 根目錄
    與 rawimage/unlabeled 兩個位置取得。
    """
    from ponychart_classifier.training.constants import RAWIMAGE_DIR
    from ponychart_classifier.training.sampling import is_original, load_samples

    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    samples = load_samples()
    labels_current = {
        str(Path(p).relative_to(RAWIMAGE_DIR)): labels for p, labels in samples
    }

    def count_orig_crop(labels: dict[str, list[int]]) -> tuple[int, int]:
        n_o = sum(1 for k in labels if is_original(k.split("/")[-1]))
        return n_o, len(labels) - n_o

    n_orig_full, n_crop_full = count_orig_crop(labels_full)
    n_orig_last, n_crop_last = count_orig_crop(labels_last)
    n_cur_orig, n_cur_crop = count_orig_crop(labels_current)

    def _count_unlabeled(dir_: Path) -> int:
        if not dir_.is_dir():
            return 0
        return sum(
            1
            for f in dir_.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg")
        )

    n_unlabeled = _count_unlabeled(RAWIMAGE_DIR) + _count_unlabeled(
        RAWIMAGE_DIR / "unlabeled"
    )

    return {
        "orig_full": n_orig_full,
        "crop_full": n_crop_full,
        "orig_last": n_orig_last,
        "crop_last": n_crop_last,
        "orig_cur": n_cur_orig,
        "crop_cur": n_cur_crop,
        "total_full": len(labels_full),
        "total_last": len(labels_last),
        "total_cur": len(labels_current),
        "unlabeled": n_unlabeled,
    }


def _extract_changes(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取變更明細。"""
    from ponychart_classifier.training.constants import RAWIMAGE_DIR
    from ponychart_classifier.training.sampling import is_original, load_samples

    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    samples = load_samples()
    labels_current = {
        str(Path(p).relative_to(RAWIMAGE_DIR)): labels for p, labels in samples
    }

    def _filename(key: str) -> str:
        return key.split("/")[-1]

    def diff_labels(
        baseline: dict[str, list[int]],
        current: dict[str, list[int]],
    ) -> tuple[set[str], set[str], set[str]]:
        base_keys = set(baseline)
        cur_keys = set(current)
        added = cur_keys - base_keys
        removed = base_keys - cur_keys

        # 檔案重新標注後可能被搬到不同子資料夾，key 會改變。
        # 用檔名配對 added/removed 來偵測這類「搬移」。
        added_by_name = {_filename(k): k for k in added}
        moved_names = {_filename(k) for k in removed if _filename(k) in added_by_name}
        moved_base = {k for k in removed if _filename(k) in moved_names}
        moved_cur = {added_by_name[n] for n in moved_names}

        # 搬移的檔案：若 label 有變動算 relabeled，否則忽略
        relabeled = {k for k in base_keys & cur_keys if baseline[k] != current[k]}
        for bk in moved_base:
            ck = added_by_name[_filename(bk)]
            if baseline[bk] != current[ck]:
                relabeled.add(ck)

        added -= moved_cur
        removed -= moved_base
        return added, removed, relabeled

    def split_orig_crop(keys: set[str]) -> tuple[int, int]:
        n_o = sum(1 for k in keys if is_original(k.split("/")[-1]))
        return n_o, len(keys) - n_o

    def diff_detail(
        baseline: dict[str, list[int]],
    ) -> list[tuple[int, int, int]]:
        added, removed, relabeled = diff_labels(baseline, labels_current)
        return [(len(s), *split_orig_crop(s)) for s in (added, removed, relabeled)]

    return {
        "full": diff_detail(labels_full),
        "last": diff_detail(labels_last),
    }


def _extract_model(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取模型架構資訊。"""
    import torch

    from ponychart_classifier.training.model import (
        BACKBONE_REGISTRY,
        build_model,
    )

    state_dict: dict[str, Any] = ckpt.get("state_dict", {})
    n_params = sum(
        p.numel()
        for p in (
            torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for v in state_dict.values()
        )
    )

    backbone_name = ckpt.get("backbone")
    if not backbone_name:
        for name in BACKBONE_REGISTRY:
            model = build_model(backbone=name, pretrained=False)
            try:
                model.load_state_dict(state_dict)
                backbone_name = name
                break
            except RuntimeError:
                continue
        else:
            backbone_name = "unknown"

    val_f1 = ckpt.get("val_f1")

    return {
        "backbone": backbone_name,
        "input_size": ckpt.get("input_size", "N/A"),
        "pre_resize": ckpt.get("pre_resize", "N/A"),
        "num_classes": ckpt.get("num_classes", "N/A"),
        "n_params": n_params,
        "n_keys": len(state_dict),
        "val_size": ckpt.get("val_size", "N/A"),
        "val_f1": f"{val_f1:.4f}" if val_f1 is not None else "N/A",
        "per_class_f1": ckpt.get("per_class_f1"),
    }


def _extract_hyperparams(ckpt: dict[str, Any]) -> dict[str, Any]:
    """從 checkpoint 萃取訓練超參數。"""
    hp_keys = [
        ("seed", "Seed"),
        ("batch_size", "Batch size"),
        ("lr_head", "LR head"),
        ("lr_features", "LR features"),
        ("lr_classifier", "LR classifier"),
        ("weight_decay", "Weight decay"),
        ("label_smoothing", "Label smoothing"),
    ]
    return {label: ckpt[key] for key, label in hp_keys if ckpt.get(key) is not None}


def _extract_split_counts() -> dict[str, Any]:
    """計算目前資料的訓練／驗證集張數與比例。"""
    import os

    from ponychart_classifier.training.constants import VAL_SIZE
    from ponychart_classifier.training.sampling import is_original, load_samples
    from ponychart_classifier.training.splitting import group_hash_split

    samples = load_samples()
    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)

    n_train = sum(1 for i in train_idx if is_original(os.path.basename(samples[i][0])))
    n_val = sum(1 for i in val_idx if is_original(os.path.basename(samples[i][0])))
    n_total = n_train + n_val
    return {
        "train_size": n_train,
        "val_size": n_val,
        "total_size": n_total,
        "train_ratio": n_train / n_total if n_total else 0.0,
        "val_ratio": n_val / n_total if n_total else 0.0,
    }


# ── LabelStore helpers ──────────────────────────────────────


def _snapshot_orig_samples(store: LabelStore) -> dict[str, list[int]]:
    """UI 執行緒呼叫：擷取 store 中「原圖」的標籤快照。"""
    orig: dict[str, list[int]] = {}
    for key in store.all_keys():
        labels = store.get(key)
        if not labels:
            continue
        if is_raw_image(Path(Path(key).name)):
            orig[key] = labels
    return orig


def _count_by_label_size(samples: dict[str, list[int]]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for n in (1, 2, 3):
        counts = [0] * _NUM_CLASSES
        for labels in samples.values():
            if len(labels) == n:
                for lbl in labels:
                    counts[lbl - 1] += 1
        result[n] = counts
    return result


def _overall_counts(samples: dict[str, list[int]]) -> list[int]:
    counts = [0] * _NUM_CLASSES
    for labels in samples.values():
        for lbl in labels:
            counts[lbl - 1] += 1
    return counts


def _combo_counts_flat(samples: dict[str, list[int]], size: int) -> list[int]:
    combos = [tuple(sorted(v)) for v in samples.values() if len(v) == size]
    cnt = Counter(combos)
    all_combos = list(combinations(range(1, _NUM_CLASSES + 1), size))
    return [cnt.get(c, 0) for c in all_combos]


# ── UI helpers ──────────────────────────────────────────────


def _section_header(parent: tk.Widget, title: str) -> None:
    tk.Label(parent, text=title, font=_FONT_HEADER, anchor="w").pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    tk.Frame(parent, height=1, bg="#ccc").pack(fill="x", padx=8, pady=(0, 4))


def _section_frame(parent: tk.Widget) -> tk.Frame:
    frame = tk.Frame(parent)
    frame.pack(anchor="w", padx=16, pady=(0, 4))
    return frame


def _kv_row(parent: tk.Widget, key: str, value: str) -> None:
    row = tk.Frame(parent)
    row.pack(anchor="w")
    tk.Label(row, text=f"{key}:", font=_FONT_BOLD, width=18, anchor="w").pack(
        side="left"
    )
    tk.Label(row, text=value, font=_FONT, anchor="w").pack(side="left")


def _grid_cell(
    frame: tk.Widget,
    text: str,
    row: int,
    col: int,
    *,
    font: tuple[str, int] | tuple[str, int, str] = _FONT,
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


def _fmt_diff(cur: int, base: int) -> str:
    diff = cur - base
    ratio = diff / base if base else 0
    return f"{diff:+,d} ({ratio:+.1%})"


def _fmt_detail_cell(total: int, n_o: int, n_c: int) -> str:
    return f"{total} ({n_o} 原圖, {n_c} 裁切)"


# ── Section protocol ────────────────────────────────────────


class Section(Protocol):
    """視窗中的一個獨立渲染單位。"""

    def render(self, parent: tk.Widget) -> None: ...


# ── Section: 圖片數量 ───────────────────────────────────────


class ImageCountsSection:
    """圖片數量：原圖／裁切／合計 × 完整訓練／上次存檔／目前。"""

    _HEADERS = ["", "完整訓練", "上次存檔", "目前", "距上次", "距完整訓練"]
    _COL_W = 14

    def __init__(self, counts: dict[str, Any], created_at: Any) -> None:
        self._counts = counts
        self._created_at = created_at

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "圖片數量")
        info = _section_frame(parent)
        _kv_row(info, "最新圖片時間", str(self._created_at or "N/A"))

        frame = _section_frame(parent)
        c = self._counts
        rows = [
            ("原圖", c["orig_full"], c["orig_last"], c["orig_cur"]),
            ("裁切", c["crop_full"], c["crop_last"], c["crop_cur"]),
            ("合計", c["total_full"], c["total_last"], c["total_cur"]),
        ]

        for col, h in enumerate(self._HEADERS):
            _grid_cell(frame, h, 0, col, font=_FONT_BOLD, width=self._COL_W)

        for r, (label, full, last, cur) in enumerate(rows, start=1):
            _grid_cell(
                frame, label, r, 0, font=_FONT_BOLD, width=self._COL_W, anchor="w"
            )
            _grid_cell(frame, f"{full:,}", r, 1, width=self._COL_W)
            _grid_cell(
                frame,
                f"{last:,}" if last is not None else "-",
                r,
                2,
                width=self._COL_W,
            )
            _grid_cell(frame, f"{cur:,}", r, 3, width=self._COL_W)
            since_last = _fmt_diff(cur, last) if last is not None else ""
            since_full = _fmt_diff(cur, full)
            _grid_cell(frame, since_last, r, 4, width=self._COL_W)
            _grid_cell(frame, since_full, r, 5, width=self._COL_W)

        _grid_cell(
            frame,
            f"({c['unlabeled']} 未標註)",
            len(rows) + 1,
            3,
            width=self._COL_W,
            fg="#999",
            sticky="e",
        )


# ── Section: 變更明細 ───────────────────────────────────────


class ChangesSection:
    """變更明細：新增／移除／重新標註 vs 上次存檔與完整訓練。"""

    _HEADERS = ["", "距完整訓練", "距上次存檔"]
    _LABELS = ["新增", "移除", "重新標註"]
    _COL_W = 24

    def __init__(self, changes: dict[str, Any]) -> None:
        self._changes = changes

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "變更明細")
        frame = _section_frame(parent)

        for col, h in enumerate(self._HEADERS):
            _grid_cell(frame, h, 0, col, font=_FONT_BOLD, width=self._COL_W)

        full_detail = self._changes["full"]
        last_detail = self._changes["last"]
        for r, (label, fd, ld) in enumerate(
            zip(self._LABELS, full_detail, last_detail), start=1
        ):
            _grid_cell(
                frame, label, r, 0, font=_FONT_BOLD, width=self._COL_W, anchor="w"
            )
            _grid_cell(frame, _fmt_detail_cell(*fd), r, 1, width=self._COL_W)
            _grid_cell(frame, _fmt_detail_cell(*ld), r, 2, width=self._COL_W)


# ── Section: 資料分布檢定 ───────────────────────────────────


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
        _section_header(parent, "資料分布檢定（原圖）")
        if not self._orig:
            tk.Label(parent, text="尚無原圖標註資料", font=_FONT, fg="#999").pack(
                anchor="w", padx=16, pady=(0, 4)
            )
            return

        frame = _section_frame(parent)
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
        by_n = _count_by_label_size(self._orig)
        overall = _overall_counts(self._orig)

        uniform_rows: list[tuple[str, list[int]]] = []
        if sum(by_n[1]) > 0:
            uniform_rows.append(("單標籤", by_n[1]))
        combo2 = _combo_counts_flat(self._orig, 2)
        if sum(combo2) > 0:
            uniform_rows.append(("雙標籤組合", combo2))
        combo3 = _combo_counts_flat(self._orig, 3)
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
            _grid_cell(frame, "", 0, ic, width=self._NAME_W)
        col = c0
        for grp_label, methods in self._GOF_GROUPS:
            span = len(methods) * 2
            _grid_cell(
                frame,
                grp_label,
                0,
                col,
                font=_FONT_BOLD,
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
            _grid_cell(
                frame,
                mname,
                2,
                col,
                font=_FONT_BOLD,
                anchor="center",
                width=self._STAT_W + self._P_W,
                columnspan=2,
            )
            tk.Frame(frame, height=1, bg="#999").grid(
                row=3, column=col, columnspan=2, sticky="ew", padx=4, pady=0
            )

        # Row 4: sub-headers
        _grid_cell(frame, "名稱", 4, 0, font=_FONT_BOLD, width=self._NAME_W, anchor="w")
        _grid_cell(frame, "n", 4, 1, font=_FONT_BOLD, width=self._N_W)
        _grid_cell(frame, "分布", 4, 2, font=_FONT_BOLD, width=self._DIST_W)
        for i in range(len(self._all_methods)):
            col = c0 + i * 2
            _grid_cell(frame, "統計量", 4, col, font=_FONT_BOLD, width=self._STAT_W)
            _grid_cell(frame, "p-value", 4, col + 1, font=_FONT_BOLD, width=self._P_W)

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
            _grid_cell(
                frame,
                row_label,
                row,
                0,
                font=_FONT_BOLD,
                width=self._NAME_W,
                anchor="w",
            )
            _grid_cell(frame, str(n), row, 1, width=self._N_W)
            _grid_cell(frame, dist_text, row, 2, width=self._DIST_W)
            for i, (method_key, _mname) in enumerate(self._all_methods):
                col = c0 + i * 2
                if n == 0:
                    _grid_cell(frame, "—", row, col, width=self._STAT_W)
                    _grid_cell(frame, "—", row, col + 1, width=self._P_W)
                    continue
                r = goodness_of_fit_test(
                    counts,
                    probs=probs,
                    method=method_key,  # type: ignore[arg-type]
                )
                _grid_cell(frame, self._fmt_stat(r), row, col, width=self._STAT_W)
                _grid_cell(frame, self._fmt_p(r), row, col + 1, width=self._P_W)
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


# ── Section: 模型架構 ───────────────────────────────────────


class ModelArchSection:
    """模型架構：backbone、輸入尺寸、參數量等。"""

    def __init__(self, model: dict[str, Any], file_size_mb: float) -> None:
        self._m = model
        self._file_size_mb = file_size_mb

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "模型架構")
        f = _section_frame(parent)
        _kv_row(f, "檔案大小", f"{self._file_size_mb:.2f} MB")
        m = self._m
        _kv_row(f, "Backbone", str(m["backbone"]))
        _kv_row(f, "Input size", str(m["input_size"]))
        _kv_row(f, "Pre-resize", str(m["pre_resize"]))
        _kv_row(f, "Classes", str(m["num_classes"]))
        _kv_row(f, "Parameters", f"{m['n_params']:,}")
        _kv_row(f, "State dict keys", f"{m['n_keys']:,}")
        _kv_row(f, "Val size", str(m["val_size"]))


# ── Section: 訓練超參數 ─────────────────────────────────────


class HyperparamsSection:
    """訓練超參數。若 hyperparams 為空則不渲染 section。"""

    def __init__(self, hyperparams: dict[str, Any]) -> None:
        self._hp = hyperparams

    def render(self, parent: tk.Widget) -> None:
        if not self._hp:
            return
        _section_header(parent, "訓練超參數")
        f = _section_frame(parent)
        for label, val in self._hp.items():
            _kv_row(f, label, str(val))


# ── Section: 提示訊息 ───────────────────────────────────────


class NoticeSection:
    """單行提示訊息，常用於資料缺失的佔位。"""

    def __init__(self, message: str, *, fg: str = "#999") -> None:
        self._message = message
        self._fg = fg

    def render(self, parent: tk.Widget) -> None:
        tk.Label(
            parent,
            text=self._message,
            font=_FONT,
            fg=self._fg,
            padx=16,
            pady=4,
        ).pack(anchor="w")


# ── Section: 資料集分割 ─────────────────────────────────────


class SplitCountsSection:
    """訓練／驗證集張數與比例。"""

    _COL_W = 8

    def __init__(self, split_counts: dict[str, Any]) -> None:
        self._sc = split_counts

    def render(self, parent: tk.Widget) -> None:
        if not self._sc:
            return
        _section_header(parent, "資料集分割")
        frame = _section_frame(parent)

        for col, h in enumerate(["", "比例", "張數"]):
            _grid_cell(frame, h, 0, col, font=_FONT_BOLD, width=self._COL_W)
        tk.Frame(frame, height=1, bg="#999").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=2
        )

        sc = self._sc
        rows = [
            ("訓練集", sc.get("train_ratio", 0), sc.get("train_size", 0)),
            ("驗證集", sc.get("val_ratio", 0), sc.get("val_size", 0)),
            ("合計", 1.0, sc.get("total_size", 0)),
        ]
        for r, (label, ratio, size) in enumerate(rows, start=2):
            _grid_cell(
                frame, label, r, 0, font=_FONT_BOLD, width=self._COL_W, anchor="w"
            )
            _grid_cell(frame, f"{ratio:.2f}", r, 1, width=self._COL_W)
            _grid_cell(frame, f"{size:,}", r, 2, width=self._COL_W)


# ── Section: 驗證集 F1 ──────────────────────────────────────


class ValF1Section:
    """驗證集 macro F1 與各角色 per-class F1。"""

    _NAME_W = 18
    _F1_W = 10

    def __init__(self, model: dict[str, Any]) -> None:
        self._m = model

    def render(self, parent: tk.Widget) -> None:
        _section_header(parent, "驗證集 F1")
        per_class_f1: list[float] | None = self._m.get("per_class_f1")
        if per_class_f1 is None:
            tk.Label(
                parent,
                text="尚無 F1 資料，請重新執行自動標註。",
                font=_FONT,
                fg="#999",
                padx=16,
            ).pack(anchor="w")
            return

        frame = _section_frame(parent)
        _grid_cell(frame, "角色", 0, 0, font=_FONT_BOLD, width=self._NAME_W, anchor="w")
        _grid_cell(frame, "F1", 0, 1, font=_FONT_BOLD, width=self._F1_W)
        tk.Frame(frame, height=1, bg="#999").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )

        for i, name in enumerate(CLASS_NAMES_LIST):
            _grid_cell(frame, name, i + 2, 0, width=self._NAME_W, anchor="w")
            _grid_cell(frame, f"{per_class_f1[i]:.4f}", i + 2, 1, width=self._F1_W)

        macro_row = len(CLASS_NAMES_LIST) + 2
        _grid_cell(frame, "Macro", macro_row, 0, width=self._NAME_W, anchor="w")
        _grid_cell(frame, str(self._m["val_f1"]), macro_row, 1, width=self._F1_W)


# ── Base viewer ─────────────────────────────────────────────


class _BaseViewer:
    """非模態視窗骨架：背景載入 → 渲染 sections → 重新載入。

    子類別實作：
    - `_preflight()`：視窗開啟前的同步檢查（可選）。
    - `_snapshot_ui_state()`：在 UI 執行緒抓取當前 UI/store 狀態（可選）。
    - `_load_async(ui_state)`：背景執行緒載入資料。
    - `_build_sections(data)`：由資料建立 sections 列表。
    """

    _title: str = ""

    def __init__(self, parent: tk.Tk) -> None:
        self._parent = parent
        self._win: tk.Toplevel | None = None
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._loading_label: tk.Label | None = None

    # ── Lifecycle ──────────────────────────────────────────

    def show(self) -> None:
        if not self._preflight():
            return
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            self._win.focus_force()
            return
        self._win = tk.Toplevel(self._parent)
        self._win.title(self._title)
        self._win.resizable(False, False)
        self._start_load()

    def _start_load(self) -> None:
        assert self._win is not None
        self._loading_label = tk.Label(
            self._win, text="載入中...", font=_FONT, padx=40, pady=20
        )
        self._loading_label.pack()
        ui_state = self._snapshot_ui_state()
        self._thread = threading.Thread(
            target=self._load, args=(ui_state,), daemon=True
        )
        self._thread.start()
        self._poll()

    def _load(self, ui_state: dict[str, Any]) -> None:
        self._result = None
        self._error = None
        try:
            data = self._load_async(ui_state)
            self._result = {**ui_state, **data}
        except Exception as e:
            self._error = str(e)

    def _poll(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._parent.after(100, self._poll)
            return
        self._thread = None
        if self._win is None or not self._win.winfo_exists():
            return
        if self._loading_label is not None:
            self._loading_label.destroy()
            self._loading_label = None
        if self._error is not None:
            tk.Label(
                self._win,
                text=f"錯誤：{self._error}",
                font=_FONT,
                fg="red",
                padx=16,
            ).pack()
            return
        assert self._result is not None
        self._render(self._result)

    def _render(self, data: dict[str, Any]) -> None:
        assert self._win is not None
        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=8, pady=8)
        for section in self._build_sections(data):
            section.render(container)
        tk.Button(container, text="重新載入", command=self._refresh).pack(pady=(8, 0))

    def _refresh(self) -> None:
        if self._win is None or not self._win.winfo_exists():
            return
        for w in self._win.winfo_children():
            w.destroy()
        self._start_load()

    # ── Hooks for subclasses ───────────────────────────────

    def _preflight(self) -> bool:
        return True

    def _snapshot_ui_state(self) -> dict[str, Any]:
        return {}

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        raise NotImplementedError


class _CheckpointViewer(_BaseViewer):
    """需要 checkpoint.pt 存在的 viewer 基底。"""

    def _preflight(self) -> bool:
        if not _CHECKPOINT_PATH.exists():
            messagebox.showwarning(
                self._title,
                f"找不到 checkpoint 檔案：\n{_CHECKPOINT_PATH}",
                parent=self._parent,
            )
            return False
        return True


# ── Concrete viewers ────────────────────────────────────────


class DataOverviewViewer(_BaseViewer):
    """資料概況：圖片數量、變更明細、資料分布檢定。

    分布檢定僅需 LabelStore，因此在沒有 checkpoint 時仍可開啟；
    checkpoint 依賴的 sections 會替換為提示訊息。
    """

    _title = "資料概況"

    def __init__(self, parent: tk.Tk, store: LabelStore) -> None:
        super().__init__(parent)
        self._store = store

    def _snapshot_ui_state(self) -> dict[str, Any]:
        # 在 UI 執行緒抓取 store 快照，避免和標註動作競爭。
        return {"orig_samples": _snapshot_orig_samples(self._store)}

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        if not _CHECKPOINT_PATH.exists():
            return {"checkpoint": None}
        ckpt = _load_checkpoint(_CHECKPOINT_PATH)
        return {
            "checkpoint": {
                "created_at": ckpt.get("created_at"),
                "counts": _extract_counts(ckpt),
                "changes": _extract_changes(ckpt),
            }
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        sections: list[Section] = []
        ckpt = data["checkpoint"]
        if ckpt is None:
            sections.append(
                NoticeSection(
                    f"找不到 checkpoint 檔案，僅顯示資料分布檢定：\n{_CHECKPOINT_PATH}"
                )
            )
        else:
            sections.append(ImageCountsSection(ckpt["counts"], ckpt["created_at"]))
            sections.append(ChangesSection(ckpt["changes"]))
        sections.append(DistributionTestSection(data["orig_samples"]))
        return sections


class ModelInfoViewer(_CheckpointViewer):
    """模型資訊：模型架構、訓練超參數。"""

    _title = "模型資訊"

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        ckpt = _load_checkpoint(_CHECKPOINT_PATH)
        return {
            "file_size_mb": _CHECKPOINT_PATH.stat().st_size / 1024 / 1024,
            "model": _extract_model(ckpt),
            "hyperparams": _extract_hyperparams(ckpt),
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        return [
            ModelArchSection(data["model"], data["file_size_mb"]),
            HyperparamsSection(data["hyperparams"]),
        ]


class ValF1Viewer(_CheckpointViewer):
    """分析結果：資料集分割、驗證集 F1。"""

    _title = "分析結果"

    def _load_async(self, ui_state: dict[str, Any]) -> dict[str, Any]:
        ckpt = _load_checkpoint(_CHECKPOINT_PATH)
        return {
            "model": _extract_model(ckpt),
            "split_counts": _extract_split_counts(),
        }

    def _build_sections(self, data: dict[str, Any]) -> list[Section]:
        return [
            SplitCountsSection(data["split_counts"]),
            ValF1Section(data["model"]),
        ]

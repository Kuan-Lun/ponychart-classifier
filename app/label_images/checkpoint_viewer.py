"""Checkpoint 資訊：非模態視窗顯示訓練 checkpoint 的 metadata。"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

from .constants import IMAGE_DIR


def _load_checkpoint_data(path: Path) -> dict[str, Any]:
    """載入 checkpoint 並整理成結構化資料。

    在背景執行緒中呼叫，避免阻塞 UI。
    """
    import torch

    from ponychart_classifier.training.constants import RAWIMAGE_DIR
    from ponychart_classifier.training.model import BACKBONE_REGISTRY, build_model
    from ponychart_classifier.training.sampling import is_original, load_samples

    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)

    # --- basic ---
    file_size_mb = path.stat().st_size / 1024 / 1024

    # --- image counts ---
    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    samples = load_samples()
    labels_current = {
        str(Path(p).relative_to(RAWIMAGE_DIR.parent)): labels for p, labels in samples
    }

    all_files = [
        f.name
        for f in Path(RAWIMAGE_DIR).rglob("*")
        if f.is_file() and f.suffix.lower() in (".png", ".jpg")
    ]
    n_cur_orig = sum(1 for f in all_files if is_original(f))
    n_cur_crop = len(all_files) - n_cur_orig

    def count_orig_crop(labels: dict[str, list[int]]) -> tuple[int, int]:
        n_o = sum(1 for k in labels if is_original(k.split("/")[-1]))
        return n_o, len(labels) - n_o

    n_orig_full, n_crop_full = count_orig_crop(labels_full)
    n_orig = ckpt.get("n_orig")
    n_crop = ckpt.get("n_crop")
    n_total_last = (n_orig or 0) + (n_crop or 0) if n_orig is not None else None
    n_unlabeled = len(all_files) - len(labels_current)

    # --- changes ---
    def diff_labels(
        baseline: dict[str, list[int]], current: dict[str, list[int]]
    ) -> tuple[set[str], set[str], set[str]]:
        base_keys = set(baseline)
        cur_keys = set(current)
        added = cur_keys - base_keys
        removed = base_keys - cur_keys
        relabeled = {k for k in base_keys & cur_keys if baseline[k] != current[k]}
        return added, removed, relabeled

    def split_orig_crop(keys: set[str]) -> tuple[int, int]:
        n_o = sum(1 for k in keys if is_original(k.split("/")[-1]))
        return n_o, len(keys) - n_o

    def diff_detail(
        baseline: dict[str, list[int]],
    ) -> list[tuple[int, int, int]]:
        added, removed, relabeled = diff_labels(baseline, labels_current)
        return [(len(s), *split_orig_crop(s)) for s in (added, removed, relabeled)]

    full_detail = diff_detail(labels_full)
    last_detail = diff_detail(labels_last)

    # --- model ---
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

    # --- hyperparameters ---
    hp_keys = [
        ("seed", "Seed"),
        ("batch_size", "Batch size"),
        ("lr_head", "LR head"),
        ("lr_features", "LR features"),
        ("lr_classifier", "LR classifier"),
        ("weight_decay", "Weight decay"),
        ("label_smoothing", "Label smoothing"),
    ]
    hyperparams = {
        label: ckpt[key] for key, label in hp_keys if ckpt.get(key) is not None
    }

    val_f1 = ckpt.get("val_f1")

    return {
        "file_size_mb": file_size_mb,
        "created_at": ckpt.get("created_at"),
        "counts": {
            "orig_full": n_orig_full,
            "crop_full": n_crop_full,
            "orig_last": n_orig,
            "crop_last": n_crop,
            "orig_cur": n_cur_orig,
            "crop_cur": n_cur_crop,
            "total_full": len(labels_full),
            "total_last": n_total_last,
            "total_cur": n_cur_orig + n_cur_crop,
            "unlabeled": n_unlabeled,
        },
        "changes": {
            "full": full_detail,
            "last": last_detail,
        },
        "model": {
            "backbone": backbone_name,
            "input_size": ckpt.get("input_size", "N/A"),
            "pre_resize": ckpt.get("pre_resize", "N/A"),
            "num_classes": ckpt.get("num_classes", "N/A"),
            "n_params": n_params,
            "n_keys": len(state_dict),
            "val_size": ckpt.get("val_size", "N/A"),
            "val_f1": f"{val_f1:.4f}" if val_f1 is not None else "N/A",
        },
        "hyperparams": hyperparams,
    }


_CHECKPOINT_PATH = IMAGE_DIR / "checkpoint.pt"
_FONT = ("Consolas", 11)
_FONT_BOLD = ("Consolas", 11, "bold")
_FONT_HEADER = ("Consolas", 12, "bold")


class CheckpointViewer:
    """非模態視窗顯示 checkpoint 的 metadata。"""

    def __init__(self, parent: tk.Tk) -> None:
        self._parent = parent
        self._win: tk.Toplevel | None = None

    def show(self) -> None:
        """顯示或重新聚焦視窗。"""
        if not _CHECKPOINT_PATH.exists():
            messagebox.showwarning(
                "模型資訊",
                f"找不到 checkpoint 檔案：\n{_CHECKPOINT_PATH}",
                parent=self._parent,
            )
            return

        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            self._win.focus_force()
            return

        self._win = tk.Toplevel(self._parent)
        self._win.title("模型資訊")
        self._win.resizable(False, False)

        self._loading_label = tk.Label(
            self._win, text="載入中...", font=_FONT, padx=40, pady=20
        )
        self._loading_label.pack()

        self._thread: threading.Thread | None = threading.Thread(
            target=self._load, daemon=True
        )
        self._thread.start()
        self._poll()

    def _load(self) -> None:
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        try:
            self._result = _load_checkpoint_data(_CHECKPOINT_PATH)
        except Exception as e:
            self._error = str(e)

    def _poll(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._parent.after(100, self._poll)
            return
        self._thread = None
        if self._win is None or not self._win.winfo_exists():
            return
        self._loading_label.destroy()
        if self._error is not None:
            tk.Label(
                self._win, text=f"錯誤：{self._error}", font=_FONT, fg="red", padx=16
            ).pack()
            return
        if self._result is not None:
            self._render(self._result)

    def _render(self, data: dict[str, Any]) -> None:
        assert self._win is not None
        container = tk.Frame(self._win)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        def _pack_section_frame(parent: tk.Widget) -> tk.Frame:
            frame = tk.Frame(parent)
            frame.pack(anchor="w", padx=16, pady=(0, 4))
            return frame

        # --- 基本資訊 ---
        self._section(container, "基本資訊")
        f = _pack_section_frame(container)
        self._kv(f, "檔案大小", f"{data['file_size_mb']:.2f} MB")
        self._kv(f, "最新圖片時間", str(data.get("created_at", "N/A")))

        # --- 圖片數量 ---
        self._section(container, "圖片數量")
        self._render_counts(container, data["counts"])

        # --- 變更明細 ---
        self._section(container, "變更明細")
        self._render_changes(container, data["changes"])

        # --- 模型 ---
        self._section(container, "模型架構")
        f = _pack_section_frame(container)
        m = data["model"]
        self._kv(f, "Backbone", str(m["backbone"]))
        self._kv(f, "Input size", str(m["input_size"]))
        self._kv(f, "Pre-resize", str(m["pre_resize"]))
        self._kv(f, "Classes", str(m["num_classes"]))
        self._kv(f, "Parameters", f"{m['n_params']:,}")
        self._kv(f, "State dict keys", f"{m['n_keys']:,}")
        self._kv(f, "Val size", str(m["val_size"]))
        self._kv(f, "Val F1", str(m["val_f1"]))

        # --- 超參數 ---
        hp = data.get("hyperparams", {})
        if hp:
            self._section(container, "訓練超參數")
            f = _pack_section_frame(container)
            for label, val in hp.items():
                self._kv(f, label, str(val))

        # --- Refresh ---
        tk.Button(container, text="重新載入", command=self._refresh).pack(pady=(8, 0))

    def _refresh(self) -> None:
        if self._win is None or not self._win.winfo_exists():
            return
        for w in self._win.winfo_children():
            w.destroy()
        self._loading_label = tk.Label(
            self._win, text="載入中...", font=_FONT, padx=40, pady=20
        )
        self._loading_label.pack()
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()
        self._poll()

    # ── 表格繪製 helpers ──────────────────────────────────────

    def _section(self, parent: tk.Widget, title: str) -> None:
        tk.Label(parent, text=title, font=_FONT_HEADER, anchor="w").pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        tk.Frame(parent, height=1, bg="#ccc").pack(fill="x", padx=8, pady=(0, 4))

    def _kv(self, parent: tk.Widget, key: str, value: str) -> None:
        row = tk.Frame(parent)
        row.pack(anchor="w")
        tk.Label(row, text=f"{key}:", font=_FONT_BOLD, width=18, anchor="w").pack(
            side="left"
        )
        tk.Label(row, text=value, font=_FONT, anchor="w").pack(side="left")

    def _render_counts(self, parent: tk.Widget, c: dict[str, Any]) -> None:
        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=16, pady=(0, 4))

        headers = ["", "完整訓練", "上次存檔", "目前", "距上次", "距完整訓練"]
        rows = [
            (
                "原圖",
                c["orig_full"],
                c["orig_last"],
                c["orig_cur"],
            ),
            (
                "裁切",
                c["crop_full"],
                c["crop_last"],
                c["crop_cur"],
            ),
            (
                "合計",
                c["total_full"],
                c["total_last"],
                c["total_cur"],
            ),
        ]

        for col, h in enumerate(headers):
            tk.Label(frame, text=h, font=_FONT_BOLD, width=14, anchor="e").grid(
                row=0, column=col, padx=2
            )

        for r, (label, full, last, cur) in enumerate(rows, start=1):
            tk.Label(frame, text=label, font=_FONT_BOLD, width=14, anchor="w").grid(
                row=r, column=0, padx=2
            )
            tk.Label(frame, text=f"{full:,}", font=_FONT, width=14, anchor="e").grid(
                row=r, column=1, padx=2
            )
            last_s = f"{last:,}" if last is not None else "-"
            tk.Label(frame, text=last_s, font=_FONT, width=14, anchor="e").grid(
                row=r, column=2, padx=2
            )
            tk.Label(frame, text=f"{cur:,}", font=_FONT, width=14, anchor="e").grid(
                row=r, column=3, padx=2
            )
            # delta columns
            since_last = self._fmt_diff(cur, last) if last is not None else ""
            since_full = self._fmt_diff(cur, full)
            tk.Label(frame, text=since_last, font=_FONT, width=14, anchor="e").grid(
                row=r, column=4, padx=2
            )
            tk.Label(frame, text=since_full, font=_FONT, width=14, anchor="e").grid(
                row=r, column=5, padx=2
            )

        # unlabeled
        tk.Label(
            frame,
            text=f"({c['unlabeled']} 未標註)",
            font=_FONT,
            fg="#999",
        ).grid(row=len(rows) + 1, column=3, padx=2, sticky="e")

    def _render_changes(self, parent: tk.Widget, changes: dict[str, Any]) -> None:
        frame = tk.Frame(parent)
        frame.pack(anchor="w", padx=16, pady=(0, 4))

        headers = ["", "距完整訓練", "距上次存檔"]
        labels = ["新增", "移除", "重新標註"]

        for col, h in enumerate(headers):
            tk.Label(frame, text=h, font=_FONT_BOLD, width=24, anchor="e").grid(
                row=0, column=col, padx=2
            )

        full_detail = changes["full"]
        last_detail = changes["last"]
        for r, (label, fd, ld) in enumerate(
            zip(labels, full_detail, last_detail), start=1
        ):
            tk.Label(frame, text=label, font=_FONT_BOLD, width=24, anchor="w").grid(
                row=r, column=0, padx=2
            )
            tk.Label(
                frame,
                text=self._detail_cell(*fd),
                font=_FONT,
                width=24,
                anchor="e",
            ).grid(row=r, column=1, padx=2)
            tk.Label(
                frame,
                text=self._detail_cell(*ld),
                font=_FONT,
                width=24,
                anchor="e",
            ).grid(row=r, column=2, padx=2)

    @staticmethod
    def _fmt_diff(cur: int, base: int) -> str:
        diff = cur - base
        ratio = diff / base if base else 0
        return f"{diff:+,d} ({ratio:+.1%})"

    @staticmethod
    def _detail_cell(total: int, n_o: int, n_c: int) -> str:
        return f"{total} ({n_o} 原圖, {n_c} 裁切)"

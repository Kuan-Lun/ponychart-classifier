"""圖片標註工具的主 UI 協調器。"""

import random
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import Literal

from PIL import Image, ImageTk

from .analysis import AnalysisManager, AnalysisTable
from .constants import (
    CLASS_NAMES_LIST,
    CONFLICT_SUBDIR,
    IMAGE_DIR,
    IMAGE_SUBDIR,
    LABEL_FILE,
    LABEL_MAP,
    MAX_SIZE,
)
from .crop_handler import CropHandler
from .file_ops import (
    cleanup_empty_dirs,
    dedup_images,
    is_raw_image,
    organize_single,
    target_path_for,
)
from .filter_builder import FilterConfig, build_filter_fn
from .label_store import LabelStore
from .navigator import ImageNavigator


class LabelApp:
    """圖片標註工具的主 UI 協調器。"""

    def __init__(self, root: tk.Tk, image_paths: list[Path]):
        self.root = root
        self.store = LabelStore(LABEL_FILE, IMAGE_SUBDIR)
        self.nav = ImageNavigator(image_paths, self.store)

        root.title(
            "Pony Chart Labeler"
            " (1..6 標記 | A/D 切換 | S 儲存 | C 裁切 | G 跳轉 | R 隨機)"
        )

        # Canvas 用於圖片顯示與裁切
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack()
        self._canvas_image_id: int | None = None

        self.crop = CropHandler(
            self.canvas,
            on_selection_complete=self._on_crop_selection_complete,
        )
        self.scale: float = 1.0
        self.current_pil_image: Image.Image | None = None

        # 數字與角色對應
        mapping_text = "  |  ".join(f"{k}: {v}" for k, v in LABEL_MAP.items())
        tk.Label(root, text=mapping_text, fg="#666", font=("Consolas", 11)).pack(
            pady=(4, 2)
        )

        tk.Label(
            root,
            text=(
                "1..6 加/取消標籤  |  A 上一張  |  D 下一張"
                "  |  S 儲存  |  C 裁切  |  G 跳轉  |  R 隨機"
            ),
            fg="#666",
        ).pack(pady=(0, 6))

        # 篩選選項
        self._build_filter_ui(root)

        # Model analysis
        self._analysis = AnalysisManager()
        self._build_action_buttons(root)
        self._build_analysis_ui(root)

        # 計數器（可點擊跳轉）
        self.counter_label = tk.Label(
            root, text="", font=("Consolas", 12), fg="#0066cc", cursor="hand2"
        )
        self.counter_label.pack()
        self.counter_label.bind("<Button-1>", lambda _: self._jump_to_image())

        # 標籤與檔名資訊
        self.info_label = tk.Label(root, text="", font=("Consolas", 12))
        self.info_label.pack()

        self.current_labels: list[int] = []
        root.bind("<Key>", self._on_key)
        self._refresh()

    # ── UI 建構 ──────────────────────────────────────────────

    def _build_filter_ui(self, root: tk.Tk) -> None:
        filter_frame = tk.Frame(root)
        filter_frame.pack(pady=(0, 4))

        self.raw_only_var = tk.BooleanVar(value=False)
        self.raw_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示原圖",
            variable=self.raw_only_var,
            command=self._on_raw_toggle,
        )
        self.raw_only_cb.pack(side="left", padx=(0, 8))

        self.uncropped_only_var = tk.BooleanVar(value=False)
        self.uncropped_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示未裁切原圖",
            variable=self.uncropped_only_var,
            command=self._on_uncropped_toggle,
        )
        self.uncropped_only_cb.pack(side="left", padx=(0, 8))

        self.crop_mismatch_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="裁切標籤不符",
            variable=self.crop_mismatch_var,
            command=self._apply_filters,
        ).pack(side="left", padx=(0, 8))

        self.filter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="只顯示未標註",
            variable=self.filter_var,
            command=self._apply_filters,
        ).pack(side="left", padx=(0, 8))

        self.train_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="只顯示 Train",
            variable=self.train_only_var,
            command=self._apply_filters,
        ).pack(side="left")

        # Class filter
        class_frame = tk.Frame(root)
        class_frame.pack(pady=(0, 4))
        tk.Label(class_frame, text="Class:").pack(side="left")
        self._class_vars: list[tk.BooleanVar] = []
        for short in CLASS_NAMES_LIST:
            var = tk.BooleanVar(value=False)
            self._class_vars.append(var)
            tk.Checkbutton(
                class_frame,
                text=short,
                variable=var,
                command=self._apply_filters,
            ).pack(side="left", padx=(2, 2))

    def _build_action_buttons(self, root: tk.Tk) -> None:
        action_frame = tk.Frame(root)
        action_frame.pack(pady=(0, 4))

        self._delete_btn = tk.Button(
            action_frame,
            text="刪除此裁切圖",
            fg="red",
            command=self._delete_crop,
            state="disabled",
        )
        self._delete_btn.pack(side="left", padx=(0, 16))

        tk.Button(
            action_frame,
            text="清理孤兒標籤",
            command=self._purge_orphans,
        ).pack(side="left", padx=(0, 16))

        tk.Button(
            action_frame,
            text="全部整理",
            command=self._organize_all,
        ).pack(side="left")

    def _build_analysis_ui(self, root: tk.Tk) -> None:
        analyze_frame = tk.Frame(root)
        analyze_frame.pack(pady=(0, 4))
        self._analyze_btn = tk.Button(
            analyze_frame,
            text="Analyze labels",
            command=self._start_analysis,
        )
        self._analyze_btn.pack(side="left", padx=(0, 8))
        self._analyze_status = tk.Label(analyze_frame, text="", fg="#999")
        self._analyze_status.pack(side="left")

        self._analysis_table = AnalysisTable(root)

        # Suspicious filter checkboxes
        suspicious_frame = tk.Frame(root)
        suspicious_frame.pack(pady=(0, 4))

        self._mislabel_var = tk.BooleanVar(value=False)
        self._mislabel_cb = tk.Checkbutton(
            suspicious_frame,
            text="Mislabel (−)",
            variable=self._mislabel_var,
            command=self._apply_filters,
        )
        self._mislabel_cb.pack(side="left", padx=(0, 4))

        self._missing_var = tk.BooleanVar(value=False)
        self._missing_cb = tk.Checkbutton(
            suspicious_frame,
            text="Missing (+)",
            variable=self._missing_var,
            command=self._apply_filters,
        )
        self._missing_cb.pack(side="left", padx=(0, 4))

        self._set_suspicious_controls_state("disabled")

    # ── UI 更新 ──────────────────────────────────────────────

    def _update_display(self, extra: str = "") -> None:
        self.counter_label.configure(text=f"{self.nav.index + 1} / {self.nav.total}")
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        info = f"labels: {label_names}\n{self.nav.current_key}"
        if extra:
            info += f"  ({extra})"
        self.info_label.configure(text=info)
        self._analysis_table.update(
            self._analysis.get_image_probs(self.nav.current_key),
            self._analysis.model_thresholds,
            self.current_labels,
            self.counter_label,
        )

    def _refresh(self) -> None:
        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self._reset_all_filters()
                if self.nav.is_empty:
                    self._show_no_images()
                    return
            else:
                self._show_no_images()
                return

        self.crop.exit()
        self._load_image(self.nav.current_path)
        self.current_labels = sorted(set(self.store.get(self.nav.current_key)))
        if is_raw_image(self.nav.current_path):
            self._delete_btn.configure(state="disabled")
        else:
            self._delete_btn.configure(state="normal")
        self._update_display()

    def _load_image(self, path: Path) -> None:
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {path}: {e}")
            return

        self.current_pil_image = im
        w, h = im.size
        self.scale = min(MAX_SIZE / max(1, w), MAX_SIZE / max(1, h), 1.0)
        display_im = im
        if self.scale < 1.0:
            display_im = im.resize((int(w * self.scale), int(h * self.scale)))

        self.tk_im = ImageTk.PhotoImage(display_im)
        dw, dh = display_im.size
        self.canvas.configure(width=dw, height=dh)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self.tk_im
        )

    def _show_no_images(self) -> None:
        messagebox.showinfo("Info", "No images found under ./rawimage")
        self.root.destroy()

    # ── 操作 ─────────────────────────────────────────────────

    def _toggle_label(self, v: int) -> None:
        if v in self.current_labels:
            self.current_labels.remove(v)
        else:
            self.current_labels.append(v)
            self.current_labels.sort()
        self._update_display()

    def _save(self) -> None:
        key = self.nav.current_key
        old_path = self.nav.current_path
        self.store.set(key, self.current_labels)

        new_path = organize_single(old_path, self.current_labels)
        if new_path != old_path:
            new_key = self.store.path_to_key(new_path)
            self.store.rename_key(key, new_key)
            self.nav.replace_path(old_path, new_path)
            key = new_key

        self.store.save()

        if not self.nav.is_filtered:
            self._update_display("saved")
            return

        self.nav.refresh_filter()
        if self.nav.is_empty:
            messagebox.showinfo("Info", "篩選結果為空。")
            self._reset_all_filters()
            self.nav.go_to_key(key)
        elif not self.nav.contains_key(key):
            self.nav.advance_after_label(key)
        self._refresh()

    def _delete_crop(self) -> None:
        if self.nav.is_empty:
            return
        path = self.nav.current_path
        if is_raw_image(path):
            messagebox.showwarning("刪除", "無法刪除原圖。")
            return

        confirm = messagebox.askyesno(
            "確認刪除",
            f"確定要刪除此裁切圖嗎？\n{path.name}\n\n此操作無法復原。",
        )
        if not confirm:
            return

        key = self.nav.current_key
        self.store.delete(key)
        self.store.save()
        self.nav.remove_path(path)
        path.unlink(missing_ok=True)

        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self._reset_all_filters()
            if self.nav.is_empty:
                self._show_no_images()
                return
        self._refresh()

    def _purge_orphans(self) -> None:
        orphans = self.store.purge_orphans(IMAGE_DIR)
        if not orphans:
            messagebox.showinfo("清理孤兒標籤", "沒有孤兒標籤。")
            return
        confirm = messagebox.askyesno(
            "清理孤兒標籤",
            f"發現 {len(orphans)} 筆孤兒標籤"
            f"（檔案不存在）：\n\n"
            + "\n".join(orphans[:20])
            + ("\n..." if len(orphans) > 20 else "")
            + "\n\n確定要從 labels.json 移除？",
        )
        if not confirm:
            return
        self.store.save()
        messagebox.showinfo(
            "清理孤兒標籤",
            f"已移除 {len(orphans)} 筆孤兒標籤。",
        )

    def _organize_all(self) -> None:
        dups = dedup_images(list(self.nav.all_paths))
        n_dedup = 0
        if dups:
            dup_lines = "\n".join(
                f"  刪除: {d.name}  (保留: {o.name})" for d, o in dups[:20]
            )
            if len(dups) > 20:
                dup_lines += "\n  ..."
            if not messagebox.askyesno(
                "去重",
                f"發現 {len(dups)} 張重複圖片（SHA-256 相同）：\n\n"
                + dup_lines
                + "\n\n刪除重複、保留最舊的？",
            ):
                return
            for dup_path, _orig in dups:
                dup_key = self.store.path_to_key(dup_path)
                self.store.delete(dup_key)
                self.nav.remove_path(dup_path)
                dup_path.unlink()
                n_dedup += 1
            self.store.save()

        pending: list[tuple[Path, str]] = []
        for p in list(self.nav.all_paths):
            key = self.store.path_to_key(p)
            labels = self.store.get(key)
            target = target_path_for(p.name, labels)
            if p != target:
                pending.append((p, key))

        if not pending and not n_dedup:
            messagebox.showinfo("全部整理", "所有圖片已在正確位置，無重複。")
            return

        n_moved = 0
        n_conflict = 0
        if pending:
            confirm = messagebox.askyesno(
                "全部整理",
                f"將整理 {len(pending)} 張圖片到對應的子資料夾。\n\n"
                + "\n".join(f"  {p.name}" for p, _ in pending[:20])
                + ("\n  ..." if len(pending) > 20 else "")
                + "\n\n確定執行？",
            )
            if not confirm:
                return

            for old_path, old_key in pending:
                labels = self.store.get(old_key)
                new_path = organize_single(old_path, labels)
                if new_path != old_path:
                    new_key = self.store.path_to_key(new_path)
                    self.store.rename_key(old_key, new_key)
                    self.nav.replace_path(old_path, new_path)
                    n_moved += 1
                    if CONFLICT_SUBDIR in new_path.parts:
                        n_conflict += 1

            self.store.save()

        cleanup_empty_dirs(IMAGE_DIR)

        parts = []
        if n_dedup:
            parts.append(f"已刪除 {n_dedup} 張重複圖片。")
        if n_moved:
            parts.append(f"已搬移 {n_moved} 張圖片。")
        if n_conflict:
            parts.append(f"其中 {n_conflict} 張因重複被移至 {CONFLICT_SUBDIR}/。")
        if not parts:
            parts.append("所有圖片已在正確位置，無重複。")
        messagebox.showinfo("全部整理", "\n".join(parts))
        self._refresh()

    # ── 篩選 ─────────────────────────────────────────────────

    def _on_raw_toggle(self) -> None:
        if not self.raw_only_var.get():
            self.uncropped_only_var.set(False)
            self.uncropped_only_cb.configure(state="normal")
        self._apply_filters()

    def _on_uncropped_toggle(self) -> None:
        if self.uncropped_only_var.get():
            self.raw_only_var.set(True)
            self.raw_only_cb.configure(state="disabled")
        else:
            self.raw_only_cb.configure(state="normal")
        self._apply_filters()

    def _apply_filters(self) -> None:
        config = FilterConfig(
            raw_only=self.raw_only_var.get(),
            uncropped_only=self.uncropped_only_var.get(),
            unlabeled_only=self.filter_var.get(),
            crop_mismatch=self.crop_mismatch_var.get(),
            train_only=self.train_only_var.get(),
            selected_classes=[i for i, var in enumerate(self._class_vars) if var.get()],
            show_mislabel=self._mislabel_var.get(),
            show_missing=self._missing_var.get(),
            model_probs=self._analysis.model_probs,
            model_thresholds=self._analysis.model_thresholds,
        )
        fn = build_filter_fn(config, self.nav.all_paths, self.store)
        if not self.nav.apply_filter(fn):
            messagebox.showinfo("Info", "篩選結果為空。")
            self._reset_all_filters()
        self._refresh()

    def _reset_all_filters(self) -> None:
        self.raw_only_var.set(False)
        self.uncropped_only_var.set(False)
        self.crop_mismatch_var.set(False)
        self.filter_var.set(False)
        self.train_only_var.set(False)
        self.raw_only_cb.configure(state="normal")
        self.uncropped_only_cb.configure(state="normal")
        self._mislabel_var.set(False)
        self._missing_var.set(False)
        for var in self._class_vars:
            var.set(False)
        self.nav.apply_filter(None)

    # ── 分析 ─────────────────────────────────────────────────

    def _start_analysis(self) -> None:
        if self._analysis.is_running:
            return
        self._analyze_btn.configure(state="disabled")
        count = len(self.nav.all_paths)
        self._analyze_status.configure(text=f"Analyzing {count} images...")

        self._analysis.start(
            nav=self.nav,
            store=self.store,
            on_complete=self._on_analysis_complete,
            on_error=self._on_analysis_error,
            root=self.root,
        )

    def _on_analysis_complete(self) -> None:
        count = len(self._analysis.model_probs) if self._analysis.model_probs else 0
        self._analyze_status.configure(text=f"Done ({count} images)")
        self._set_suspicious_controls_state("normal")
        self._refresh()

    def _on_analysis_error(self, error: str) -> None:
        self._analyze_btn.configure(state="normal")
        self._analyze_status.configure(text="")
        messagebox.showerror("Analysis Error", error)

    def _set_suspicious_controls_state(
        self,
        state: Literal["normal", "disabled"],
    ) -> None:
        self._mislabel_cb.configure(state=state)
        self._missing_cb.configure(state=state)

    # ── 跳轉 ─────────────────────────────────────────────────

    def _jump_to_random(self) -> None:
        if self.nav.total <= 1:
            return
        n = random.randint(1, self.nav.total)
        self.nav.go_to(n)
        self._refresh()

    def _jump_to_image(self) -> None:
        n = simpledialog.askinteger(
            "跳轉",
            f"輸入圖片編號 (1-{self.nav.total})：",
            minvalue=1,
            maxvalue=self.nav.total,
            parent=self.root,
        )
        if n is not None:
            self.nav.go_to(n)
            self._refresh()

    # ── 裁切 ─────────────────────────────────────────────────

    def _on_crop_selection_complete(self) -> None:
        self.info_label.configure(text="裁切模式：Enter 確認儲存，Escape 取消")

    def _save_crop(self) -> None:
        if self.current_pil_image is None:
            return
        sel = self.crop.get_selection()
        if sel is None:
            messagebox.showwarning("裁切", "選取區域太小，請重新拖曳。")
            return

        x1, y1, x2, y2 = sel
        w, h = self.current_pil_image.size
        orig = (
            max(0, min(int(x1 / self.scale), w)),
            max(0, min(int(y1 / self.scale), h)),
            max(0, min(int(x2 / self.scale), w)),
            max(0, min(int(y2 / self.scale), h)),
        )
        cropped = self.current_pil_image.crop(orig)

        current_path = self.nav.current_path
        base_stem = re.sub(r"_crop\d+$", "", current_path.stem)
        base_path = current_path.parent / f"{base_stem}{current_path.suffix}"
        save_path = self._next_crop_name(base_path)

        cropped.save(save_path)
        self.nav.add_path(save_path)
        self._refresh()
        self._update_display(f"已儲存裁切圖：{save_path.name}")

    @staticmethod
    def _next_crop_name(base_path: Path) -> Path:
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        n = 1
        while True:
            candidate = parent / f"{stem}_crop{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    # ── 鍵盤事件 ─────────────────────────────────────────────

    def _on_key(self, e: "tk.Event[tk.Misc]") -> None:
        k = e.keysym.lower()

        if self.crop.active:
            match k:
                case "return":
                    self._save_crop()
                case "escape":
                    self.crop.exit()
                    self._refresh()
            return

        match k:
            case "1" | "2" | "3" | "4" | "5" | "6":
                self._toggle_label(int(k))
            case "a":
                self.nav.go_prev()
                self._refresh()
            case "d":
                self.nav.go_next()
                self._refresh()
            case "s":
                self._save()
            case "c":
                self.crop.enter()
                self.info_label.configure(
                    text="裁切模式：拖曳選取區域，Enter 確認，Escape 取消"
                )
            case "g":
                self._jump_to_image()
            case "r":
                self._jump_to_random()

"""圖片標註工具的主 UI 協調器。"""

import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from .analysis import AnalysisManager, AnalysisTable
from .checkpoint_viewer import DataStatusViewer, ModelInfoViewer
from .constants import IMAGE_SUBDIR, LABEL_FILE, LABEL_MAP
from .file_actions import FileActions
from .file_ops import is_raw_image, organize_single
from .filter_panel import FilterPanel
from .image_viewer import ImageViewer
from .label_store import LabelStore
from .navigator import ImageNavigator


class LabelApp:
    """圖片標註工具的主 UI 協調器。"""

    def __init__(self, root: tk.Tk, image_paths: list[Path]):
        self.root = root
        self.store = LabelStore(LABEL_FILE, IMAGE_SUBDIR)
        self.nav = ImageNavigator(image_paths, self.store)
        self._file_actions = FileActions(self.nav, self.store)

        root.title(
            "Pony Chart Labeler"
            " (1..6 標記 | A/D 切換 | S 儲存 | C 裁切 | G 跳轉 | R 隨機)"
        )

        # 圖片顯示
        self._viewer = ImageViewer(
            root, on_crop_complete=self._on_crop_selection_complete
        )

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

        # 篩選面板
        self._filter_panel = FilterPanel(root, on_filter_changed=self._apply_filters)

        # Model analysis & checkpoint viewer
        self._analysis = AnalysisManager()
        self._data_status_viewer = DataStatusViewer(root)
        self._model_info_viewer = ModelInfoViewer(root)
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
            command=self._file_actions.purge_orphans,
        ).pack(side="left", padx=(0, 16))

        tk.Button(
            action_frame,
            text="全部整理",
            command=self._organize_all,
        ).pack(side="left", padx=(0, 16))

        tk.Button(
            action_frame,
            text="資料狀態",
            command=self._data_status_viewer.show,
        ).pack(side="left", padx=(0, 16))

        tk.Button(
            action_frame,
            text="模型資訊",
            command=self._model_info_viewer.show,
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

        self._viewer.crop.exit()
        self._viewer.load(self.nav.current_path)
        self.current_labels = sorted(set(self.store.get(self.nav.current_key)))
        if is_raw_image(self.nav.current_path):
            self._delete_btn.configure(state="disabled")
        else:
            self._delete_btn.configure(state="normal")
        self._update_display()

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
        if not self._file_actions.delete_crop():
            return
        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self._reset_all_filters()
            if self.nav.is_empty:
                self._show_no_images()
                return
        self._refresh()

    def _organize_all(self) -> None:
        self._file_actions.organize_all()
        self._refresh()

    # ── 篩選 ─────────────────────────────────────────────────

    def _apply_filters(self) -> None:
        fn = self._filter_panel.build_filter_fn(
            self.nav,
            self.store,
            model_probs=self._analysis.model_probs,
            model_thresholds=self._analysis.model_thresholds,
        )
        if not self.nav.apply_filter(fn):
            messagebox.showinfo("Info", "篩選結果為空。")
            self._reset_all_filters()
        self._refresh()

    def _reset_all_filters(self) -> None:
        self._filter_panel.reset()
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
        self._filter_panel.set_suspicious_state("normal")
        self._refresh()

    def _on_analysis_error(self, error: str) -> None:
        self._analyze_btn.configure(state="normal")
        self._analyze_status.configure(text="")
        messagebox.showerror("Analysis Error", error)

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
        save_path = self._viewer.save_crop(self.nav.current_path)
        if save_path is None:
            return
        self.nav.add_path(save_path)
        self._refresh()
        self._update_display(f"已儲存裁切圖：{save_path.name}")

    # ── 鍵盤事件 ─────────────────────────────────────────────

    def _on_key(self, e: "tk.Event[tk.Misc]") -> None:
        k = e.keysym.lower()

        if self._viewer.crop.active:
            match k:
                case "return":
                    self._save_crop()
                case "escape":
                    self._viewer.crop.exit()
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
                self._viewer.crop.enter()
                self.info_label.configure(
                    text="裁切模式：拖曳選取區域，Enter 確認，Escape 取消"
                )
            case "g":
                self._jump_to_image()
            case "r":
                self._jump_to_random()

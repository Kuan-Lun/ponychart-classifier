"""圖片標註工具的主 UI 協調器。"""

import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from ponychart_classifier.training.sampling import Sample

from .analysis import AnalysisManager, AnalysisTable
from .constants import IMAGE_SUBDIR, LABEL_FILE, LABEL_MAP
from .data_viewer import DataOverviewViewer, ModelInfoViewer, ValF1Viewer
from .file_actions import FileActions
from .file_ops import is_raw_image, organize_single
from .filter_panel import FilterPanel
from .image_viewer import ImageViewer
from .label_store import LabelStore
from .navigator import ImageNavigator


class _KeyHandler:
    """鍵盤事件分派：將按鍵對應到 LabelApp 操作。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app

    def handle(self, e: "tk.Event[tk.Misc]") -> None:
        k = e.keysym.lower()
        a = self._app

        if a.viewer.crop.active:
            fast = e.keysym.isupper()
            match k:
                case "return":
                    a.crop_actions.save()
                case "escape":
                    a.viewer.crop.exit()
                    a.refresh()
                case "w":
                    a.viewer.crop.move_up(fast)
                case "s":
                    a.viewer.crop.move_down(fast)
                case "a":
                    a.viewer.crop.move_left(fast)
                case "d":
                    a.viewer.crop.move_right(fast)
                case "q":
                    a.viewer.crop.rotate_ccw(fast)
                case "e":
                    a.viewer.crop.rotate_cw(fast)
                case "r":
                    a.viewer.crop.scale_up(fast)
                case "f":
                    a.viewer.crop.scale_down(fast)
            return

        match k:
            case "1" | "2" | "3" | "4" | "5" | "6":
                a.label_actions.toggle(int(k))
            case "a":
                a.nav.go_prev()
                a.refresh()
            case "d":
                a.nav.go_next()
                a.refresh()
            case "s":
                a.label_actions.save()
            case "c":
                a.crop_actions.enter()
            case "g":
                a.nav_actions.jump_to_image()
            case "r":
                a.nav_actions.jump_to_random()


class _LabelActions:
    """標籤操作：切換與儲存。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app

    def toggle(self, v: int) -> None:
        labels = self._app.current_labels
        if v in labels:
            labels.remove(v)
        else:
            labels.append(v)
            labels.sort()
        self._app.update_display()

    def save(self) -> None:
        a = self._app
        key = a.nav.current_key
        old_path = a.nav.current_path
        a.store.set(key, a.current_labels)

        new_path = organize_single(old_path, a.current_labels)
        if new_path != old_path:
            new_key = a.store.path_to_key(new_path)
            a.store.rename_key(key, new_key)
            a.analysis.rename_key(key, new_key)
            a.nav.replace_path(old_path, new_path)
            key = new_key

        a.store.save()

        if not a.nav.is_filtered:
            a.update_display("saved")
            return

        a.nav.refresh_filter()
        if a.nav.is_empty:
            messagebox.showinfo("Info", "篩選結果為空。")
            a.filter_actions.reset()
            a.nav.go_to_key(key)
        elif not a.nav.contains_key(key):
            a.nav.advance_after_label(key)
        a.refresh()


class _NavigationActions:
    """導航操作：跳轉、隨機。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app

    def jump_to_random(self) -> None:
        a = self._app
        if a.nav.total <= 1:
            return
        n = random.randint(1, a.nav.total)
        a.nav.go_to(n)
        a.refresh()

    def jump_to_image(self) -> None:
        a = self._app
        n = simpledialog.askinteger(
            "跳轉",
            f"輸入圖片編號 (1-{a.nav.total})：",
            minvalue=1,
            maxvalue=a.nav.total,
            parent=a.root,
        )
        if n is not None:
            a.nav.go_to(n)
            a.refresh()


class _CropActions:
    """裁切操作：進入裁切模式、儲存裁切。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app

    def enter(self) -> None:
        a = self._app
        a.viewer.crop.enter(a.viewer.display_size, a.viewer.scale)
        a.info_label.configure(
            text="裁切模式：WASD 移動、QE 旋轉、RF 縮放（按住 Shift 加大幅度），"
            "Enter 確認，Escape 取消"
        )

    def save(self) -> None:
        a = self._app
        save_path = a.viewer.save_crop(a.nav.current_path)
        if save_path is None:
            return
        a.nav.add_path(save_path)
        a.refresh()
        a.update_display(f"已儲存裁切圖：{save_path.name}")


class _FilterActions:
    """篩選操作：套用與重置。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app

    def apply(self) -> None:
        a = self._app
        fn = a.filter_panel.build_filter_fn(
            a.nav,
            a.store,
            model_probs=a.analysis.model_probs,
            model_thresholds=a.analysis.model_thresholds,
        )
        if not a.nav.apply_filter(fn):
            messagebox.showinfo("Info", "篩選結果為空。")
            self.reset()
        a.refresh()

    def reset(self) -> None:
        a = self._app
        a.filter_panel.reset()
        a.nav.apply_filter(None)


class _AnalysisActions:
    """模型分析操作：啟動、完成回呼、錯誤回呼。"""

    def __init__(self, app: "LabelApp") -> None:
        self._app = app
        self._just_ran_full = False

    def start(self) -> None:
        """完整模式：對所有尚未有預測的圖片補跑推論。"""
        a = self._app
        existing = a.analysis.model_probs or {}
        samples, keys = a.analysis.collect_samples(
            a.nav, a.store, key_filter=lambda k: k not in existing
        )
        if not samples:
            messagebox.showinfo("Info", "所有圖片皆已分析過。")
            return
        self._launch(samples, keys, scope="", full=True)

    def start_unlabeled_only(self) -> None:
        """僅未標註模式：對尚未手動標註且尚未分析的圖片跑推論。"""
        a = self._app
        existing = a.analysis.model_probs or {}
        samples, keys = a.analysis.collect_samples(
            a.nav,
            a.store,
            key_filter=lambda k: not a.store.has(k) and k not in existing,
        )
        if not samples:
            messagebox.showinfo("Info", "目前沒有未標註且未分析的圖片。")
            return
        self._launch(samples, keys, scope="unlabeled ", full=False)

    def _launch(
        self,
        samples: list[Sample],
        keys: list[str],
        *,
        scope: str,
        full: bool,
    ) -> None:
        a = self._app
        if a.analysis.is_running:
            return
        a.analyze_btn.configure(state="disabled")
        a.analyze_unlabeled_btn.configure(state="disabled")
        a.analyze_status.configure(text=f"Analyzing 0 / {len(samples)}...")
        self._just_ran_full = full

        a.analysis.start(
            samples=samples,
            keys=keys,
            on_complete=self._on_complete,
            on_error=self._on_error,
            root=a.root,
            on_progress=self._on_progress,
        )

    def _on_complete(self) -> None:
        a = self._app
        # 「僅未標註」跑完它涵蓋的圖片後就沒有事可做（兩種模式都會涵蓋未標註集合）。
        a.analyze_unlabeled_btn.configure(state="disabled")
        if self._just_ran_full:
            # 完整模式跑完後，model_probs 已涵蓋所有圖片，「自動標註」自身也無事可做。
            a.analyze_btn.configure(state="disabled")
        else:
            a.analyze_btn.configure(state="normal")
        a.analyze_status.configure(text="")
        a.filter_panel.set_suspicious_state("normal")
        a.refresh()

    def _on_progress(self, done: int, total: int) -> None:
        self._app.analyze_status.configure(text=f"Analyzing {done} / {total}...")

    def _on_error(self, error: str) -> None:
        a = self._app
        a.analyze_btn.configure(state="normal")
        a.analyze_unlabeled_btn.configure(state="normal")
        a.analyze_status.configure(text="")
        messagebox.showerror("Analysis Error", error)


class LabelApp:
    """圖片標註工具的主 UI 協調器。"""

    def __init__(self, root: tk.Tk, image_paths: list[Path]):
        self.root = root
        self.store = LabelStore(LABEL_FILE, IMAGE_SUBDIR)
        self.nav = ImageNavigator(image_paths, self.store)
        self.analysis = AnalysisManager()

        # Action groups
        self.label_actions = _LabelActions(self)
        self.nav_actions = _NavigationActions(self)
        self.crop_actions = _CropActions(self)
        self.filter_actions = _FilterActions(self)
        self.analysis_actions = _AnalysisActions(self)
        self._file_actions = FileActions(self.nav, self.store, self.analysis)
        self._key_handler = _KeyHandler(self)

        root.title(
            "Pony Chart Labeler"
            " (1..6 標記 | A/D 切換 | S 儲存 | C 裁切 | G 跳轉 | R 隨機)"
        )

        # 圖片顯示
        self.viewer = ImageViewer(root)

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
        self.filter_panel = FilterPanel(
            root, on_filter_changed=self.filter_actions.apply
        )

        # Model analysis & checkpoint viewer
        self._data_overview_viewer = DataOverviewViewer(root, self.store)
        self._model_info_viewer = ModelInfoViewer(root)
        self._val_f1_viewer = ValF1Viewer(root)
        self._build_action_buttons(root)
        self._analysis_table = AnalysisTable(root)

        # 計數器（可點擊跳轉）
        self.counter_label = tk.Label(
            root, text="", font=("Consolas", 12), fg="#0066cc", cursor="hand2"
        )
        self.counter_label.pack()
        self.counter_label.bind(
            "<Button-1>", lambda _: self.nav_actions.jump_to_image()
        )

        # 標籤與檔名資訊
        self.info_label = tk.Label(root, text="", font=("Consolas", 12))
        self.info_label.pack()

        self.current_labels: list[int] = []
        root.bind("<Key>", self._key_handler.handle)
        if self.analysis.model_probs is not None:
            self.filter_panel.set_suspicious_state("normal")
        self.refresh()

    # ── UI 建構 ──────────────────────────────────────────────

    def _build_action_buttons(self, root: tk.Tk) -> None:
        # 第 1 排：自動標註 + 可疑篩選
        primary_frame = tk.Frame(root)
        primary_frame.pack(pady=(0, 4))

        self.analyze_btn = tk.Button(
            primary_frame,
            text="自動標註",
            command=self.analysis_actions.start,
        )
        self.analyze_btn.pack(side="left", padx=(0, 4))
        self.analyze_unlabeled_btn = tk.Button(
            primary_frame,
            text="自動標註(僅未標註)",
            command=self.analysis_actions.start_unlabeled_only,
        )
        self.analyze_unlabeled_btn.pack(side="left", padx=(0, 4))
        self.analyze_status = tk.Label(primary_frame, text="", fg="#999")
        self.analyze_status.pack(side="left", padx=(0, 4))

        tk.Button(
            primary_frame,
            text="分析結果",
            command=self._val_f1_viewer.show,
        ).pack(side="left", padx=(0, 4))

        self.filter_panel.build_suspicious_checkboxes(primary_frame)

        # 第 2 排：資訊查看
        info_frame = tk.Frame(root)
        info_frame.pack(pady=(0, 4))

        tk.Button(
            info_frame,
            text="資料概況",
            command=self._data_overview_viewer.show,
        ).pack(side="left", padx=(0, 16))

        tk.Button(
            info_frame,
            text="模型資訊",
            command=self._model_info_viewer.show,
        ).pack(side="left")

        # 第 3 排：批次操作 + 破壞性操作
        batch_frame = tk.Frame(root)
        batch_frame.pack(pady=(0, 4))

        tk.Button(
            batch_frame,
            text="全部整理",
            command=self._organize_all,
        ).pack(side="left", padx=(0, 16))

        self._delete_btn = tk.Button(
            batch_frame,
            text="刪除此裁切圖",
            fg="red",
            command=self._delete_crop,
            state="disabled",
        )
        self._delete_btn.pack(side="left")

    # ── UI 更新 ──────────────────────────────────────────────

    def update_display(self, extra: str = "") -> None:
        self.counter_label.configure(text=f"{self.nav.index + 1} / {self.nav.total}")
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        info = f"labels: {label_names}\n{self.nav.current_key}"
        if extra:
            info += f"  ({extra})"
        self.info_label.configure(text=info)
        self._analysis_table.update(
            self.analysis.get_image_probs(self.nav.current_key),
            self.analysis.model_thresholds,
            self.current_labels,
            self.counter_label,
        )

    def refresh(self) -> None:
        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self.filter_actions.reset()
                if self.nav.is_empty:
                    self._show_no_images()
                    return
            else:
                self._show_no_images()
                return

        self.viewer.crop.exit()
        self.viewer.load(self.nav.current_path)
        self.current_labels = sorted(set(self.store.get(self.nav.current_key)))
        if is_raw_image(self.nav.current_path):
            self._delete_btn.configure(state="disabled")
        else:
            self._delete_btn.configure(state="normal")
        self.update_display()

    def _show_no_images(self) -> None:
        messagebox.showinfo("Info", "No images found under ./rawimage")
        self.root.destroy()

    # ── 檔案操作 ─────────────────────────────────────────────

    def _delete_crop(self) -> None:
        if not self._file_actions.delete_crop():
            return
        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self.filter_actions.reset()
            if self.nav.is_empty:
                self._show_no_images()
                return
        self.refresh()

    def _organize_all(self) -> None:
        self._file_actions.organize_all()
        self.refresh()

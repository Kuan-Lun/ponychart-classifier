"""批次檔案操作：刪除裁切圖、全部整理（含去重、清理孤兒標籤、搬移）。"""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from .constants import CONFLICT_SUBDIR, IMAGE_DIR, MAX_SIZE
from .file_ops import (
    cleanup_empty_dirs,
    dedup_images,
    dedup_near_images,
    is_raw_image,
    organize_single,
    target_path_for,
)
from .label_store import LabelStore
from .navigator import ImageNavigator


def _load_thumbnail(path: Path) -> Image.Image:
    """載入圖片並縮放至預覽大小。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    preview_max = MAX_SIZE // 2
    scale = min(preview_max / max(1, w), preview_max / max(1, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _confirm_near_dup(dup_path: Path, orig_path: Path) -> bool:
    """彈出視窗並排顯示兩張圖片，讓使用者決定是否刪除較新的。"""
    dlg = tk.Toplevel()
    dlg.title("近似重複圖片確認")
    dlg.grab_set()

    result: list[bool] = [False]

    img_keep = _load_thumbnail(orig_path)
    img_del = _load_thumbnail(dup_path)
    tk_keep = ImageTk.PhotoImage(img_keep)
    tk_del = ImageTk.PhotoImage(img_del)

    # 保留（左）
    frm_keep = tk.Frame(dlg)
    frm_keep.pack(side="left", padx=8, pady=8)
    tk.Label(frm_keep, text="保留（較早）", font=("", 13, "bold")).pack()
    tk.Label(frm_keep, text=orig_path.name, wraplength=400).pack()
    tk.Label(frm_keep, image=tk_keep).pack()

    # 刪除（右）
    frm_del = tk.Frame(dlg)
    frm_del.pack(side="left", padx=8, pady=8)
    tk.Label(frm_del, text="刪除（較晚）", font=("", 13, "bold"), fg="red").pack()
    tk.Label(frm_del, text=dup_path.name, wraplength=400).pack()
    tk.Label(frm_del, image=tk_del).pack()

    # 按鈕
    btn_frame = tk.Frame(dlg)
    btn_frame.pack(side="bottom", pady=8)

    def on_delete() -> None:
        result[0] = True
        dlg.destroy()

    def on_skip() -> None:
        dlg.destroy()

    tk.Button(btn_frame, text="刪除較晚的", command=on_delete, fg="red").pack(
        side="left", padx=16
    )
    tk.Button(btn_frame, text="跳過", command=on_skip).pack(side="left", padx=16)

    dlg.protocol("WM_DELETE_WINDOW", on_skip)
    dlg.wait_window()
    return result[0]


class FileActions:
    """封裝需要使用者確認的批次檔案操作。"""

    def __init__(self, nav: ImageNavigator, store: LabelStore) -> None:
        self._nav = nav
        self._store = store

    def delete_crop(self) -> bool:
        """刪除目前的裁切圖。回傳 True 表示有刪除。"""
        if self._nav.is_empty:
            return False
        path = self._nav.current_path
        if is_raw_image(path):
            messagebox.showwarning("刪除", "無法刪除原圖。")
            return False

        confirm = messagebox.askyesno(
            "確認刪除",
            f"確定要刪除此裁切圖嗎？\n{path.name}\n\n此操作無法復原。",
        )
        if not confirm:
            return False

        key = self._nav.current_key
        self._store.delete(key)
        self._store.save()
        self._nav.remove_path(path)
        path.unlink(missing_ok=True)
        return True

    def organize_all(self) -> None:
        """去重、清理孤兒標籤、並將所有圖片搬到正確的子資料夾。"""
        dups = dedup_images(list(self._nav.all_paths))
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
                dup_key = self._store.path_to_key(dup_path)
                self._store.delete(dup_key)
                self._nav.remove_path(dup_path)
                dup_path.unlink()
                n_dedup += 1
            self._store.save()

        near_dups = dedup_near_images(list(self._nav.all_paths))
        n_near = 0
        if near_dups:
            for dup_path, orig_path in near_dups:
                if not _confirm_near_dup(dup_path, orig_path):
                    continue
                dup_key = self._store.path_to_key(dup_path)
                self._store.delete(dup_key)
                self._nav.remove_path(dup_path)
                dup_path.unlink()
                n_near += 1
            if n_near:
                self._store.save()

        orphans = self._store.purge_orphans(IMAGE_DIR)
        n_orphan = len(orphans)
        if n_orphan:
            self._store.save()

        pending: list[tuple[Path, str]] = []
        for p in list(self._nav.all_paths):
            key = self._store.path_to_key(p)
            labels = self._store.get(key)
            target = target_path_for(p.name, labels)
            if p != target:
                pending.append((p, key))

        if not pending and not n_dedup and not n_near and not n_orphan:
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
                labels = self._store.get(old_key)
                new_path = organize_single(old_path, labels)
                if new_path != old_path:
                    new_key = self._store.path_to_key(new_path)
                    self._store.rename_key(old_key, new_key)
                    self._nav.replace_path(old_path, new_path)
                    n_moved += 1
                    if CONFLICT_SUBDIR in new_path.parts:
                        n_conflict += 1

            self._store.save()

        cleanup_empty_dirs(IMAGE_DIR)

        parts = []
        if n_dedup:
            parts.append(f"已刪除 {n_dedup} 張重複圖片（SHA-256 相同）。")
        if n_near:
            parts.append(f"已刪除 {n_near} 張近似重複圖片（像素幾乎相同）。")
        if n_orphan:
            parts.append(f"已清理 {n_orphan} 筆孤兒標籤。")
        if n_moved:
            parts.append(f"已搬移 {n_moved} 張圖片。")
        if n_conflict:
            parts.append(f"其中 {n_conflict} 張因重複被移至 {CONFLICT_SUBDIR}/。")
        if not parts:
            parts.append("所有圖片已在正確位置，無重複。")
        messagebox.showinfo("全部整理", "\n".join(parts))

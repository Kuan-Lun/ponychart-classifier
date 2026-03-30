"""批次檔案操作：刪除裁切圖、清理孤兒標籤、全部整理。"""

from pathlib import Path
from tkinter import messagebox

from .constants import CONFLICT_SUBDIR, IMAGE_DIR
from .file_ops import (
    cleanup_empty_dirs,
    dedup_images,
    is_raw_image,
    organize_single,
    target_path_for,
)
from .label_store import LabelStore
from .navigator import ImageNavigator


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

    def purge_orphans(self) -> None:
        """清理 labels.json 中沒有對應檔案的孤兒標籤。"""
        orphans = self._store.purge_orphans(IMAGE_DIR)
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
        self._store.save()
        messagebox.showinfo(
            "清理孤兒標籤",
            f"已移除 {len(orphans)} 筆孤兒標籤。",
        )

    def organize_all(self) -> None:
        """去重並將所有圖片搬到正確的子資料夾。"""
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

        pending: list[tuple[Path, str]] = []
        for p in list(self._nav.all_paths):
            key = self._store.path_to_key(p)
            labels = self._store.get(key)
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
            parts.append(f"已刪除 {n_dedup} 張重複圖片。")
        if n_moved:
            parts.append(f"已搬移 {n_moved} 張圖片。")
        if n_conflict:
            parts.append(f"其中 {n_conflict} 張因重複被移至 {CONFLICT_SUBDIR}/。")
        if not parts:
            parts.append("所有圖片已在正確位置，無重複。")
        messagebox.showinfo("全部整理", "\n".join(parts))

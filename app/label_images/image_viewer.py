"""圖片顯示元件：管理 Canvas、圖片載入縮放與裁切模式。"""

import re
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from .constants import MAX_SIZE
from .crop_handler import CropHandler


class ImageViewer:
    """管理 Canvas 上的圖片顯示、縮放與裁切互動。"""

    def __init__(
        self,
        root: tk.Tk,
        *,
        on_crop_complete: Callable[[], None],
    ) -> None:
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack()
        self._canvas_image_id: int | None = None

        self.crop = CropHandler(
            self.canvas,
            on_selection_complete=on_crop_complete,
        )
        self.scale: float = 1.0
        self.current_pil_image: Image.Image | None = None
        self._tk_im: ImageTk.PhotoImage | None = None

    def load(self, path: Path) -> None:
        """載入並顯示圖片，自動縮放至 MAX_SIZE。"""
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

        self._tk_im = ImageTk.PhotoImage(display_im)
        dw, dh = display_im.size
        self.canvas.configure(width=dw, height=dh)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self._tk_im
        )

    def save_crop(self, current_path: Path) -> Path | None:
        """裁切目前圖片並儲存，回傳新檔案路徑；失敗時回傳 None。"""
        if self.current_pil_image is None:
            return None
        sel = self.crop.get_selection()
        if sel is None:
            messagebox.showwarning("裁切", "選取區域太小，請重新拖曳。")
            return None

        x1, y1, x2, y2 = sel
        w, h = self.current_pil_image.size
        orig = (
            max(0, min(int(x1 / self.scale), w)),
            max(0, min(int(y1 / self.scale), h)),
            max(0, min(int(x2 / self.scale), w)),
            max(0, min(int(y2 / self.scale), h)),
        )
        cropped = self.current_pil_image.crop(orig)

        base_stem = re.sub(r"_crop\d+$", "", current_path.stem)
        base_path = current_path.parent / f"{base_stem}{current_path.suffix}"
        save_path = self._next_crop_name(base_path)

        cropped.save(save_path)
        return save_path

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

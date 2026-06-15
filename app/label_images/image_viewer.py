"""圖片顯示元件：管理 Canvas、圖片載入縮放與裁切模式。"""

import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from .constants import DISPLAY_HEIGHT, DISPLAY_WIDTH, IMAGE_DIR
from .crop_handler import CropHandler


class ImageViewer:
    """管理 Canvas 上的圖片顯示、縮放與裁切互動。"""

    def __init__(self, root: tk.Tk) -> None:
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack()
        self._canvas_image_id: int | None = None

        self.crop = CropHandler(self.canvas)
        self.scale: float = 1.0
        self.display_size: tuple[int, int] = (0, 0)
        self.image_offset: tuple[float, float] = (0.0, 0.0)
        self.current_pil_image: Image.Image | None = None
        self._tk_im: ImageTk.PhotoImage | None = None

    def load(self, path: Path) -> None:
        """載入並顯示圖片，等比縮小至固定顯示區域內。"""
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {path}: {e}")
            return

        self.current_pil_image = im
        w, h = im.size
        self.scale = min(DISPLAY_WIDTH / max(1, w), DISPLAY_HEIGHT / max(1, h), 1.0)
        display_im = im
        if self.scale < 1.0:
            display_im = im.resize((int(w * self.scale), int(h * self.scale)))

        self._tk_im = ImageTk.PhotoImage(display_im)
        dw, dh = display_im.size
        self.display_size = (dw, dh)
        self.image_offset = ((DISPLAY_WIDTH - dw) / 2, (DISPLAY_HEIGHT - dh) / 2)
        self.canvas.configure(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            *self.image_offset, anchor="nw", image=self._tk_im
        )

    def save_crop(self, current_path: Path) -> Path | None:
        """裁切目前圖片並儲存，回傳新檔案路徑；失敗時回傳 None。"""
        if self.current_pil_image is None:
            return None

        corners = self.crop.get_corners()
        quad = tuple(c / self.scale for c in corners.as_quad())
        width_orig, height_orig = self.crop.get_size_orig()

        cropped = self.current_pil_image.transform(
            (width_orig, height_orig),
            Image.Transform.QUAD,
            quad,
            Image.Resampling.BICUBIC,
        )

        base_stem = re.sub(r"_crop\d+$", "", current_path.stem)
        save_path = self._next_crop_name(
            current_path.parent, base_stem, current_path.suffix
        )

        cropped.save(save_path)
        return save_path

    @staticmethod
    def _next_crop_name(parent: Path, base_stem: str, suffix: str) -> Path:
        """找出 base_stem 尚未使用的最小 _cropN 編號。

        裁切圖標註後會被搬到依標籤分類的子資料夾，因此已用過的編號在
        `parent` 目錄中可能已不存在；必須在整個 IMAGE_DIR 樹狀結構中
        檢查，避免編號重複造成不同裁切圖檔名衝突。
        """
        pattern = re.compile(rf"^{re.escape(base_stem)}_crop(\d+){re.escape(suffix)}$")
        used = {
            int(m.group(1))
            for p in IMAGE_DIR.rglob(f"{base_stem}_crop*{suffix}")
            if (m := pattern.match(p.name))
        }
        n = 1
        while n in used:
            n += 1
        return parent / f"{base_stem}_crop{n}{suffix}"

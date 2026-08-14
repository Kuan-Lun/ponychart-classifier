"""圖片顯示元件：管理 Canvas、圖片載入縮放與裁切模式。"""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from ponychart_classifier.image_names import extract_source_stem

from .constants import DISPLAY_HEIGHT, DISPLAY_WIDTH
from .crop_handler import CropHandler
from .file_ops import next_crop_name


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
        base_stem = extract_source_stem(current_path.name)
        if base_stem is None:
            messagebox.showerror(
                "裁切失敗",
                f"圖片檔名不符合目前格式：\n{current_path.name}",
            )
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

        save_path = next_crop_name(current_path.parent, base_stem, current_path.suffix)

        cropped.save(save_path)
        return save_path

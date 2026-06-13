"""入口點：掃描圖片並啟動標註 UI。

Usage:
    uv run --extra train python -m app.label_images
"""

import tkinter as tk
from tkinter import messagebox

from .app import LabelApp
from .constants import IMAGE_DIR
from .file_ops import scan_image_paths


def main() -> None:
    if not IMAGE_DIR.exists():
        messagebox.showerror("Error", f"找不到資料夾: {IMAGE_DIR}")
        return
    paths = scan_image_paths(IMAGE_DIR)
    paths.sort()
    root = tk.Tk()
    LabelApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()

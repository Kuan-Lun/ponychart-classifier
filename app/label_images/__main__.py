"""入口點：掃描圖片並啟動標註 UI。

Usage:
    uv run --extra train python -m app.label_images
"""

import glob
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from .app import LabelApp
from .constants import IMAGE_DIR


def main() -> None:
    if not IMAGE_DIR.exists():
        messagebox.showerror("Error", f"找不到資料夾: {IMAGE_DIR}")
        return
    paths = [Path(p) for p in glob.glob(str(IMAGE_DIR / "**" / "*"), recursive=True)]
    paths = [
        p
        for p in paths
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.is_file()
    ]
    paths.sort()
    root = tk.Tk()
    LabelApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()

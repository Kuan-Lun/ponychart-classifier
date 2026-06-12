from __future__ import annotations

import tkinter as tk
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from app.label_images.constants import CROP_ASPECT_RATIO
from app.label_images.image_viewer import ImageViewer

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)


def _make_quad_image(path: Path, size: tuple[int, int]) -> None:
    """建立一張左上/右上/左下/右下四色區塊的測試圖片。"""
    w, h = size
    half_w, half_h = w // 2, h // 2
    im = Image.new("RGB", size)
    im.paste(_RED, (0, 0, half_w, half_h))
    im.paste(_GREEN, (half_w, 0, w, half_h))
    im.paste(_BLUE, (0, half_h, half_w, h))
    im.paste(_YELLOW, (half_w, half_h, w, h))
    im.save(path)


@pytest.fixture
def viewer() -> Iterator[ImageViewer]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    try:
        yield ImageViewer(root)
    finally:
        root.destroy()


def test_save_crop_axis_aligned(tmp_path: Path, viewer: ImageViewer) -> None:
    src = tmp_path / "source.png"
    _make_quad_image(src, (1000, 600))

    viewer.load(src)
    viewer.crop.enter(viewer.display_size, viewer.scale)

    save_path = viewer.save_crop(src)
    assert save_path is not None

    cropped = Image.open(save_path).convert("RGB")
    w, h = cropped.size
    assert w / h == pytest.approx(CROP_ASPECT_RATIO, rel=0.01)

    margin = 30
    assert cropped.getpixel((margin, margin)) == _RED
    assert cropped.getpixel((w - margin, margin)) == _GREEN
    assert cropped.getpixel((margin, h - margin)) == _BLUE
    assert cropped.getpixel((w - margin, h - margin)) == _YELLOW


def test_save_crop_rotated_derotates(tmp_path: Path, viewer: ImageViewer) -> None:
    src = tmp_path / "source.png"
    _make_quad_image(src, (1000, 600))

    viewer.load(src)
    viewer.crop.enter(viewer.display_size, viewer.scale)
    viewer.crop.rotate_cw()

    save_path = viewer.save_crop(src)
    assert save_path is not None

    cropped = Image.open(save_path).convert("RGB")
    w, h = cropped.size
    assert w / h == pytest.approx(CROP_ASPECT_RATIO, rel=0.01)

    margin = 30
    assert cropped.getpixel((margin, margin)) == _RED
    assert cropped.getpixel((w - margin, margin)) == _GREEN
    assert cropped.getpixel((margin, h - margin)) == _BLUE
    assert cropped.getpixel((w - margin, h - margin)) == _YELLOW

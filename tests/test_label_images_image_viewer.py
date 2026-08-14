import tkinter as tk
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from app.label_images.constants import CROP_ASPECT_RATIO, DISPLAY_HEIGHT, DISPLAY_WIDTH
from app.label_images.file_ops import next_crop_name
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
    src = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh.png"
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


def test_load_centers_small_image(tmp_path: Path, viewer: ImageViewer) -> None:
    src = tmp_path / "small.png"
    _make_quad_image(src, (100, 100))

    viewer.load(src)

    expected_offset = ((DISPLAY_WIDTH - 100) / 2, (DISPLAY_HEIGHT - 100) / 2)
    assert viewer.image_offset == expected_offset
    assert viewer._canvas_image_id is not None
    assert tuple(viewer.canvas.coords(viewer._canvas_image_id)) == expected_offset


def test_load_full_size_image_has_zero_offset(
    tmp_path: Path, viewer: ImageViewer
) -> None:
    src = tmp_path / "large.png"
    _make_quad_image(src, (DISPLAY_WIDTH * 2, DISPLAY_HEIGHT * 2))

    viewer.load(src)

    assert viewer.image_offset == (0.0, 0.0)
    assert viewer.display_size == (DISPLAY_WIDTH, DISPLAY_HEIGHT)


def test_next_crop_name_skips_numbers_used_in_other_subdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已標註的裁切圖會被搬到標籤子資料夾，編號仍須視為已使用。"""
    monkeypatch.setattr("app.label_images.file_ops.IMAGE_DIR", tmp_path)

    organized = tmp_path / "1" / "twilight"
    organized.mkdir(parents=True)
    (organized / "pony_chart_20260101_000000_000000_abcdefgh_crop1.png").touch()

    result = next_crop_name(
        tmp_path,
        "pony_chart_20260101_000000_000000_abcdefgh",
        ".png",
    )

    assert result == (tmp_path / "pony_chart_20260101_000000_000000_abcdefgh_crop2.png")


def test_save_crop_rotated_derotates(tmp_path: Path, viewer: ImageViewer) -> None:
    src = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh.png"
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

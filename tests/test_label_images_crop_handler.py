from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Iterator

import pytest

from app.label_images.constants import (
    CROP_ASPECT_RATIO,
    CROP_INITIAL_FRACTION,
    CROP_MIN_WIDTH_ORIG,
    CROP_MOVE_STEP,
    CROP_MOVE_STEP_FAST,
    CROP_ROTATE_STEP_DEG,
    CROP_ROTATE_STEP_DEG_FAST,
    CROP_SCALE_STEP,
    CROP_SCALE_STEP_FAST,
)
from app.label_images.crop_handler import CropCorners, CropHandler


@pytest.fixture
def canvas() -> Iterator[tk.Canvas]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    try:
        yield tk.Canvas(root)
    finally:
        root.destroy()


@pytest.fixture
def handler(canvas: tk.Canvas) -> CropHandler:
    return CropHandler(canvas)


def _center(corners: CropCorners) -> tuple[float, float]:
    xs = (corners.ul.x, corners.ll.x, corners.lr.x, corners.ur.x)
    ys = (corners.ul.y, corners.ll.y, corners.lr.y, corners.ur.y)
    return sum(xs) / 4, sum(ys) / 4


def _corners_approx(a: CropCorners, b: CropCorners) -> bool:
    return all(
        getattr(a, name).x == pytest.approx(getattr(b, name).x)
        and getattr(a, name).y == pytest.approx(getattr(b, name).y)
        for name in ("ul", "ll", "lr", "ur")
    )


def _angle_of(corners: CropCorners) -> float:
    return math.atan2(corners.ur.y - corners.ul.y, corners.ur.x - corners.ul.x)


def test_enter_initial_geometry(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)

    assert handler.active is True

    width, height = handler.get_canvas_size()
    assert width / height == pytest.approx(CROP_ASPECT_RATIO)
    assert width <= 1000 * CROP_INITIAL_FRACTION + 1e-9
    assert height <= 600 * CROP_INITIAL_FRACTION + 1e-9
    assert width == pytest.approx(
        1000 * CROP_INITIAL_FRACTION
    ) or height == pytest.approx(600 * CROP_INITIAL_FRACTION)

    cx, cy = _center(handler.get_corners())
    assert cx == pytest.approx(500)
    assert cy == pytest.approx(300)


def test_enter_resets_state(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    baseline = handler.get_corners()

    handler.move_right()
    handler.rotate_cw()
    handler.scale_up()

    handler.enter((1000, 600), 1.0)
    assert _corners_approx(handler.get_corners(), baseline)


def test_move_all_directions(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    start = handler.get_corners()

    handler.move_right()
    handler.move_down()
    after = handler.get_corners()
    assert after.ul.x == pytest.approx(start.ul.x + CROP_MOVE_STEP)
    assert after.ul.y == pytest.approx(start.ul.y + CROP_MOVE_STEP)

    handler.move_left()
    handler.move_up()
    assert _corners_approx(handler.get_corners(), start)


def test_move_fast_uses_larger_step(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    before = handler.get_corners()

    handler.move_right(fast=True)
    after = handler.get_corners()
    assert after.ul.x == pytest.approx(before.ul.x + CROP_MOVE_STEP_FAST)
    assert after.ul.y == pytest.approx(before.ul.y)


def test_move_clamped_at_canvas_edge(handler: CropHandler) -> None:
    cw, ch = 1000, 600
    handler.enter((cw, ch), 1.0)

    for _ in range(100):
        handler.move_right()

    corners = handler.get_corners()
    assert corners.all_in_bounds(cw, ch)
    assert max(corners.ul.x, corners.ll.x, corners.lr.x, corners.ur.x) == pytest.approx(
        cw
    )

    before = handler.get_corners()
    handler.move_right()
    assert _corners_approx(handler.get_corners(), before)


def test_rotate_changes_angle_about_center(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    before = handler.get_corners()
    cx, cy = _center(before)

    handler.rotate_cw()
    after = handler.get_corners()

    assert after.ul.y != pytest.approx(after.ur.y)
    acx, acy = _center(after)
    assert acx == pytest.approx(cx)
    assert acy == pytest.approx(cy)

    handler.rotate_ccw()
    assert _corners_approx(handler.get_corners(), before)


def test_rotate_fast_uses_larger_step(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    handler.rotate_cw()
    normal_angle = _angle_of(handler.get_corners())

    handler.enter((1000, 600), 1.0)
    handler.rotate_cw(fast=True)
    fast_angle = _angle_of(handler.get_corners())

    assert normal_angle == pytest.approx(math.radians(CROP_ROTATE_STEP_DEG))
    assert fast_angle == pytest.approx(math.radians(CROP_ROTATE_STEP_DEG_FAST))


def test_rotate_clamped_when_corner_would_exit_bounds(handler: CropHandler) -> None:
    cw, ch = 1000, 600
    handler.enter((cw, ch), 1.0)

    for _ in range(200):
        handler.rotate_cw()

    settled = handler.get_corners()
    assert settled.all_in_bounds(cw, ch)
    assert settled.ul.y != pytest.approx(settled.ur.y)

    handler.rotate_cw()
    assert _corners_approx(handler.get_corners(), settled)


def test_scale_up_and_down_preserve_ratio(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    width0, height0 = handler.get_canvas_size()

    handler.scale_up()
    width1, height1 = handler.get_canvas_size()
    assert width1 == pytest.approx(width0 * CROP_SCALE_STEP)
    assert width1 / height1 == pytest.approx(CROP_ASPECT_RATIO)

    handler.scale_down()
    width2, height2 = handler.get_canvas_size()
    assert width2 == pytest.approx(width0)
    assert height2 == pytest.approx(height0)


def test_scale_fast_uses_larger_factor(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    width0, _ = handler.get_canvas_size()

    handler.scale_up(fast=True)
    width1, _ = handler.get_canvas_size()
    assert width1 == pytest.approx(width0 * CROP_SCALE_STEP_FAST)

    handler.scale_down(fast=True)
    width2, _ = handler.get_canvas_size()
    assert width2 == pytest.approx(width0)


def test_scale_down_respects_min_width_floor(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    floor = CROP_MIN_WIDTH_ORIG * 1.0

    for _ in range(100):
        handler.scale_down()

    width, _ = handler.get_canvas_size()
    assert width >= floor - 1e-9

    before = handler.get_canvas_size()
    handler.scale_down()
    after = handler.get_canvas_size()
    assert after == pytest.approx(before)


def test_min_width_floor_degrades_for_small_images(handler: CropHandler) -> None:
    cw, ch = 300, 166
    handler.enter((cw, ch), 1.0)

    width, _ = handler.get_canvas_size()
    expected_floor = min(cw, ch * CROP_ASPECT_RATIO)
    assert expected_floor < CROP_MIN_WIDTH_ORIG
    assert width == pytest.approx(expected_floor)

    handler.scale_down()
    after_width, _ = handler.get_canvas_size()
    assert after_width == pytest.approx(width)


def test_get_corners_ordering_and_quad_shape(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    corners = handler.get_corners()

    assert corners.ul.x == pytest.approx(corners.ll.x)
    assert corners.lr.x == pytest.approx(corners.ur.x)
    assert corners.ul.y == pytest.approx(corners.ur.y)
    assert corners.ll.y == pytest.approx(corners.lr.y)
    assert corners.ul.x < corners.lr.x
    assert corners.ul.y < corners.lr.y

    quad = corners.as_quad()
    assert len(quad) == 8
    assert all(isinstance(v, float) for v in quad)


def test_exit_clears_polygon(handler: CropHandler, canvas: tk.Canvas) -> None:
    handler.enter((1000, 600), 1.0)
    assert canvas.find_all() != ()

    handler.exit()
    assert canvas.find_all() == ()
    assert handler.active is False


def test_get_size_orig_matches_scale(handler: CropHandler) -> None:
    scale = 0.5
    handler.enter((1000, 600), scale)

    canvas_w, canvas_h = handler.get_canvas_size()
    width_orig, height_orig = handler.get_size_orig()
    assert width_orig == round(canvas_w / scale)
    assert height_orig == round(canvas_h / scale)


def test_status_overlay_drawn_and_cleared(
    handler: CropHandler, canvas: tk.Canvas
) -> None:
    handler.enter((1000, 600), 1.0)
    # 裁切框 + 狀態文字
    assert len(canvas.find_all()) == 2

    handler.exit()
    assert canvas.find_all() == ()


def test_enter_with_offset_shifts_drawn_polygon(
    handler: CropHandler, canvas: tk.Canvas
) -> None:
    offset = (50.0, 20.0)
    handler.enter((1000, 600), 1.0, offset)

    corners = handler.get_corners().translated(*offset)
    assert handler._polygon_id is not None
    drawn = canvas.coords(handler._polygon_id)

    assert drawn == pytest.approx(list(corners.as_quad()))


def test_enter_with_offset_shifts_status_overlay(
    handler: CropHandler, canvas: tk.Canvas
) -> None:
    offset = (50.0, 20.0)
    handler.enter((1000, 600), 1.0, offset)

    assert handler._status_text_id is not None
    x, y = canvas.coords(handler._status_text_id)
    assert (x, y) == pytest.approx((8 + offset[0], 8 + offset[1]))


def test_toggle_orientation_swaps_dimensions(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    width0, height0 = handler.get_canvas_size()

    handler.toggle_orientation()
    width1, height1 = handler.get_canvas_size()

    assert width1 == pytest.approx(height0)
    assert height1 == pytest.approx(width0)
    assert width1 / height1 == pytest.approx(1.0 / CROP_ASPECT_RATIO)


def test_toggle_orientation_round_trip(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    baseline = handler.get_corners()

    handler.toggle_orientation()
    handler.toggle_orientation()

    assert _corners_approx(handler.get_corners(), baseline)


def test_toggle_orientation_preserves_center(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    cx, cy = _center(handler.get_corners())

    handler.toggle_orientation()

    acx, acy = _center(handler.get_corners())
    assert acx == pytest.approx(cx)
    assert acy == pytest.approx(cy)


def test_toggle_orientation_shrinks_and_repositions_when_oversized(
    handler: CropHandler,
) -> None:
    # 畫面比例較扁，橫向裁切框可放入，但交換寬高後的直向框放不進畫布高度，
    # 需先縮小到貼齊上下邊緣，再平移回畫布範圍內。
    cw, ch = 1000, 500
    handler.enter((cw, ch), 1.0)
    _, height0 = handler.get_canvas_size()

    handler.toggle_orientation()
    width1, height1 = handler.get_canvas_size()

    assert handler.get_corners().all_in_bounds(cw, ch)
    assert width1 / height1 == pytest.approx(1.0 / CROP_ASPECT_RATIO)
    assert width1 < height0
    assert height1 == pytest.approx(ch)


def test_toggle_orientation_updates_size_orig(handler: CropHandler) -> None:
    handler.enter((1000, 600), 1.0)
    width_orig0, height_orig0 = handler.get_size_orig()

    handler.toggle_orientation()
    width_orig1, height_orig1 = handler.get_size_orig()

    assert width_orig1 == height_orig0
    assert height_orig1 == width_orig0

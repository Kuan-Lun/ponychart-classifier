"""裁切模式的狀態管理與畫布繪製：鍵盤驅動的可旋轉、比例鎖定裁切框。"""

import dataclasses
import math
import tkinter as tk

from .constants import (
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

# 浮點誤差容許值（畫布像素），用於邊界判斷
_BOUNDS_EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class Point:
    """畫布座標系中的一個 2D 點。"""

    x: float
    y: float


@dataclasses.dataclass(frozen=True)
class CropCorners:
    """裁切框的 4 個角點，依 PIL QUAD 所需順序排列。

    順序為「裁切框自身座標系」的 upper-left, lower-left, lower-right,
    upper-right；當旋轉角度為 0 時，分別對應畫面的左上、左下、右下、右上。
    """

    ul: Point
    ll: Point
    lr: Point
    ur: Point

    def as_quad(self) -> tuple[float, float, float, float, float, float, float, float]:
        """展開為 PIL ``Image.transform(..., QUAD, data)`` 所需的 8-tuple。"""
        return (
            self.ul.x,
            self.ul.y,
            self.ll.x,
            self.ll.y,
            self.lr.x,
            self.lr.y,
            self.ur.x,
            self.ur.y,
        )

    def all_in_bounds(self, max_x: float, max_y: float) -> bool:
        """判斷 4 個角點是否都落在 ``[0, max_x] x [0, max_y]`` 範圍內。

        容許 :data:`_BOUNDS_EPS` 的浮點誤差，避免角點剛好落在邊界時
        因捨入誤差被誤判為超界。
        """
        return all(
            -_BOUNDS_EPS <= p.x <= max_x + _BOUNDS_EPS
            and -_BOUNDS_EPS <= p.y <= max_y + _BOUNDS_EPS
            for p in (self.ul, self.ll, self.lr, self.ur)
        )


class CropHandler:
    """裁切模式的狀態管理與畫布繪製。

    裁切框長寬比鎖定為 :data:`CROP_ASPECT_RATIO`（或其倒數，視目前方向），
    使用者透過鍵盤調整位置（WASD）、旋轉角度（QE）、大小（RF）與方向
    （橫向/直向，:meth:`toggle_orientation`），確認後以
    :class:`CropCorners` 搭配 ``Image.transform(..., QUAD, ...)`` 一步
    完成裁切與去旋轉。
    """

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self.active: bool = False
        self._center: Point = Point(0.0, 0.0)
        self._width: float = 0.0
        self._angle_rad: float = 0.0
        self._portrait: bool = False
        self._scale: float = 1.0
        self._polygon_id: int | None = None
        self._status_text_id: int | None = None
        self._min_width_canvas: float = 0.0
        self._canvas_size: tuple[float, float] = (0.0, 0.0)

    def enter(self, canvas_size: tuple[int, int], scale: float) -> None:
        """進入裁切模式：以畫面置中、橫向、比例鎖定的預設矩形重置狀態。"""
        self.active = True
        cw, ch = float(canvas_size[0]), float(canvas_size[1])
        self._canvas_size = (cw, ch)
        self._scale = scale
        self._portrait = False

        width_from_cw = cw * CROP_INITIAL_FRACTION
        width_from_ch = ch * CROP_INITIAL_FRACTION * CROP_ASPECT_RATIO
        self._width = min(width_from_cw, width_from_ch)

        self._center = Point(cw / 2.0, ch / 2.0)
        self._angle_rad = 0.0

        self._min_width_canvas = self._min_width_for(self._portrait, self._angle_rad)
        self._width = max(self._width, self._min_width_canvas)

        self._redraw()

    def exit(self) -> None:
        """離開裁切模式並清除畫面上的裁切框。"""
        self.active = False
        self._clear_polygon()
        self._clear_status_overlay()

    def get_corners(self) -> CropCorners:
        """回傳目前裁切框的 4 個角點（畫布座標）。"""
        return self._compute_corners(
            self._center, self._width, self._angle_rad, self._portrait
        )

    def get_canvas_size(self) -> tuple[float, float]:
        """回傳目前裁切框的 ``(width, height)``（畫布像素）。"""
        return (self._width, self._height(self._width, self._portrait))

    def get_size_orig(self) -> tuple[int, int]:
        """回傳目前裁切框對應到原圖的 ``(width, height)``（像素）。"""
        canvas_w, canvas_h = self.get_canvas_size()
        return (
            max(1, round(canvas_w / self._scale)),
            max(1, round(canvas_h / self._scale)),
        )

    # ── 鍵盤操作：移動 ──────────────────────────────────────────

    def move_up(self, fast: bool = False) -> None:
        self._move(0.0, -(CROP_MOVE_STEP_FAST if fast else CROP_MOVE_STEP))

    def move_down(self, fast: bool = False) -> None:
        self._move(0.0, CROP_MOVE_STEP_FAST if fast else CROP_MOVE_STEP)

    def move_left(self, fast: bool = False) -> None:
        self._move(-(CROP_MOVE_STEP_FAST if fast else CROP_MOVE_STEP), 0.0)

    def move_right(self, fast: bool = False) -> None:
        self._move(CROP_MOVE_STEP_FAST if fast else CROP_MOVE_STEP, 0.0)

    # ── 鍵盤操作：旋轉 ──────────────────────────────────────────

    def rotate_ccw(self, fast: bool = False) -> None:
        step = CROP_ROTATE_STEP_DEG_FAST if fast else CROP_ROTATE_STEP_DEG
        self._rotate(-math.radians(step))

    def rotate_cw(self, fast: bool = False) -> None:
        step = CROP_ROTATE_STEP_DEG_FAST if fast else CROP_ROTATE_STEP_DEG
        self._rotate(math.radians(step))

    # ── 鍵盤操作：縮放 ──────────────────────────────────────────

    def scale_up(self, fast: bool = False) -> None:
        self._scale_by(CROP_SCALE_STEP_FAST if fast else CROP_SCALE_STEP)

    def scale_down(self, fast: bool = False) -> None:
        step = CROP_SCALE_STEP_FAST if fast else CROP_SCALE_STEP
        self._scale_by(1.0 / step)

    # ── 鍵盤操作：方向 ──────────────────────────────────────────

    def toggle_orientation(self) -> None:
        """切換裁切框為橫向/直向（寬高互換）。

        直接以目標方向計算結果，不經過旋轉的中間角度，因此不受旋轉
        dead zone（AABB 在 0°~90° 之間先升到對角線峰值再降回交換後的
        寬高）影響。若交換後在目前角度下放不進畫布，會先縮小到該角度
        下置中可容納的最大尺寸，再將裁切框平移回畫布範圍內。
        """
        candidate_portrait = not self._portrait
        angle = self._angle_rad

        max_width = self._max_width_canvas(candidate_portrait, angle)
        min_width = self._min_width_for(candidate_portrait, angle)
        ideal_width = self._height(self._width, self._portrait)
        candidate_width = min(max(ideal_width, min_width), max_width)
        candidate_height = self._height(candidate_width, candidate_portrait)

        cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
        half_w = (candidate_width * cos_a + candidate_height * sin_a) / 2.0
        half_h = (candidate_width * sin_a + candidate_height * cos_a) / 2.0

        cw, ch = self._canvas_size
        self._center = Point(
            min(max(self._center.x, half_w), cw - half_w),
            min(max(self._center.y, half_h), ch - half_h),
        )
        self._width = candidate_width
        self._portrait = candidate_portrait
        self._min_width_canvas = min_width
        self._redraw()

    # ── 內部實作 ────────────────────────────────────────────────

    @staticmethod
    def _height(width: float, portrait: bool) -> float:
        return width * CROP_ASPECT_RATIO if portrait else width / CROP_ASPECT_RATIO

    def _max_width_canvas(self, portrait: bool, angle_rad: float) -> float:
        """計算指定方向與角度下，置中於畫布時可容納的最大寬度。"""
        cw, ch = self._canvas_size
        k = CROP_ASPECT_RATIO if portrait else 1.0 / CROP_ASPECT_RATIO
        cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
        return min(cw / (cos_a + k * sin_a), ch / (sin_a + k * cos_a))

    def _min_width_for(self, portrait: bool, angle_rad: float) -> float:
        """計算指定方向與角度下，裁切框寬度（畫布像素）的下限。

        下限對應原圖長邊 >= :data:`CROP_MIN_WIDTH_ORIG`，並以
        :meth:`_max_width_canvas` 封頂，避免裁切框超出畫布範圍。
        """
        floor_orig = (
            CROP_MIN_WIDTH_ORIG / CROP_ASPECT_RATIO if portrait else CROP_MIN_WIDTH_ORIG
        )
        return min(
            floor_orig * self._scale, self._max_width_canvas(portrait, angle_rad)
        )

    def _move(self, dx: float, dy: float) -> None:
        candidate = Point(self._center.x + dx, self._center.y + dy)
        corners = self._compute_corners(
            candidate, self._width, self._angle_rad, self._portrait
        )
        if corners.all_in_bounds(*self._canvas_size):
            self._center = candidate
            self._redraw()

    def _rotate(self, delta_rad: float) -> None:
        candidate_angle = self._angle_rad + delta_rad
        corners = self._compute_corners(
            self._center, self._width, candidate_angle, self._portrait
        )
        if corners.all_in_bounds(*self._canvas_size):
            self._angle_rad = candidate_angle
            self._redraw()

    def _scale_by(self, factor: float) -> None:
        candidate_width = self._width * factor
        if candidate_width < self._min_width_canvas:
            return
        corners = self._compute_corners(
            self._center, candidate_width, self._angle_rad, self._portrait
        )
        if corners.all_in_bounds(*self._canvas_size):
            self._width = candidate_width
            self._redraw()

    @staticmethod
    def _compute_corners(
        center: Point, width: float, angle_rad: float, portrait: bool
    ) -> CropCorners:
        height = CropHandler._height(width, portrait)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        # width 軸方向 (ux,uy)；height 軸方向 (vx,vy)，與 width 軸垂直
        ux, uy = cos_a, sin_a
        vx, vy = -sin_a, cos_a
        hw, hh = width / 2.0, height / 2.0

        def corner(sx: float, sy: float) -> Point:
            return Point(
                center.x + sx * hw * ux + sy * hh * vx,
                center.y + sx * hw * uy + sy * hh * vy,
            )

        return CropCorners(
            ul=corner(-1.0, -1.0),
            ll=corner(-1.0, 1.0),
            lr=corner(1.0, 1.0),
            ur=corner(1.0, -1.0),
        )

    def _redraw(self) -> None:
        self._clear_polygon()
        corners = self.get_corners()
        self._polygon_id = self._canvas.create_polygon(
            *corners.as_quad(),
            outline="red",
            fill="",
            width=2,
            dash=(4, 4),
        )
        self._draw_status_overlay()

    def _clear_polygon(self) -> None:
        if self._polygon_id is not None:
            self._canvas.delete(self._polygon_id)
            self._polygon_id = None

    def _draw_status_overlay(self) -> None:
        self._clear_status_overlay()
        width_orig, height_orig = self.get_size_orig()
        text = (
            f"裁切尺寸: {width_orig} x {height_orig} px\n"
            f"移動: {CROP_MOVE_STEP}px (Shift {CROP_MOVE_STEP_FAST}px)\n"
            f"旋轉: {CROP_ROTATE_STEP_DEG:.1f}° (Shift {CROP_ROTATE_STEP_DEG_FAST:.1f}°)\n"
            f"縮放: x{CROP_SCALE_STEP:.2f} (Shift x{CROP_SCALE_STEP_FAST:.2f})\n"
            f"方向: 2 (橫向/直向切換)"
        )
        self._status_text_id = self._canvas.create_text(
            8,
            8,
            anchor="nw",
            text=text,
            fill="red",
            font=("Consolas", 11),
            justify="left",
        )

    def _clear_status_overlay(self) -> None:
        if self._status_text_id is not None:
            self._canvas.delete(self._status_text_id)
            self._status_text_id = None

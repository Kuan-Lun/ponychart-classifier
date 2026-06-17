"""旋轉裁切框的純幾何運算：與畫面/輸入裝置無關，供互動裁切與自動化腳本共用。"""

import dataclasses
import math

# 浮點誤差容許值，用於邊界判斷
_BOUNDS_EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class Point:
    """平面座標系中的一個 2D 點。"""

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

    def translated(self, dx: float, dy: float) -> "CropCorners":
        """回傳所有角點平移 ``(dx, dy)`` 後的新 :class:`CropCorners`。"""
        return CropCorners(
            ul=Point(self.ul.x + dx, self.ul.y + dy),
            ll=Point(self.ll.x + dx, self.ll.y + dy),
            lr=Point(self.lr.x + dx, self.lr.y + dy),
            ur=Point(self.ur.x + dx, self.ur.y + dy),
        )


def height_for(width: float, aspect_ratio: float, portrait: bool) -> float:
    """依長寬比與方向，計算裁切框高度。"""
    return width * aspect_ratio if portrait else width / aspect_ratio


def max_width(
    bound_w: float,
    bound_h: float,
    aspect_ratio: float,
    portrait: bool,
    angle_rad: float,
) -> float:
    """計算指定方向與角度下，置中於 ``bound_w x bound_h`` 範圍內可容納的最大寬度。"""
    k = aspect_ratio if portrait else 1.0 / aspect_ratio
    cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    return min(bound_w / (cos_a + k * sin_a), bound_h / (sin_a + k * cos_a))


def min_width(
    bound_w: float,
    bound_h: float,
    aspect_ratio: float,
    min_width_floor: float,
    portrait: bool,
    angle_rad: float,
) -> float:
    """計算指定方向與角度下，裁切框寬度的下限。

    下限對應長邊 >= ``min_width_floor``，並以 :func:`max_width` 封頂，避免裁切框
    超出 ``bound_w x bound_h`` 範圍。
    """
    floor = min_width_floor / aspect_ratio if portrait else min_width_floor
    return min(floor, max_width(bound_w, bound_h, aspect_ratio, portrait, angle_rad))


def half_extents(width: float, height: float, angle_rad: float) -> tuple[float, float]:
    """計算旋轉後裁切框的 axis-aligned bounding box 半寬、半高。"""
    cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    half_w = (width * cos_a + height * sin_a) / 2.0
    half_h = (width * sin_a + height * cos_a) / 2.0
    return half_w, half_h


def compute_corners(
    center: Point, width: float, aspect_ratio: float, angle_rad: float, portrait: bool
) -> CropCorners:
    """計算旋轉裁切框的 4 個角點。"""
    height = height_for(width, aspect_ratio, portrait)
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

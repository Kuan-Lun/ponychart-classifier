"""Model specification constants shared by inference and training."""

import dataclasses
from enum import StrEnum


class PonyClass(StrEnum):
    """Stable machine identifiers for PonyChart classes."""

    TWILIGHT_SPARKLE = "twilight_sparkle"
    RARITY = "rarity"
    FLUTTERSHY = "fluttershy"
    RAINBOW_DASH = "rainbow_dash"
    PINKIE_PIE = "pinkie_pie"
    APPLEJACK = "applejack"


PONY_CLASSES = (
    PonyClass.TWILIGHT_SPARKLE,
    PonyClass.RARITY,
    PonyClass.FLUTTERSHY,
    PonyClass.RAINBOW_DASH,
    PonyClass.PINKIE_PIE,
    PonyClass.APPLEJACK,
)

DISPLAY_NAME_BY_CLASS: dict[PonyClass, str] = {
    PonyClass.TWILIGHT_SPARKLE: "Twilight Sparkle",
    PonyClass.RARITY: "Rarity",
    PonyClass.FLUTTERSHY: "Fluttershy",
    PonyClass.RAINBOW_DASH: "Rainbow Dash",
    PonyClass.PINKIE_PIE: "Pinkie Pie",
    PonyClass.APPLEJACK: "Applejack",
}

DISPLAY_NAME_TO_CLASS = {
    name: pony_class for pony_class, name in DISPLAY_NAME_BY_CLASS.items()
}

NUM_CLASSES = len(PONY_CLASSES)
CLASS_NAMES = [DISPLAY_NAME_BY_CLASS[pony_class] for pony_class in PONY_CLASSES]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclasses.dataclass(frozen=True)
class ImageSize:
    """Image dimensions in ``(height, width)`` order.

    Follows the PyTorch / torchvision convention.  Use :meth:`hw` when a
    plain ``(height, width)`` tuple is needed (e.g. ``torch.randn``,
    ``torchvision.transforms``), and :meth:`wh` at the PIL / OpenCV
    boundary where ``(width, height)`` is expected.
    """

    height: int
    width: int

    def hw(self) -> tuple[int, int]:
        """Return ``(height, width)`` — for PyTorch / torchvision."""
        return (self.height, self.width)

    def wh(self) -> tuple[int, int]:
        """Return ``(width, height)`` — for PIL / OpenCV."""
        return (self.width, self.height)


INPUT_SIZE = ImageSize(277, 502)


def parse_class_key(key: str) -> PonyClass:
    """Parse either a stable class id or a display name into :class:`PonyClass`."""
    try:
        return PonyClass(key)
    except ValueError:
        try:
            return DISPLAY_NAME_TO_CLASS[key]
        except KeyError as exc:
            raise KeyError(f"Unknown PonyChart class key: {key}") from exc

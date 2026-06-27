"""Section Protocol 與所有具體 section class。"""

import tkinter as tk
from typing import Protocol

from .changes import ChangesSection
from .distribution_test import DistributionTestSection
from .hyperparams import HyperparamsSection
from .image_counts import ImageCountsSection
from .model_arch import ModelArchSection
from .notice import NoticeSection
from .split_counts import SplitCountsSection
from .val_f1 import ValF1Section


class Section(Protocol):
    """視窗中的一個獨立渲染單位。"""

    def render(self, parent: tk.Widget) -> None: ...


__all__ = [
    "ChangesSection",
    "DistributionTestSection",
    "HyperparamsSection",
    "ImageCountsSection",
    "ModelArchSection",
    "NoticeSection",
    "Section",
    "SplitCountsSection",
    "ValF1Section",
]

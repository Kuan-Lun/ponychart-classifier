"""Per-backbone default input sizes for CPU inference benchmarking."""

from __future__ import annotations

from ponychart_classifier.model_spec import INPUT_SIZE, ImageSize

# 每個 backbone 預設使用的推論輸入解析度。
# B0 用 production 設定（INPUT_SIZE），其他用各自原生解析度。
BACKBONE_INPUT_SIZE: dict[str, ImageSize] = {
    "mobilenet_v3_small": ImageSize(224, 224),
    "mobilenet_v3_large": ImageSize(224, 224),
    "efficientnet_b0": INPUT_SIZE,
    "efficientnet_b2": ImageSize(260, 260),
    "efficientnet_b4": ImageSize(380, 380),
}

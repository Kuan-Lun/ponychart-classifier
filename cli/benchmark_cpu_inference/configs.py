"""Per-backbone default input sizes for CPU inference benchmarking."""

from __future__ import annotations

from ponychart_classifier.model_spec import ImageSize

# 每個 backbone 預設使用的推論輸入解析度。
# 跟 cli/compare_backbones 的 BACKBONE_CONFIGS 一致：
# B0 用 production 設定 320，其他用各自原生 (B4=380)。
BACKBONE_INPUT_SIZE: dict[str, ImageSize] = {
    "mobilenet_v3_small": ImageSize(224, 224),
    "mobilenet_v3_large": ImageSize(224, 224),
    "efficientnet_b0": ImageSize(320, 320),
    "efficientnet_b2": ImageSize(260, 260),
    "efficientnet_b4": ImageSize(380, 380),
}

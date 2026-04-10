"""Per-backbone default input sizes for CPU inference benchmarking."""

from __future__ import annotations

# 每個 backbone 預設使用的推論輸入解析度。
# 跟 cli/compare_backbones 的 BACKBONE_CONFIGS 一致：
# B0 用 production 設定 320，其他用各自原生 (B4=380)。
BACKBONE_INPUT_SIZE: dict[str, int] = {
    "mobilenet_v3_small": 224,
    "mobilenet_v3_large": 224,
    "efficientnet_b0": 320,
    "efficientnet_b2": 260,
    "efficientnet_b4": 380,
}

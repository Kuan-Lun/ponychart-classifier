#!/usr/bin/env bash
# 將 rawimage/ 封存成 zip，排除 macOS 專屬檔案
# （.DS_Store、__MACOSX、AppleDouble 的 ._* 檔）。
#
# 使用方式：
#   ./scripts/archive-rawimage.sh [輸出檔名]
#
# 未指定輸出檔名時，預設為 rawimage-<timestamp>.zip，存放於 repo 根目錄。
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT="${1:-rawimage-$(date +%Y%m%d-%H%M%S).zip}"

zip -r -X -0 -q "$OUTPUT" rawimage \
  -x "*.DS_Store" \
  -x "*__MACOSX*" \
  -x "*/._*"

echo "已建立: $OUTPUT"

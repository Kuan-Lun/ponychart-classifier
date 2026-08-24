#!/usr/bin/env bash
# 打包在 Colab 上從零訓練所需的最少檔案：
#   - pyproject.toml、README.md（editable install 需要）
#   - src/ponychart_classifier/ 套件原始碼
#   - scripts/train.py
#   - rawimage/labels.json 與其引用的所有圖片
#
# 不含 checkpoint.pt（--from-scratch 用不到）、analysis_cache.json、
# app/、cli/、tests/、artifacts/。
#
# Colab 端解壓後執行：
#   python -m pip install -e ".[train]"
#   python scripts/train.py --from-scratch
#
# 使用方式：
#   ./scripts/package-train-bundle.sh [輸出檔名]
#
# 未指定輸出檔名時，預設為 train-bundle-<timestamp>.zip，存放於 repo 根目錄。
# 圖片為 PNG，壓縮無益，一律以 store 模式（-0）封存。
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT="${1:-train-bundle-$(date +%Y%m%d-%H%M%S).zip}"

FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT

uv run python - "$FILELIST" <<'EOF'
"""列出訓練 bundle 應包含的檔案（相對於 repo 根目錄）。"""

import json
import os
import sys

filelist_path = sys.argv[1]
files: list[str] = [
    "pyproject.toml",
    "README.md",
    "scripts/train.py",
    "rawimage/labels.json",
]

for dirpath, dirnames, filenames in os.walk("src/ponychart_classifier"):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for name in filenames:
        if name.startswith("."):
            continue
        files.append(os.path.join(dirpath, name))

with open("rawimage/labels.json", encoding="utf-8") as f:
    labels = json.load(f)

missing = 0
for key in labels:
    # 與 load_samples 相同的解析邏輯：先試 key 路徑，再以檔名於根目錄尋找
    path = os.path.join("rawimage", key)
    if not os.path.isfile(path):
        path = os.path.join("rawimage", key.split("/")[-1])
        if not os.path.isfile(path):
            missing += 1
            continue
    files.append(path)

with open(filelist_path, "w", encoding="utf-8") as f:
    f.write("\n".join(files) + "\n")

n_images = len(files) - 4 - sum(1 for p in files if p.startswith("src/"))
print(f"檔案清單：{len(files)} 個檔案（圖片 {n_images} 張，缺漏 {missing} 筆）")
EOF

zip -X -0 -q "$OUTPUT" -@ < "$FILELIST"

echo "已建立: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"

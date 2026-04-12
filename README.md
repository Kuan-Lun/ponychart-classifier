# PonyChart Classifier

PonyChart 角色辨識模型，用於自動辨識 HentaiVerse 戰鬥中出現的 PonyChart 圖片中的角色。

## 目錄結構

```text
ponychart-classifier/
├── app/
│   └── label_images/                  # 圖片標註工具 (Tkinter GUI)
│       ├── __main__.py                # 進入點
│       ├── app.py                     # LabelApp 主應用
│       ├── analysis.py                # 模型分析 (背景推論)
│       ├── constants.py               # GUI 常數
│       ├── crop_handler.py            # 裁切處理
│       ├── data_viewer/               # 資料概況 / 模型資訊 / 分析結果視窗
│       │   ├── __init__.py            # public viewers re-export
│       │   ├── viewers.py             # _BaseViewer 及三個 public viewer
│       │   ├── extractors.py          # checkpoint 載入與資料萃取
│       │   ├── stats.py               # 原圖樣本聚合統計
│       │   ├── widgets.py             # Tk 排版 helper
│       │   └── sections/              # 可組合的 Section（Protocol + 實作）
│       │       ├── __init__.py
│       │       ├── changes.py         # 資料變更摘要
│       │       ├── distribution_test.py # 分布檢定
│       │       ├── hyperparams.py     # 超參數顯示
│       │       ├── image_counts.py    # 圖片數量統計
│       │       ├── model_arch.py      # 模型架構資訊
│       │       ├── notice.py          # 通知訊息
│       │       ├── split_counts.py    # 資料分割統計
│       │       └── val_f1.py          # 驗證 F1 分數
│       ├── file_actions.py            # 批次檔案操作
│       ├── file_ops.py                # 檔案操作
│       ├── filter_builder.py          # 篩選條件建構
│       ├── filter_panel.py            # 篩選面板 UI
│       ├── image_viewer.py            # 圖片顯示元件
│       ├── label_store.py             # 標籤儲存
│       └── navigator.py              # 圖片導覽
├── src/
│   └── ponychart_classifier/          # PyPI 套件
│       ├── __init__.py                # 公開 API (predict, update, preload, get_thresholds)
│       ├── model_spec.py              # 推論常數 + PredictionResult / ClassThresholds
│       ├── inference.py               # PonyChartClassifier (ONNX 推論)
│       ├── _http.py                   # SSL-aware URL opener
│       ├── py.typed                   # PEP 561 type marker
│       ├── model.onnx                 # 隨套件發佈的 ONNX 模型
│       ├── thresholds.json            # 隨套件發佈的分類閾值
│       ├── stats/                     # 多項分布適合度檢定
│       │   ├── __init__.py            # 公開 API re-export
│       │   ├── asymptotic.py          # 漸近檢定 (chi-square / G-test)
│       │   ├── exact.py               # 精確檢定 (全排列枚舉)
│       │   ├── gof.py                 # 統一入口 goodness_of_fit_test()
│       │   ├── result.py              # GoFTestResult 資料類別
│       │   └── statistics.py          # 檢定統計量 (Pearson / G / logpmf)
│       └── training/                  # 訓練函式庫
│           ├── __init__.py            # Re-export 所有 symbol
│           ├── constants.py           # 常數與訓練超參數 (single source of truth)
│           ├── device.py              # 裝置偵測
│           ├── dataset.py             # 資料載入、Dataset、transforms
│           ├── model.py               # Backbone registry + build_model()
│           ├── training.py            # 訓練迴圈、evaluate、threshold 優化
│           ├── checkpoint.py          # Checkpoint val_f1 重新計算與更新
│           ├── sampling.py            # 樣本載入與平衡
│           ├── splitting.py           # Hash-based group splitting
│           ├── log_helpers.py         # 日誌輔助
│           ├── script_utils.py        # 腳本共用工具
│           ├── experiment_io.py       # 跨機器實驗結果 I/O
│           └── export.py              # ONNX 匯出
├── cli/                               # CLI 工具 (python -m cli.<name>)
│   ├── experiment.py                  # ExperimentCLI 抽象基底 (Template Method)
│   ├── training_runner.py             # 可組合的訓練管線
│   ├── compare_resolution/            # 輸入解析度比較
│   ├── compare_backbones/             # Backbone 架構比較
│   ├── compare_aspect_ratio/          # 長寬比 (正方形 vs 長方形) 比較
│   ├── search_batch_lr/               # Batch size / LR 超參數搜尋
│   ├── benchmark_cpu_inference/       # CPU 推論延遲 benchmark
│   └── analyze_augmentations/         # 資料增強 ablation study
├── tests/                             # 測試套件 (pytest)
│   ├── test_stats.py                  # stats 模組整合測試
│   └── stats/                         # stats 模組單元測試
│       ├── test_compositions.py
│       ├── test_convergence.py
│       ├── test_gof_advanced.py
│       ├── test_gof_basic.py
│       └── test_helpers.py
├── scripts/                           # 開發用腳本 (不隨套件發佈)
│   ├── train.py                       # 模型訓練腳本
│   ├── rebuild-env.sh                 # 重建 .venv 與快取
│   ├── compare_crops.py               # 裁切圖片效果分析
│   ├── compare_pos_weight.py          # pos_weight 效果比較
│   ├── compare_resume_scratch.py      # Resume vs from-scratch 分析
│   ├── evaluate_holdout.py            # Holdout 評估
│   ├── fit_cards.py                   # PonyChart 角色分布模型擬合
│   ├── learning_curve.py              # Learning curve 分析 + power-law 外推
│   └── profile_dataloader.py          # DataLoader 效能分析
├── rawimage/                          # 訓練用原始圖片 (PNG)
│   ├── labels.json                    # 標註資料 {"1/twilight/filename.png": [1,3]}
│   └── checkpoint.pt                  # PyTorch checkpoint (resume 訓練用)
├── results/                           # CLI 實驗結果 JSON 輸出目錄
├── mypy.ini                           # MyPy strict 設定
├── pyproject.toml
├── uv.lock
└── README.md
```

## 標籤對照

| 編號 | 角色 |
| ---- | ---- |
| 1 | Twilight Sparkle |
| 2 | Rarity |
| 3 | Fluttershy |
| 4 | Rainbow Dash |
| 5 | Pinkie Pie |
| 6 | Applejack |

## 安裝

```bash
# 推論用 (hbrowser 會自動安裝)
uv pip install ponychart-classifier

# 開發用 (包含訓練依賴)
uv pip install -e ".[train]"
```

### 環境損毀時重建

Python 版本升級後若 `uv run black` / `uv run mypy` 等工具出現
`ModuleNotFoundError: No module named '..._mypyc'` 這類錯誤，代表
mypyc 編譯的 extension 找不到內部模組，執行下列指令把 `.venv`
與所有快取重建：

```bash
./scripts/rebuild-env.sh
```

## 使用方式

```python
from ponychart_classifier import predict, preload, update, get_thresholds
from ponychart_classifier import PonyChartClassifier, PredictionResult, ClassThresholds

# 預先載入模型
preload()

# 檢查並更新模型至最新版本（比對 ETag，有新版才下載）
updated: bool = update()

# 預測圖片中的角色
result: PredictionResult = predict("path/to/image.png")
print(result.labels)  # frozenset({'Rarity', 'Fluttershy'})
print(result.rarity)  # 0.95
print(result.twilight_sparkle)  # 0.02

# 取得各角色的分類閾值
thresholds: ClassThresholds = get_thresholds()
```

也可以直接使用 `PonyChartClassifier` 類別：

```python
from ponychart_classifier import PonyChartClassifier

classifier = PonyChartClassifier(
    model_path="model.onnx", thresholds_path="thresholds.json"
)
result = classifier.predict("path/to/image.png", min_k=1, max_k=3)
```

## 工作流程

### 1. 收集圖片

將新的 PonyChart 截圖 (PNG) 放入 `rawimage/` 資料夾。

### 2. 安裝訓練依賴

```bash
# 只需一次，標註工具與訓練皆需要
uv pip install -e ".[train]"
```

### 3. 標註圖片

```bash
uv run python -m app.label_images
```

### 4. 訓練模型

```bash

# 執行訓練 (若存在 checkpoint.pt 則自動從上次結果繼續訓練)
uv run python scripts/train.py

# 強制從頭訓練 (忽略 checkpoint，從 ImageNet 預訓練權重開始)
uv run python scripts/train.py --from-scratch
```

訓練完成後會覆寫 `model.onnx`、`thresholds.json` 和 `checkpoint.pt`，下次推論自動使用新模型。

### Resume 訓練

新增圖片並標註後，直接執行 `train.py` 即可。腳本會自動偵測 `checkpoint.pt`：

- **有 checkpoint**: 載入之前的模型權重，跳過 Phase 1 (head-only)，直接進入 Phase 2 fine-tuning，收斂更快
- **無 checkpoint**: 從 ImageNet 預訓練權重開始完整兩階段訓練

### 訓練超參數

所有超參數集中於 `src/ponychart_classifier/training/constants.py`，修改後對所有腳本生效：

| 參數 | 預設值 | 說明 |
| ---- | ------ | ---- |
| `BACKBONE` | `efficientnet_b0` | 見下方支援的 backbone |
| `BATCH_SIZE` | 64 | 批次大小 |
| `SEED` | 42 | 隨機種子 |
| `PHASE1_EPOCHS` | 30 | Phase 1 (head-only) 訓練輪數 |
| `PHASE1_PATIENCE` | 5 | Phase 1 early stopping patience |
| `PHASE2_EPOCHS` | 100 | Phase 2 (full fine-tuning) 最大訓練輪數 |
| `PHASE2_PATIENCE` | 12 | Phase 2 early stopping patience |
| `LR_HEAD` | 4e-3 | Head 層學習率 |
| `LR_FEATURES` | 1.2e-4 | Backbone 特徵提取層學習率 |
| `LR_CLASSIFIER` | 1.2e-3 | 分類器層學習率 |
| `VAL_SIZE` | 0.15 | 驗證集比例 |
| `HOLDOUT_TEST_SIZE` | 0.20 | Holdout 測試集比例 |

## 支援的 Backbone

| Backbone | 參數量 | ONNX 大小 | 說明 |
| -------- | ------ | --------- | ---- |
| `mobilenet_v3_small` | 2.5M | ~4MB | 輕量快速 |
| `mobilenet_v3_large` | 5.4M | ~9MB | 精度最高 |
| `efficientnet_b0` | 5.3M | ~11MB | 預設，精度接近 Large，但訓練較慢 |
| `efficientnet_b2` | 9.1M | ~18MB | 最大模型，較高精度但較慢 |

所有 backbone 都使用 ImageNet 預訓練權重 + transfer learning。
推論端使用 ONNX Runtime，backbone 更換後只需重新匯出 `model.onnx`，推論程式碼不需改動。

## 分析腳本

分析腳本使用 `training/constants.py` 中的超參數設定：

```bash
# 比較不同 backbone 的效果
uv run --extra train python -m cli.compare_backbones --run efficientnet_b0
uv run --extra train python -m cli.compare_backbones --report

# 分析裁切圖片的影響
uv run python scripts/compare_crops.py

# 資料增強 ablation study
uv run --extra train python -m cli.analyze_augmentations --run hflip
uv run --extra train python -m cli.analyze_augmentations --report

# Learning curve 分析 (估算增加資料的邊際效益)
uv run python scripts/learning_curve.py

# LR 超參數搜尋
uv run --extra train python -m cli.search_batch_lr --run 64
uv run --extra train python -m cli.search_batch_lr --report
```

## 模型架構

- **Backbone**: 可選 MobileNetV3-Small/Large 或 EfficientNet-B0/B2 (預設 EfficientNet-B0，ImageNet 預訓練)
- **訓練策略**: Phase 1 head-only + Phase 2 full fine-tuning，支援從 checkpoint 繼續訓練
- **輸出**: 6 個 sigmoid 節點 (多標籤分類)
- **推論引擎**: ONNX Runtime (CPU)
- **推論速度**: 3-21ms / 張

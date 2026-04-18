"""
Batch size 超參數搜尋（Stage 1）— 跨機器分別運行再合併比較。

此 CLI 拆成兩種模式，方便把不同 batch size 派到不同設備上跑
（例如：bs<=64 在本地，bs>=96 在 Colab）：

1. 單跑一個 batch size 並把結果寫成 JSON：
     uv run --extra train python -m cli.search_batch_lr --run 32
     uv run --extra train python -m cli.search_batch_lr --run 64
     uv run --extra train python -m cli.search_batch_lr --run 96

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.search_batch_lr --report

預設 results dir 是 ``results/search_batch_lr/``，可用 ``--results-dir``
指定其他路徑。

固定 lr_scale=1.0，掃不同 batch size，依 Linear Scaling Rule
等比例調整 LR（``linear_factor = batch / BATCH_SIZE``）。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``batch{bs}__{hash12}.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。

train/val split 由 SEED + 樣本內容決定（``group_hash_split``），
所以同一份資料在任何機器上都會切出相同的訓練/驗證集。
"""

from __future__ import annotations

import gc
import socket
from pathlib import Path

import torch

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    INPUT_SIZE,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    SEARCH_PATIENCE,
    SEARCH_PHASE1_EPOCHS,
    SEARCH_PHASE2_EPOCHS,
    SEED,
    VAL_SIZE,
    describe_device,
    get_device,
    get_performance_cpu_count,
    group_hash_split,
    hash_samples,
    load_all_json_results,
    load_samples_logged,
    log_section,
    measure_training_memory,
    seed_all,
    select_consistent_results,
)
from ponychart_classifier.training.dataset import (
    PonyChartDataset,
    compute_cache_budget,
    get_transforms,
)

from .configs import SUGGESTED_BATCH_SIZES, build_config
from .result import SearchResult, parse_result_file, save_result
from .training import run_experiment


class SearchBatchLrCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Search batch sizes one-at-a-time and merge results across "
            "machines. Use --run to train one batch size (writes JSON), then "
            "--report to print a comparison from all collected JSON files."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "search_batch_lr"

    def available_keys(self) -> list[str]:
        return [str(bs) for bs in SUGGESTED_BATCH_SIZES]

    def run_one(self, key: str, results_dir: Path) -> None:
        batch_size = int(key)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        config = build_config(batch_size)
        seed_all(SEED)
        device = get_device()
        num_workers = get_performance_cpu_count()
        self.logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

        samples = load_samples_logged(self.logger)
        data_hash = hash_samples(samples)
        self.logger.info(
            "Data fingerprint: %s (full=%s)",
            data_hash[:HASH_PREFIX_LEN],
            data_hash,
        )

        train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        self.logger.info(
            "Train: %s  Val: %s",
            f"{len(train_samples):,}",
            f"{len(val_samples):,}",
        )

        log_section(self.logger, "BATCH SIZE: %d", batch_size, width=70)
        self.logger.info("  Backbone:    %s", BACKBONE)
        self.logger.info(
            "  lr_scale:    %.1fx  (linear_factor=%.3f)",
            config.lr_scale,
            batch_size / BATCH_SIZE,
        )
        self.logger.info(
            "  LRs:         head=%.1e  feat=%.1e  cls=%.1e",
            config.lr_head,
            config.lr_features,
            config.lr_classifier,
        )
        self.logger.info(
            "  Phase 1: %d epochs, Phase 2: %d epochs (patience=%d)",
            SEARCH_PHASE1_EPOCHS,
            SEARCH_PHASE2_EPOCHS,
            SEARCH_PATIENCE,
        )
        self.logger.info("  Data hash:   %s", data_hash[:HASH_PREFIX_LEN])

        # Cache budget reflects this run only (single batch size).
        training_reserve = measure_training_memory(
            BACKBONE,
            batch_size,
            INPUT_SIZE,
            device,
        )
        total_budget = compute_cache_budget(
            INPUT_SIZE,
            n_datasets=2,
            training_reserve=training_reserve,
        )
        n_total = len(train_samples) + len(val_samples)
        train_budget = int(total_budget * len(train_samples) / n_total)
        val_budget = total_budget - train_budget

        train_ds = PonyChartDataset(
            train_samples,
            get_transforms(is_train=True),
            max_cached=train_budget,
        )
        val_ds = PonyChartDataset(
            val_samples,
            get_transforms(is_train=False),
            max_cached=val_budget,
        )

        seed_all(SEED)
        (
            best_f1,
            best_per_class,
            phase1_stopped_epoch,
            stopped_epoch,
            elapsed_s,
        ) = run_experiment(
            train_ds,
            val_ds,
            device,
            num_workers,
            batch_size,
            config.lr_head,
            config.lr_features,
            config.lr_classifier,
            BACKBONE,
        )

        # Best-effort cleanup so the run process exits with a small footprint.
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        result = SearchResult(
            batch_size=batch_size,
            lr_scale=config.lr_scale,
            lr_head=config.lr_head,
            lr_features=config.lr_features,
            lr_classifier=config.lr_classifier,
            best_f1=best_f1,
            per_class_f1=best_per_class,
            phase1_stopped_epoch=phase1_stopped_epoch,
            stopped_epoch=stopped_epoch,
            time_s=elapsed_s,
            train_size=len(train_samples),
            val_size=len(val_samples),
            seed=SEED,
            backbone=BACKBONE,
            data_hash=data_hash,
            hostname=socket.gethostname(),
            device=describe_device(device),
        )

        self.logger.info(
            ">> batch=%d  F1=%.4f  p1_stop=%d  p2_stop=%d  time=%.1fs",
            batch_size,
            best_f1,
            phase1_stopped_epoch,
            stopped_epoch,
            elapsed_s,
        )

        out_path = save_result(result, results_dir)
        self.logger.info("Saved result to %s", out_path)

    def format_report(self, results_dir: Path) -> None:
        raw_results = load_all_json_results(results_dir, parse_result_file, self.logger)
        if not raw_results:
            msg = f"No results found in {results_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        by_batch = select_consistent_results(
            raw_results,
            key=lambda r: r.batch_size,
            key_label="batch_size",
            logger=self.logger,
        )
        data_hash = next(iter(by_batch.values())).data_hash
        backbones = {r.backbone for r in by_batch.values()}

        log_section(self.logger, "BATCH SIZE SEARCH RESULTS")
        self.logger.info("  Loaded %d result(s) from %s", len(by_batch), results_dir)
        self.logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])
        if len(backbones) > 1:
            self.logger.warning(
                "  Mixed backbones in results dir: %s",
                ", ".join(sorted(backbones)),
            )
        else:
            self.logger.info("  Backbone:      %s", next(iter(backbones)))

        # -- Results table sorted by F1 --
        sorted_by_f1 = sorted(by_batch.values(), key=lambda r: r.best_f1, reverse=True)

        self.logger.info("")
        self.logger.info(
            "  %-4s  %-6s  %-8s  %-10s  %-10s  %-10s  %-8s  %-6s  %-6s  %-7s",
            "Rank",
            "Batch",
            "LR scale",
            "LR head",
            "LR feat",
            "LR cls",
            "Macro F1",
            "P1 Ep",
            "P2 Ep",
            "Time",
        )
        self.logger.info("  " + "-" * 93)
        for rank, r in enumerate(sorted_by_f1, 1):
            self.logger.info(
                "  #%-3d  %-6d  %-8s  %-10.1e  %-10.1e  %-10.1e"
                "  %-8.4f  %-6d  %-6d  %-7.1fs",
                rank,
                r.batch_size,
                f"{r.lr_scale:.1f}x",
                r.lr_head,
                r.lr_features,
                r.lr_classifier,
                r.best_f1,
                r.phase1_stopped_epoch,
                r.stopped_epoch,
                r.time_s,
            )

        # Per-batch environment
        self.logger.info("")
        self.logger.info("Run environment:")
        for r in sorted(by_batch.values(), key=lambda r: r.batch_size):
            self.logger.info(
                "  batch=%-4d  host=%s  device=%s",
                r.batch_size,
                r.hostname,
                r.device,
            )

        # -- Per-class detail for all --
        self.logger.info("")
        self.logger.info("Per-class F1 for all configs:")
        for rank, r in enumerate(sorted_by_f1, 1):
            self.logger.info(
                "  #%d (batch=%d, scale=%.1fx, F1=%.4f):",
                rank,
                r.batch_size,
                r.lr_scale,
                r.best_f1,
            )
            for i, name in enumerate(CLASS_NAMES):
                self.logger.info("    %-20s  %.4f", name, r.per_class_f1[i])

        # -- Recommendation --
        best = sorted_by_f1[0]
        log_section(self.logger, "RECOMMENDATION")
        self.logger.info("  Best config:")
        self.logger.info("    --batch-size %d", best.batch_size)
        self.logger.info(
            "    Phase 1 lr: %.1e  (train.py default: %.1e)",
            best.lr_head,
            LR_HEAD,
        )
        self.logger.info(
            "    Phase 2 lr_features: %.1e  (train.py default: %.1e)",
            best.lr_features,
            LR_FEATURES,
        )
        self.logger.info(
            "    Phase 2 lr_classifier: %.1e  (train.py default: %.1e)",
            best.lr_classifier,
            LR_CLASSIFIER,
        )
        self.logger.info("")

        # Compare with baseline (batch=BATCH_SIZE)
        baseline = by_batch.get(BATCH_SIZE)
        if baseline is not None and best.batch_size != BATCH_SIZE:
            diff = best.best_f1 - baseline.best_f1
            same_device = baseline.device == best.device
            if same_device and best.time_s > 0:
                speedup = baseline.time_s / best.time_s
                self.logger.info(
                    "  vs baseline (batch=%d): F1 %+.4f, %.1fx speed",
                    BATCH_SIZE,
                    diff,
                    speedup,
                )
            else:
                self.logger.info(
                    "  vs baseline (batch=%d): F1 %+.4f  "
                    "(time speedup hidden — runs were on different devices)",
                    BATCH_SIZE,
                    diff,
                )
        elif baseline is not None:
            self.logger.info(
                "  Baseline (batch=%d) is already the best config.", BATCH_SIZE
            )
        else:
            self.logger.info(
                "  Baseline (batch=%d) not in results dir — skipping" " comparison.",
                BATCH_SIZE,
            )
        self.logger.info("=" * 90)


if __name__ == "__main__":
    SearchBatchLrCLI().main()

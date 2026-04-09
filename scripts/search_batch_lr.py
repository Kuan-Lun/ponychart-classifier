"""
Batch size 超參數搜尋（Stage 1）— 跨機器分別運行再合併比較。

此腳本拆成兩種模式，方便把不同 batch size 派到不同設備上跑
（例如：bs<=64 在本地，bs>=96 在 Colab）：

1. 單跑一個 batch size 並把結果寫成 JSON：
     uv run python scripts/search_batch_lr.py --run 32
     uv run python scripts/search_batch_lr.py --run 64
     uv run python scripts/search_batch_lr.py --run 96

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run python scripts/search_batch_lr.py --report

預設 results dir 是 ``scripts/batch_lr_results/``，可用 ``--results-dir``
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

import argparse
import gc
import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    INPUT_SIZE,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    MIN_DELTA_F1,
    PRE_RESIZE,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEARCH_PATIENCE,
    SEARCH_PHASE1_EPOCHS,
    SEARCH_PHASE2_EPOCHS,
    SEED,
    VAL_SIZE,
    WEIGHT_DECAY,
    EnvDict,
    build_model,
    describe_device,
    evaluate,
    get_device,
    get_performance_cpu_count,
    group_hash_split,
    hash_samples,
    load_all_json_results,
    load_samples_logged,
    log_section,
    make_dataloader,
    measure_training_memory,
    seed_all,
    select_consistent_results,
    train_one_epoch,
)
from ponychart_classifier.training.dataset import (
    PonyChartDataset,
    compute_cache_budget,
    get_transforms,
)
from ponychart_classifier.training.model import _extract_submodules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "batch_lr_results"

# Stage 1 固定 lr_scale=1.0；lr_scale 留給 Stage 2 在贏的 batch 上微調。
LR_SCALE = 1.0

# Base LRs from constants (single source of truth with train.py)
BASE_LR_HEAD = LR_HEAD
BASE_LR_FEATURES = LR_FEATURES
BASE_LR_CLASSIFIER = LR_CLASSIFIER


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    """Result of a single batch-size search experiment."""

    batch_size: int
    lr_scale: float
    lr_head: float
    lr_features: float
    lr_classifier: float
    best_f1: float
    per_class_f1: list[float]
    stopped_epoch: int
    time_s: float
    train_size: int
    val_size: int
    seed: int
    backbone: str
    data_hash: str
    hostname: str
    device: str


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------
class SplitDict(TypedDict):
    train_size: int
    val_size: int


class SearchResultDict(TypedDict):
    batch_size: int
    lr_scale: float
    lr_head: float
    lr_features: float
    lr_classifier: float
    best_f1: float
    per_class_f1: list[float]
    stopped_epoch: int
    time_s: float
    split: SplitDict
    seed: int
    backbone: str
    data_hash: str
    env: EnvDict


def result_to_dict(result: SearchResult) -> SearchResultDict:
    return SearchResultDict(
        batch_size=result.batch_size,
        lr_scale=result.lr_scale,
        lr_head=result.lr_head,
        lr_features=result.lr_features,
        lr_classifier=result.lr_classifier,
        best_f1=result.best_f1,
        per_class_f1=list(result.per_class_f1),
        stopped_epoch=result.stopped_epoch,
        time_s=result.time_s,
        split=SplitDict(
            train_size=result.train_size,
            val_size=result.val_size,
        ),
        seed=result.seed,
        backbone=result.backbone,
        data_hash=result.data_hash,
        env=EnvDict(hostname=result.hostname, device=result.device),
    )


def result_from_dict(data: SearchResultDict) -> SearchResult:
    split = data["split"]
    env = data["env"]
    return SearchResult(
        batch_size=data["batch_size"],
        lr_scale=data["lr_scale"],
        lr_head=data["lr_head"],
        lr_features=data["lr_features"],
        lr_classifier=data["lr_classifier"],
        best_f1=data["best_f1"],
        per_class_f1=list(data["per_class_f1"]),
        stopped_epoch=data["stopped_epoch"],
        time_s=data["time_s"],
        train_size=split["train_size"],
        val_size=split["val_size"],
        seed=data["seed"],
        backbone=data["backbone"],
        data_hash=data["data_hash"],
        hostname=env["hostname"],
        device=env["device"],
    )


def _parse_result_json(raw: str) -> SearchResultDict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(SearchResultDict, parsed)


def result_filename(batch_size: int, data_hash: str) -> str:
    """Return the canonical JSON filename for a (batch_size, data) pair."""
    return f"batch{batch_size:03d}__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(result: SearchResult, results_dir: Path) -> Path:
    """Write *result* to ``<results_dir>/batch{bs}__{hash12}.json``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(result.batch_size, result.data_hash)
    out_path.write_text(json.dumps(result_to_dict(result), indent=2))
    return out_path


def _parse_result_file(raw: str) -> SearchResult:
    """Parse one JSON document into a :class:`SearchResult`."""
    return result_from_dict(_parse_result_json(raw))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_experiment(
    train_ds: PonyChartDataset,
    val_ds: PonyChartDataset,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    lr_head: float,
    lr_features: float,
    lr_classifier: float,
    backbone: str,
) -> tuple[float, list[float], int, float]:
    """Run one training experiment.

    Returns ``(best_f1, best_per_class_f1, stopped_epoch, elapsed_s)``.
    """
    t0 = time.monotonic()
    train_loader = make_dataloader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    model = build_model(backbone=backbone, pretrained=True).to(device)
    features, classifier = _extract_submodules(model)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: Head only
    for param in features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr_head, weight_decay=WEIGHT_DECAY
    )
    for _epoch in range(1, SEARCH_PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Phase 2: Full fine-tuning
    for param in features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": features.parameters(), "lr": lr_features},
            {"params": classifier.parameters(), "lr": lr_classifier},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    best_f1 = 0.0
    best_per_class: list[float] = []
    patience_counter = 0
    stopped_epoch = SEARCH_PHASE2_EPOCHS

    for epoch in range(1, SEARCH_PHASE2_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = val_result.macro_f1
        scheduler.step(val_f1)

        if val_f1 > best_f1 + MIN_DELTA_F1:
            best_f1 = val_f1
            best_per_class = list(val_result.per_class_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= SEARCH_PATIENCE:
            stopped_epoch = epoch
            break

    return best_f1, best_per_class, stopped_epoch, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_run(batch_size: int, results_dir: Path) -> None:
    """Train one batch_size config and persist the result to *results_dir*."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    seed_all(SEED)
    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

    samples = load_samples_logged(logger)
    data_hash = hash_samples(samples)
    logger.info(
        "Data fingerprint: %s (full=%s)", data_hash[:HASH_PREFIX_LEN], data_hash
    )

    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info(
        "Train: %s  Val: %s", f"{len(train_samples):,}", f"{len(val_samples):,}"
    )

    linear_factor = batch_size / BATCH_SIZE
    lr_head = BASE_LR_HEAD * linear_factor * LR_SCALE
    lr_features = BASE_LR_FEATURES * linear_factor * LR_SCALE
    lr_classifier = BASE_LR_CLASSIFIER * linear_factor * LR_SCALE

    log_section(logger, "BATCH SIZE: %d", batch_size, width=70)
    logger.info("  Backbone:    %s", BACKBONE)
    logger.info("  lr_scale:    %.1fx  (linear_factor=%.3f)", LR_SCALE, linear_factor)
    logger.info(
        "  LRs:         head=%.1e  feat=%.1e  cls=%.1e",
        lr_head,
        lr_features,
        lr_classifier,
    )
    logger.info(
        "  Phase 1: %d epochs, Phase 2: %d epochs (patience=%d)",
        SEARCH_PHASE1_EPOCHS,
        SEARCH_PHASE2_EPOCHS,
        SEARCH_PATIENCE,
    )
    logger.info("  Data hash:   %s", data_hash[:HASH_PREFIX_LEN])

    # Cache budget reflects this run only (single batch size).
    training_reserve = measure_training_memory(
        BACKBONE,
        batch_size,
        INPUT_SIZE,
        device,
    )
    total_budget = compute_cache_budget(
        PRE_RESIZE,
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
    best_f1, best_per_class, stopped_epoch, elapsed_s = run_experiment(
        train_ds,
        val_ds,
        device,
        num_workers,
        batch_size,
        lr_head,
        lr_features,
        lr_classifier,
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
        lr_scale=LR_SCALE,
        lr_head=lr_head,
        lr_features=lr_features,
        lr_classifier=lr_classifier,
        best_f1=best_f1,
        per_class_f1=best_per_class,
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

    logger.info(
        ">> batch=%d  F1=%.4f  stopped_epoch=%d  time=%.1fs",
        batch_size,
        best_f1,
        stopped_epoch,
        elapsed_s,
    )

    out_path = save_result(result, results_dir)
    logger.info("Saved result to %s", out_path)


def cmd_report(results_dir: Path) -> None:
    """Print a comparison table from all JSON files in *results_dir*."""
    raw_results = load_all_json_results(results_dir, _parse_result_file, logger)
    if not raw_results:
        msg = f"No results found in {results_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    by_batch = select_consistent_results(
        raw_results,
        key=lambda r: r.batch_size,
        key_label="batch_size",
        logger=logger,
    )
    data_hash = next(iter(by_batch.values())).data_hash
    backbones = {r.backbone for r in by_batch.values()}

    log_section(logger, "BATCH SIZE SEARCH RESULTS")
    logger.info("  Loaded %d result(s) from %s", len(by_batch), results_dir)
    logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])
    if len(backbones) > 1:
        logger.warning(
            "  Mixed backbones in results dir: %s", ", ".join(sorted(backbones))
        )
    else:
        logger.info("  Backbone:      %s", next(iter(backbones)))

    # ── Results table sorted by F1 ──
    sorted_by_f1 = sorted(by_batch.values(), key=lambda r: r.best_f1, reverse=True)

    logger.info("")
    logger.info(
        "  %-4s  %-6s  %-8s  %-10s  %-10s  %-10s  %-8s  %-6s  %-7s",
        "Rank",
        "Batch",
        "LR scale",
        "LR head",
        "LR feat",
        "LR cls",
        "Macro F1",
        "Epoch",
        "Time",
    )
    logger.info("  " + "-" * 85)
    for rank, r in enumerate(sorted_by_f1, 1):
        logger.info(
            "  #%-3d  %-6d  %-8s  %-10.1e  %-10.1e  %-10.1e  %-8.4f  %-6d  %-7.1fs",
            rank,
            r.batch_size,
            f"{r.lr_scale:.1f}x",
            r.lr_head,
            r.lr_features,
            r.lr_classifier,
            r.best_f1,
            r.stopped_epoch,
            r.time_s,
        )

    # Per-batch environment (so cross-machine time numbers are interpretable)
    logger.info("")
    logger.info("Run environment:")
    for r in sorted(by_batch.values(), key=lambda r: r.batch_size):
        logger.info(
            "  batch=%-4d  host=%s  device=%s", r.batch_size, r.hostname, r.device
        )

    # ── Per-class detail for all ──
    logger.info("")
    logger.info("Per-class F1 for all configs:")
    for rank, r in enumerate(sorted_by_f1, 1):
        logger.info(
            "  #%d (batch=%d, scale=%.1fx, F1=%.4f):",
            rank,
            r.batch_size,
            r.lr_scale,
            r.best_f1,
        )
        for i, name in enumerate(CLASS_NAMES):
            logger.info("    %-20s  %.4f", name, r.per_class_f1[i])

    # ── Recommendation ──
    best = sorted_by_f1[0]
    log_section(logger, "RECOMMENDATION")
    logger.info("  Best config:")
    logger.info("    --batch-size %d", best.batch_size)
    logger.info("    Phase 1 lr: %.1e  (train.py default: %.1e)", best.lr_head, LR_HEAD)
    logger.info(
        "    Phase 2 lr_features: %.1e  (train.py default: %.1e)",
        best.lr_features,
        LR_FEATURES,
    )
    logger.info(
        "    Phase 2 lr_classifier: %.1e  (train.py default: %.1e)",
        best.lr_classifier,
        LR_CLASSIFIER,
    )
    logger.info("")

    # Compare with baseline (batch=BATCH_SIZE)
    baseline = by_batch.get(BATCH_SIZE)
    if baseline is not None and best.batch_size != BATCH_SIZE:
        diff = best.best_f1 - baseline.best_f1
        # Time speedup is only meaningful if both runs were on the same device.
        same_device = baseline.device == best.device
        if same_device and best.time_s > 0:
            speedup = baseline.time_s / best.time_s
            logger.info(
                "  vs baseline (batch=%d): F1 %+.4f, %.1fx speed",
                BATCH_SIZE,
                diff,
                speedup,
            )
        else:
            logger.info(
                "  vs baseline (batch=%d): F1 %+.4f  "
                "(time speedup hidden — runs were on different devices)",
                BATCH_SIZE,
                diff,
            )
    elif baseline is not None:
        logger.info("  Baseline (batch=%d) is already the best config.", BATCH_SIZE)
    else:
        logger.info(
            "  Baseline (batch=%d) not in results dir — skipping comparison.",
            BATCH_SIZE,
        )
    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search batch sizes one-at-a-time and merge results across "
            "machines. Use --run to train one batch size (writes JSON), then "
            "--report to print a comparison from all collected JSON files."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run",
        metavar="BATCH_SIZE",
        type=int,
        help="Train one batch size and save its result JSON.",
    )
    mode.add_argument(
        "--report",
        action="store_true",
        help="Print a comparison from all result JSONs in --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Where result JSONs live (default: {DEFAULT_RESULTS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run is not None:
        cmd_run(args.run, args.results_dir)
    else:
        cmd_report(args.results_dir)


if __name__ == "__main__":
    main()

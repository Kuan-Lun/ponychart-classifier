"""
比較 resume training vs from-scratch training，找出 checkpoint 過時的臨界點。

以時間順序將資料分為「舊資料」與「新資料」，模擬不同 checkpoint 年齡下的
resume 效果，與同資料量的 from-scratch 比較，找出 crossover point。

實驗設計：
  - 20% holdout test set（僅原圖）
  - 對每個 base_fraction（模擬 checkpoint 只看過前 F% 的資料）：
    1. 用前 F% 資料 from scratch 訓練 -> 取得 checkpoint
    2. 用該 checkpoint resume，在 100% 資料上 fine-tune -> resume_f1
  - Baseline：100% 資料 from scratch -> scratch_f1
  - 比較 delta = scratch_f1 - resume_f1，找出何時 from-scratch 開始勝出

使用方式：
  uv run python scripts/compare_resume_scratch.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    configure_logging,
    get_transforms,
    log_per_class_table,
    log_section,
    make_test_loader,
    prepare_balanced_train_val_for_group_keys,
    prepare_group_holdout_setup_logged,
    train_and_evaluate_on_test,
)

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeExperiment:
    fraction: float
    base_n: int
    new_n: int
    new_ratio: float
    resume_f1: float
    resume_per_class_f1: list[float]
    delta: float


BASE_FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]
SAFETY_MARGIN = 0.75


def find_crossover(
    ratios: list[float],
    deltas: list[float],
) -> float | None:
    """Find the new_data_ratio where delta crosses from negative to positive.

    Uses linear interpolation between adjacent points.
    Returns None if no crossover is found.
    """
    for i in range(len(deltas) - 1):
        if deltas[i] <= 0 < deltas[i + 1]:
            # Linear interpolation
            t = -deltas[i] / (deltas[i + 1] - deltas[i])
            return ratios[i] + t * (ratios[i + 1] - ratios[i])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup = prepare_group_holdout_setup_logged(
        logger,
        seed=SEED,
        test_size=HOLDOUT_TEST_SIZE,
        val_size=VAL_SIZE,
    )
    val_group_keys = setup.val_group_keys

    # Prepare shared test loader
    test_loader = make_test_loader(
        setup.test_samples,
        BATCH_SIZE,
        setup.num_workers,
        setup.device,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Collect pool samples ──
    train_val_group_keys = setup.train_val_group_keys
    pool_samples = [
        setup.all_samples[i]
        for gk in train_val_group_keys
        for i in setup.groups[gk]
    ]

    # Sort pool group keys chronologically
    sorted_pool_groups = sorted(train_val_group_keys)
    logger.info("Pool groups (chronological): %d", len(sorted_pool_groups))

    # ── Full train/val split for baseline and resume targets ──
    full_train, full_val = prepare_balanced_train_val_for_group_keys(
        setup.all_samples,
        setup.groups,
        train_val_group_keys,
        seed=SEED,
        val_group_keys=val_group_keys,
    )
    logger.info("Full pool: train=%d  val=%d", len(full_train), len(full_val))

    # ── Baseline: 100% from scratch ──
    log_section(logger, "BASELINE: %d%% data from scratch", 100, width=70)

    scratch_run = train_and_evaluate_on_test(
        full_train,
        full_val,
        device=setup.device,
        num_workers=setup.num_workers,
        label="Baseline (100% from scratch)",
        test_loader=test_loader,
        criterion=criterion,
        train_transform=get_transforms(is_train=True),
        backbone=BACKBONE,
    )
    scratch_result = scratch_run.eval_result
    scratch_f1 = scratch_result.macro_f1
    logger.info(">> Baseline scratch F1: %.4f", scratch_f1)

    # ── Per-fraction experiments ──
    experiment_results: list[ResumeExperiment] = []

    for frac in BASE_FRACTIONS:
        n_groups = max(1, int(np.ceil(frac * len(sorted_pool_groups))))
        base_group_keys = sorted_pool_groups[:n_groups]
        base_n = sum(len(setup.groups[gk]) for gk in base_group_keys)

        new_n = len(pool_samples) - base_n
        new_ratio = new_n / max(base_n, 1)
        pct = int(frac * 100)

        # Step 1: Train from scratch on base data -> checkpoint
        log_section(
            logger,
            "BASE %d%% (%d samples) -> new_data_ratio=%.1f%%",
            pct,
            base_n,
            new_ratio * 100,
            width=70,
        )

        base_train, base_val = prepare_balanced_train_val_for_group_keys(
            setup.all_samples,
            setup.groups,
            base_group_keys,
            seed=SEED,
            val_group_keys=val_group_keys,
        )
        logger.info("  Base train=%d  val=%d", len(base_train), len(base_val))

        base_run = train_and_evaluate_on_test(
            base_train,
            base_val,
            device=setup.device,
            num_workers=setup.num_workers,
            label=f"Base {pct}% from scratch",
            test_loader=test_loader,
            criterion=criterion,
            train_transform=get_transforms(is_train=True),
            backbone=BACKBONE,
        )

        # Step 2: Resume from base checkpoint with 100% data
        resume_run = train_and_evaluate_on_test(
            full_train,
            full_val,
            device=setup.device,
            num_workers=setup.num_workers,
            label=f"Resume from {pct}% checkpoint",
            test_loader=test_loader,
            criterion=criterion,
            train_transform=get_transforms(is_train=True),
            backbone=BACKBONE,
            resume_state_dict=base_run.state_dict,
        )
        resume_result = resume_run.eval_result

        delta = scratch_f1 - resume_result.macro_f1
        experiment_results.append(
            ResumeExperiment(
                fraction=frac,
                base_n=base_n,
                new_n=new_n,
                new_ratio=new_ratio,
                resume_f1=resume_result.macro_f1,
                resume_per_class_f1=resume_result.per_class_f1,
                delta=delta,
            )
        )
        logger.info(
            ">> Base %d%%: resume_f1=%.4f  scratch_f1=%.4f  delta=%+.4f",
            pct,
            resume_result.macro_f1,
            scratch_f1,
            delta,
        )

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    log_section(logger, "COMPARISON TABLE (scratch F1=%.4f)", scratch_f1)
    logger.info("")
    logger.info(
        "  %-6s  %-8s  %-8s  %-12s  %-10s  %-10s  %-10s  %-10s",
        "Base%",
        "Base N",
        "New N",
        "New Ratio",
        "Resume F1",
        "Scratch F1",
        "Delta",
        "Winner",
    )
    logger.info("  " + "-" * 88)

    for r in experiment_results:
        winner = "SCRATCH" if r.delta > 0 else "RESUME"
        logger.info(
            "  %-6d  %-8d  %-8d  %-12.1f%%  %-10.4f  %-10.4f  %+-10.4f  %-10s",
            int(r.fraction * 100),
            r.base_n,
            r.new_n,
            r.new_ratio * 100,
            r.resume_f1,
            scratch_f1,
            r.delta,
            winner,
        )

    # ── Per-class breakdown ──
    log_section(
        logger,
        "PER-CLASS DELTA (scratch_f1 - resume_f1, positive = scratch better)",
    )
    logger.info("")
    log_per_class_table(
        logger,
        [f"{int(r.fraction * 100)}%" for r in experiment_results],
        lambda i, j: f"{scratch_result.per_class_f1[i] - experiment_results[j].resume_per_class_f1[i]:+.4f}",
        col_width=8,
    )

    # ── Crossover analysis ──
    log_section(logger, "CROSSOVER ANALYSIS")

    ratios = [r.new_ratio for r in experiment_results]
    deltas = [r.delta for r in experiment_results]
    crossover = find_crossover(ratios, deltas)

    if crossover is not None:
        logger.info("  Crossover point: new_data_ratio ≈ %.1f%%", crossover * 100)
        logger.info(
            "  (當新增資料超過 checkpoint 訓練量的 %.1f%% 時，"
            "from-scratch 開始優於 resume)",
            crossover * 100,
        )
    elif all(d <= 0 for d in deltas):
        logger.info("  在所有測試的 base fractions 下，resume 始終優於或等於 scratch。")
        logger.info("  未找到 crossover point。")
    elif all(d > 0 for d in deltas):
        logger.info("  在所有測試的 base fractions 下，scratch 始終優於 resume。")
        logger.info("  建議始終使用 from-scratch 訓練。")

    # ── Per-class warnings ──
    logger.info("")
    logger.info("Per-class 警示 (在最保守的 base fraction 下):")
    worst_idx = -1  # Use the largest base fraction (smallest new_ratio)
    worst_r = experiment_results[worst_idx]
    warned = False
    for i, name in enumerate(CLASS_NAMES):
        class_delta = scratch_result.per_class_f1[i] - worst_r.resume_per_class_f1[i]
        if class_delta > 0.03:
            logger.info(
                "  %s: 即使 base=%d%%, resume 仍比 scratch 差 %.4f F1",
                name,
                int(worst_r.fraction * 100),
                class_delta,
            )
            warned = True
    if not warned:
        logger.info(
            "  無異常。在 base=%d%% 下各 class 的 resume 表現均正常。",
            int(worst_r.fraction * 100),
        )

    # ══════════════════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════
    log_section(logger, "RECOMMENDATION")

    if crossover is not None:
        recommended = crossover * SAFETY_MARGIN
        logger.info("  Crossover: new_data_ratio ≈ %.1f%%", crossover * 100)
        logger.info(
            "  安全邊界 (×%.0f%%): %.1f%%",
            SAFETY_MARGIN * 100,
            recommended * 100,
        )
        logger.info("")
        logger.info("  具體行動:")
        logger.info(
            "    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = %.2f",
            recommended,
        )
        logger.info("")
        logger.info(
            "  效果: 當 load_samples() 的樣本數超過 checkpoint 訓練量的 %.1f%% 時，",
            recommended * 100,
        )
        logger.info("        train.py 會自動切換為 from-scratch 訓練。")
    elif all(d <= 0 for d in deltas):
        max_ratio = max(ratios)
        recommended = max_ratio * 2
        logger.info(
            "  Resume 在所有測試範圍內皆優於 scratch (最大 new_ratio=%.1f%%)。",
            max_ratio * 100,
        )
        logger.info("")
        logger.info("  具體行動:")
        logger.info(
            "    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = %.2f",
            recommended,
        )
        logger.info("")
        logger.info(
            "  說明: 設為 %.2f (測試範圍 %.1f%% 的兩倍)，" "保留充足 resume 空間。",
            recommended,
            max_ratio * 100,
        )
        logger.info("  建議: 當資料量大幅增加後重新執行此分析工具。")
    else:
        logger.info("  Scratch 在所有測試範圍內皆優於 resume。")
        logger.info("")
        logger.info("  具體行動:")
        logger.info("    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = 0.0")
        logger.info("")
        logger.info("  效果: train.py 將永遠使用 from-scratch 訓練。")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()

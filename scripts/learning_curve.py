"""
Learning Curve 分析腳本：估算增加 raw 資料對模型 F1 的邊際效益。

以不同比例的訓練 groups 進行巢狀、接續訓練（模擬 train.py 的增量流程），
在共用 test set 上評估，以 power-law 外推預測加 N 張新 raw 圖的預期 F1 提升。

使用方式：
  uv run python scripts/learning_curve.py
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
    LIMITED_HEADROOM,
    RETRAIN_NEW_DATA_RATIO,
    SATURATED_HEADROOM,
    SEED,
    VAL_SIZE,
    EvalResult,
    PerClassColumn,
    configure_logging,
    get_transforms,
    log_per_class_f1_matrix,
    log_per_class_table,
    log_section,
    make_test_loader,
    prepare_balanced_train_val_for_group_keys,
    prepare_group_holdout_setup_logged,
    train_and_evaluate_on_test,
)

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

# 每一步的樣本數都比上一步多出「剛好超過 RETRAIN_NEW_DATA_RATIO」，
# 讓 sequential resume 的每個 fraction 都會觸發 from-scratch retrain。
# 額外加 0.002 的 margin 吸收浮點誤差與 nested_subsample_groups 的 ceil() rounding。
_FRACTION_STEP = 1.0 + RETRAIN_NEW_DATA_RATIO + 0.002
DATA_FRACTIONS = sorted(1.0 / _FRACTION_STEP**k for k in range(10))


@dataclass(frozen=True)
class CurvePoint:
    eval_result: EvalResult
    fraction: float
    n_samples: int
    n_train: int
    n_val: int
    thresholds: list[float]


# ---------------------------------------------------------------------------
# Unique helpers for learning curve
# ---------------------------------------------------------------------------
def nested_subsample_groups(
    group_keys: list[str],
    fraction: float,
    shuffled_order: list[int],
) -> list[str]:
    """從 group_keys 中取前 fraction 比例的 groups（巢狀抽樣）。

    使用預先洗牌的索引順序，保證較小 fraction 是較大 fraction 的子集。
    """
    n = max(1, int(np.ceil(fraction * len(group_keys))))
    return [group_keys[i] for i in shuffled_order[:n]]


def fit_power_law(
    ns: list[int],
    f1s: list[float],
    max_asymptote: float = 1.0,
) -> tuple[float, float, float] | None:
    """Fit F1 ~ a - b * N^(-c) using least-squares grid search on c.

    Constraints:
      - a <= max_asymptote (F1 cannot exceed 1.0)
      - b > 0 (F1 increases with N)

    Returns (a, b, c) or None if fitting fails.
    """
    if len(ns) < 3:
        return None

    x = np.array(ns, dtype=float)
    y = np.array(f1s, dtype=float)

    best_residual = float("inf")
    best_params: tuple[float, float, float] | None = None

    for c in np.arange(0.1, 2.01, 0.05):
        # F1 = a - b * N^(-c); let z = -N^(-c), then F1 = a + b*z
        z = -(x ** (-c))
        design = np.column_stack([np.ones_like(z), z])
        try:
            params, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a_hat, b_hat = params

        if b_hat < 0 or a_hat > max_asymptote:
            continue

        pred = a_hat - b_hat * (x ** (-c))
        res = float(np.sum((y - pred) ** 2))
        if res < best_residual:
            best_residual = res
            best_params = (float(a_hat), float(b_hat), float(c))

    return best_params


def extrapolate_f1(params: tuple[float, float, float], n: int) -> float:
    """Predict F1 at sample count n using fitted power-law parameters."""
    a, b, c = params
    return float(a - b * (n ** (-c)))


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
    train_val_group_keys = setup.train_val_group_keys

    # Prepare test loader (shared)
    test_loader = make_test_loader(
        setup.test_samples,
        BATCH_SIZE,
        setup.num_workers,
        setup.device,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Prepare nested sampling order (shuffle once) ──
    rng = np.random.RandomState(SEED)
    shuffled_order = list(rng.permutation(len(train_val_group_keys)))

    # ── Run experiments at different data fractions (sequential resume) ──
    experiment_results: list[CurvePoint] = []
    prev_state_dict: dict[str, object] | None = None

    for frac in DATA_FRACTIONS:
        # Nested subsample: smaller fractions are always subsets of larger ones
        selected_groups = nested_subsample_groups(
            train_val_group_keys, frac, shuffled_order
        )
        train_samples, val_samples = prepare_balanced_train_val_for_group_keys(
            setup.all_samples,
            setup.groups,
            selected_groups,
            seed=SEED,
            val_group_keys=val_group_keys,
        )
        n_samples = sum(len(setup.groups[gk]) for gk in selected_groups)

        pct = int(frac * 100)
        name = f"{pct}% data ({n_samples} samples)"

        run = train_and_evaluate_on_test(
            train_samples,
            val_samples,
            device=setup.device,
            num_workers=setup.num_workers,
            label=name,
            test_loader=test_loader,
            criterion=criterion,
            reset_seed=True,
            train_transform=get_transforms(is_train=True),
            backbone=BACKBONE,
            resume_state_dict=prev_state_dict,
        )
        prev_state_dict = run.state_dict
        experiment_results.append(
            CurvePoint(
                eval_result=run.eval_result,
                fraction=frac,
                n_samples=n_samples,
                n_train=len(train_samples),
                n_val=len(val_samples),
                thresholds=run.train_result.thresholds,
            )
        )

        logger.info(
            ">> %s: test Macro F1=%.4f  thresholds=%s",
            name,
            run.eval_result.macro_f1,
            dict(zip(CLASS_NAMES, run.train_result.thresholds)),
        )

    # ── Comparison table ──
    log_section(
        logger,
        "LEARNING CURVE RESULTS (test set: %d original images)",
        len(setup.test_samples),
    )

    logger.info("")
    logger.info(
        "%-8s  %-10s  %-8s  %-8s  %-10s  %-10s",
        "Frac",
        "Samples",
        "Train",
        "Val",
        "Macro F1",
        "Delta",
    )
    logger.info("-" * 65)
    prev_f1 = 0.0
    for r in experiment_results:
        pct_str = f"{int(r.fraction * 100)}%"
        delta = r.eval_result.macro_f1 - prev_f1 if prev_f1 > 0 else 0.0
        delta_str = f"{delta:+.4f}" if prev_f1 > 0 else "---"
        logger.info(
            "%-8s  %-10d  %-8d  %-8d  %-10.4f  %-10s",
            pct_str,
            r.n_samples,
            r.n_train,
            r.n_val,
            r.eval_result.macro_f1,
            delta_str,
        )
        prev_f1 = r.eval_result.macro_f1

    # Per-class F1 table
    columns = [
        PerClassColumn(
            label=f"{int(r.fraction * 100)}%",
            macro_f1=r.eval_result.macro_f1,
            per_class_f1=r.eval_result.per_class_f1,
        )
        for r in experiment_results
    ]
    log_per_class_f1_matrix(logger, list(CLASS_NAMES), columns)

    # Per-class threshold table
    logger.info("")
    logger.info("Per-class optimized thresholds:")
    log_per_class_table(
        logger,
        [f"{int(r.fraction * 100)}%" for r in experiment_results],
        lambda i, j: f"{experiment_results[j].thresholds[i]:.4f}",
    )

    # ── Power-law extrapolation ──
    log_section(logger, "POWER-LAW EXTRAPOLATION")

    ns = [r.n_samples for r in experiment_results]
    macro_f1s = [r.eval_result.macro_f1 for r in experiment_results]

    # Macro F1 extrapolation
    params = fit_power_law(ns, macro_f1s)
    if params is not None:
        a, b, c = params
        logger.info("  Fitted: F1 ~ %.4f - %.4f * N^(-%.2f)", a, b, c)
        logger.info("  Asymptotic F1 (N->inf): %.4f", a)
        logger.info("")

        current_n = ns[-1]
        current_f1 = macro_f1s[-1]
        logger.info(
            "  %-20s  %-12s  %-12s",
            "Additional samples",
            "Predicted F1",
            "Delta",
        )
        logger.info("  " + "-" * 50)
        for extra in [1, 5, 10, 20, 50, 100, 200, 400]:
            pred = extrapolate_f1(params, current_n + extra)
            delta = pred - current_f1
            logger.info(
                "  +%-19d  %-12.4f  %+.4f",
                extra,
                min(pred, a),
                min(delta, a - current_f1),
            )

        # 計算達到特定 F1 所需的資料量
        logger.info("")
        logger.info("  Estimated samples needed for target F1:")
        gap = a - current_f1
        for target_pct in [0.25, 0.50, 0.75, 0.90]:
            target_f1 = current_f1 + gap * target_pct
            if target_f1 >= a:
                logger.info(
                    "    %.0f%% of gap (F1=%.4f): " "unreachable (asymptote=%.4f)",
                    target_pct * 100,
                    target_f1,
                    a,
                )
                continue
            # a - b * N^(-c) = target_f1
            # => N = (b / (a - target_f1))^(1/c)
            needed_n = (b / (a - target_f1)) ** (1 / c)
            extra_needed = max(0, int(needed_n) - current_n)
            logger.info(
                "    %.0f%% of gap (F1=%.4f): " "~%d total samples (+%d new)",
                target_pct * 100,
                target_f1,
                int(needed_n),
                extra_needed,
            )
    else:
        logger.info("  Power-law fitting failed (insufficient data points).")

    # ── Per-class saturation analysis ──
    log_section(logger, "PER-CLASS SATURATION ANALYSIS")

    logger.info("")
    logger.info(
        "  %-20s  %-8s  %-8s  %-10s  %-10s  %s",
        "Class",
        f"F1@{int(DATA_FRACTIONS[0] * 100)}%",
        "F1@100%",
        "Gain",
        "Slope",
        "Status",
    )
    logger.info("  " + "-" * 80)

    for i, name in enumerate(CLASS_NAMES):
        f1_first = experiment_results[0].eval_result.per_class_f1[i]
        f1_last = experiment_results[-1].eval_result.per_class_f1[i]
        gain = f1_last - f1_first

        # Slope between last two points (marginal gain)
        if len(experiment_results) >= 2:
            f1_prev = experiment_results[-2].eval_result.per_class_f1[i]
            n_prev = experiment_results[-2].n_samples
            n_last = experiment_results[-1].n_samples
            slope = (f1_last - f1_prev) / max(n_last - n_prev, 1) * 100
        else:
            slope = 0.0

        if gain < 0.01 and f1_last > 0.85:
            status = "SATURATED"
        elif slope > 0.005:
            status = "GROWING"
        elif slope > 0.001:
            status = "SLOWING"
        elif gain < 0.02:
            status = "FLAT"
        else:
            status = "SATURATING"

        logger.info(
            "  %-20s  %-8.4f  %-8.4f  %+-.4f     %.5f/100  %s",
            name,
            f1_first,
            f1_last,
            gain,
            slope,
            status,
        )

    # ── Per-class extrapolation ──
    logger.info("")
    logger.info("Per-class extrapolation:")
    current_n = ns[-1]
    extras = [1, 5, 10, 20, 50, 100, 200, 400]

    # Header
    hdr = f"  {'Class':<20s}  {'now':>8s}"
    for ex in extras:
        hdr += f"  {'+' + str(ex):>8s}"
    hdr += f"  {'asymptote':>10s}"
    logger.info(hdr)
    logger.info("  " + "-" * (20 + 2 + 8 + (2 + 8) * len(extras) + 12))

    for i, name in enumerate(CLASS_NAMES):
        class_f1s = [r.eval_result.per_class_f1[i] for r in experiment_results]
        class_params = fit_power_law(ns, class_f1s)
        if class_params is not None:
            asymptote = class_params[0]
            row = f"  {name:<20s}  {class_f1s[-1]:>8.4f}"
            for ex in extras:
                pred = extrapolate_f1(class_params, current_n + ex)
                row += f"  {min(pred, asymptote):>8.4f}"
            row += f"  {asymptote:>10.4f}"
            logger.info(row)
        else:
            logger.info("  %-20s  fitting failed", name)

    # ── Summary ──
    log_section(logger, "SUMMARY")
    logger.info(
        "  Current: %d samples -> Macro F1=%.4f",
        ns[-1],
        macro_f1s[-1],
    )

    if params is not None:
        asymptote = params[0]
        headroom = asymptote - macro_f1s[-1]
        logger.info(
            "  Asymptotic Macro F1: %.4f (headroom: %.4f)",
            asymptote,
            headroom,
        )
        if headroom < SATURATED_HEADROOM:
            logger.info(
                "  結論: 模型在目前架構下已接近飽和 (headroom < 0.01)，"
                "加更多資料幾乎無法提升 F1。"
            )
            logger.info("  建議: 考慮更大的模型架構、更好的特徵工程、或 ensemble。")
        elif headroom < LIMITED_HEADROOM:
            logger.info(
                "  結論: 剩餘提升空間有限 (headroom=%.4f)，"
                "加資料的邊際效益已明顯遞減。",
                headroom,
            )
            pred_400 = extrapolate_f1(params, ns[-1] + 400)
            logger.info(
                "  預估再加 400 張 raw 可提升 F1 約 %+.4f",
                min(pred_400 - macro_f1s[-1], headroom),
            )
        else:
            logger.info(
                "  結論: 仍有提升空間 (headroom=%.4f)，" "加更多資料可能有效。",
                headroom,
            )
            pred_400 = extrapolate_f1(params, ns[-1] + 400)
            logger.info(
                "  預估再加 400 張 raw 可提升 F1 約 %+.4f",
                min(pred_400 - macro_f1s[-1], headroom),
            )

    logger.info("=" * 90)


if __name__ == "__main__":
    main()

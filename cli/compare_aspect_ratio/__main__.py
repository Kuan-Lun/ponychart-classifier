"""
比較不同長寬比（正方形 vs 長方形）對訓練效果的影響。

PonyChart 原圖為 1004x554 (~1.813:1)，目前 production 將其壓縮為正方形。
此 CLI 評估保留原始長寬比是否能改善分類效果。

此 CLI 拆成兩種模式，方便把不同設定派到不同設備上跑：

1. 單跑一種設定並把結果寫成 JSON：
     uv run --extra train python -m cli.compare_aspect_ratio --run square_320
     uv run --extra train python -m cli.compare_aspect_ratio --run rect_238x431

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.compare_aspect_ratio --report

預設 results dir 是 ``results/compare_aspect_ratio/``，
可用 ``--results-dir`` 指定其他路徑。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``<label>__<hash12>.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。
"""

from __future__ import annotations

from pathlib import Path

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from cli.training_runner import prepare_training_setup, run_training_experiment
from ponychart_classifier.model_spec import ImageSize
from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    PerClassColumn,
    PerClassMetricsBlock,
    RunEnvironmentRow,
    get_transforms,
    is_significant_improvement,
    is_significant_regression,
    load_all_json_results,
    log_per_class_f1_matrix,
    log_per_class_precision_recall_blocks,
    log_run_environments,
    log_section,
    select_consistent_results,
)

from .configs import ASPECT_RATIO_CONFIGS
from .result import (
    ExperimentResult,
    measurements_to_result,
    parse_result_file,
    save_result,
)


def _ordered_labels(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded labels in canonical (config) order, then extras."""
    canonical = [lbl for lbl in ASPECT_RATIO_CONFIGS if lbl in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


def _fmt_size(size: ImageSize) -> str:
    """Format an ImageSize as ``'HxW'``."""
    return f"{size.height}x{size.width}"


def _pixel_count(size: ImageSize) -> int:
    return size.height * size.width


class CompareAspectRatioCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Compare square vs rectangular (native aspect-ratio) training. "
            "Use --run to train one config (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "compare_aspect_ratio"

    def available_keys(self) -> list[str]:
        return list(ASPECT_RATIO_CONFIGS.keys())

    def run_one(self, key: str, results_dir: Path) -> None:
        if key not in ASPECT_RATIO_CONFIGS:
            available = ", ".join(ASPECT_RATIO_CONFIGS.keys())
            msg = (
                f"Aspect-ratio config {key!r} not configured. "
                f"Currently configured: {available}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        config = ASPECT_RATIO_CONFIGS[key]

        setup = prepare_training_setup(self.logger)

        log_section(
            self.logger,
            "ASPECT RATIO: %s  (%s)",
            key,
            config.description,
            width=70,
        )
        self.logger.info(
            "  INPUT_SIZE=%s  pixels=%dK",
            _fmt_size(config.input_size),
            _pixel_count(config.input_size) // 1000,
        )
        self.logger.info("  Data hash: %s", setup.data_hash[:HASH_PREFIX_LEN])

        measurements = run_training_experiment(
            train_samples=setup.split.train,
            val_samples=setup.split.val,
            test_samples=setup.split.test,
            device=setup.device,
            num_workers=setup.num_workers,
            run_label=key,
            backbone=BACKBONE,
            input_size=config.input_size,
            batch_size=BATCH_SIZE,
            train_transform=get_transforms(is_train=True),
        )

        self.logger.info(
            ">> %s: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
            key,
            measurements.test_result.macro_f1,
            measurements.param_count // 1000,
            measurements.onnx_size_mb,
            measurements.train_time_s,
        )

        experiment = measurements_to_result(key, config, measurements, setup)
        out_path = save_result(experiment, results_dir)
        self.logger.info("Saved result to %s", out_path)

    def format_report(self, results_dir: Path) -> None:
        raw_results = load_all_json_results(results_dir, parse_result_file, self.logger)
        if not raw_results:
            msg = f"No results found in {results_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        results = select_consistent_results(
            raw_results,
            key=lambda r: r.label,
            key_label="aspect_ratio",
            logger=self.logger,
        )
        ordered = _ordered_labels(results)
        data_hash = next(iter(results.values())).data_hash

        log_section(self.logger, "ASPECT RATIO COMPARISON RESULTS")
        self.logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
        self.logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

        # Main comparison table
        self.logger.info("")
        self.logger.info(
            "  %-18s  %-12s  %-8s  %-10s  %-10s  %-10s  %-10s",
            "Config",
            "INPUT_SIZE",
            "Pixels",
            "Macro F1",
            "Params",
            "ONNX Size",
            "Time",
        )
        self.logger.info("  " + "-" * 88)

        for lbl in ordered:
            r = results[lbl]
            self.logger.info(
                "  %-18s  %-12s  %-8s  %-10.4f  %-10s  %-10s  %s",
                lbl,
                _fmt_size(r.input_size),
                f"{_pixel_count(r.input_size) // 1000}K",
                r.test_result.macro_f1,
                f"{r.param_count / 1e6:.1f}M",
                f"{r.onnx_size_mb:.1f}MB",
                f"{r.train_time_s:.0f}s",
            )

        # Run environment
        log_run_environments(
            self.logger,
            [
                RunEnvironmentRow(
                    label=lbl,
                    hostname=results[lbl].hostname,
                    device=results[lbl].device,
                )
                for lbl in ordered
            ],
        )

        # Per-class matrix (class rows, config columns, * = best)
        columns = [
            PerClassColumn(
                label=lbl,
                macro_f1=results[lbl].test_result.macro_f1,
                per_class_f1=list(results[lbl].test_result.per_class_f1),
            )
            for lbl in ordered
        ]
        log_per_class_f1_matrix(self.logger, list(CLASS_NAMES), columns)

        # Per-class precision/recall
        log_per_class_precision_recall_blocks(
            self.logger,
            list(CLASS_NAMES),
            [
                PerClassMetricsBlock(
                    label=f"{lbl} ({results[lbl].description})",
                    precision=results[lbl].test_result.per_class_precision,
                    recall=results[lbl].test_result.per_class_recall,
                    f1=results[lbl].test_result.per_class_f1,
                )
                for lbl in ordered
            ],
        )

        # Summary
        log_section(self.logger, "SUMMARY")

        # Separate square and rect configs for comparison
        square_labels = [lbl for lbl in ordered if lbl.startswith("square_")]
        rect_labels = [lbl for lbl in ordered if lbl.startswith("rect_")]

        best_lbl = max(ordered, key=lambda lbl: results[lbl].test_result.macro_f1)
        best_f1 = results[best_lbl].test_result.macro_f1
        self.logger.info("  Best config: %s (Macro F1=%.4f)", best_lbl, best_f1)

        self.logger.info("")
        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            diff = f1 - best_f1
            if lbl == best_lbl:
                self.logger.info("  * %s: F1=%.4f (BEST)", lbl, f1)
            else:
                self.logger.info("    %s: F1=%.4f (%+.4f vs best)", lbl, f1, diff)

        # Square vs rect comparison at similar pixel budget
        if square_labels and rect_labels:
            best_square = max(
                square_labels,
                key=lambda lbl: results[lbl].test_result.macro_f1,
            )
            best_rect = max(
                rect_labels,
                key=lambda lbl: results[lbl].test_result.macro_f1,
            )
            sq_f1 = results[best_square].test_result.macro_f1
            rc_f1 = results[best_rect].test_result.macro_f1
            delta = rc_f1 - sq_f1

            self.logger.info("")
            self.logger.info("  Square vs Rectangular:")
            self.logger.info("    Best square: %s (F1=%.4f)", best_square, sq_f1)
            self.logger.info("    Best rect:   %s (F1=%.4f)", best_rect, rc_f1)
            self.logger.info("    Delta: %+.4f", delta)

            if is_significant_improvement(delta):
                self.logger.info(
                    "  結論: 長方形有顯著改善 (%+.4f F1)，"
                    "建議更新 model_spec.py 的 INPUT_SIZE",
                    delta,
                )
            elif is_significant_regression(delta):
                self.logger.info(
                    "  結論: 正方形表現更好 (%+.4f F1)，維持現有設定",
                    delta,
                )
            else:
                self.logger.info(
                    "  結論: 兩者差異不大 (%+.4f F1)，" "考量相容性建議維持正方形",
                    delta,
                )

        self.logger.info("=" * 100)


if __name__ == "__main__":
    CompareAspectRatioCLI().main()

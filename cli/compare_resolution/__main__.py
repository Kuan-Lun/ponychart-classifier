"""
比較不同輸入解析度對訓練效果的影響。

此 CLI 拆成兩種模式，方便把不同解析度派到不同設備上跑：

1. 單跑一種解析度並把結果寫成 JSON：
     uv run --extra train python -m cli.compare_resolution --run 320
     uv run --extra train python -m cli.compare_resolution --run 448

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.compare_resolution --report

預設 results dir 是 ``results/compare_resolution/``，
可用 ``--results-dir`` 指定其他路徑。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``<input_size>px__<hash12>.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。

測試五種 (PRE_RESIZE, INPUT_SIZE) 組合：
  224: PRE_RESIZE=256, INPUT_SIZE=224
  288: PRE_RESIZE=320, INPUT_SIZE=288
  320: PRE_RESIZE=384, INPUT_SIZE=320  (current production)
  380: PRE_RESIZE=448, INPUT_SIZE=380
  448: PRE_RESIZE=512, INPUT_SIZE=448
"""

from __future__ import annotations

from pathlib import Path

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from cli.training_runner import prepare_training_setup, run_training_experiment
from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    load_all_json_results,
    log_section,
    select_consistent_results,
)

from .configs import RESOLUTION_CONFIGS
from .result import (
    ExperimentResult,
    measurements_to_result,
    parse_result_file,
    save_result,
)


def _ordered_labels(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded labels in canonical (config) order, then extras."""
    canonical = [lbl for lbl in RESOLUTION_CONFIGS if lbl in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


class CompareResolutionCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Train resolutions one-at-a-time and merge results across machines. "
            "Use --run to train one resolution (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "compare_resolution"

    def available_keys(self) -> list[str]:
        return list(RESOLUTION_CONFIGS.keys())

    def run_one(self, key: str, results_dir: Path) -> None:
        if key not in RESOLUTION_CONFIGS:
            available = ", ".join(RESOLUTION_CONFIGS.keys())
            msg = (
                f"Resolution {key!r} not configured. "
                f"Currently configured: {available}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        config = RESOLUTION_CONFIGS[key]

        setup = prepare_training_setup(self.logger)

        log_section(
            self.logger,
            "RESOLUTION: %spx  (PRE_RESIZE=%d  INPUT_SIZE=%d)",
            key,
            config.pre_resize,
            config.input_size,
            width=70,
        )
        self.logger.info("  Data hash: %s", setup.data_hash[:HASH_PREFIX_LEN])

        measurements = run_training_experiment(
            train_samples=setup.split.train,
            val_samples=setup.split.val,
            test_samples=setup.split.test,
            device=setup.device,
            num_workers=setup.num_workers,
            run_label=f"{key}px",
            backbone=BACKBONE,
            input_size=config.input_size,
            pre_resize=config.pre_resize,
            batch_size=BATCH_SIZE,
        )

        self.logger.info(
            ">> %spx: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
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
            key_label="resolution",
            logger=self.logger,
        )
        ordered = _ordered_labels(results)
        data_hash = next(iter(results.values())).data_hash

        log_section(self.logger, "RESOLUTION COMPARISON RESULTS")
        self.logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
        self.logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

        self.logger.info("")
        self.logger.info(
            "  %-12s  %-12s  %-12s  %-10s  %-10s  %-10s  %-10s",
            "Resolution",
            "PRE_RESIZE",
            "INPUT_SIZE",
            "Macro F1",
            "Params",
            "ONNX Size",
            "Time",
        )
        self.logger.info("  " + "-" * 82)

        baseline_label = ordered[-1]  # smallest resolution as baseline
        baseline_f1 = results[baseline_label].test_result.macro_f1

        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            self.logger.info(
                "  %-12s  %-12d  %-12d  %-10.4f  %-10s  %-10s  %s",
                f"{lbl}px",
                r.pre_resize,
                r.input_size,
                f1,
                f"{r.param_count / 1e6:.1f}M",
                f"{r.onnx_size_mb:.1f}MB",
                f"{r.train_time_s:.0f}s",
            )

        # Run environment
        self.logger.info("")
        self.logger.info("Run environment:")
        for lbl in ordered:
            r = results[lbl]
            self.logger.info(
                "  %-12s  host=%s  device=%s", f"{lbl}px", r.hostname, r.device
            )

        # Per-class F1
        self.logger.info("")
        self.logger.info("Per-class F1:")
        header = f"  {'Class':<20s}"
        for lbl in ordered:
            header += f"  {lbl + 'px':<14s}"
        self.logger.info(header)
        self.logger.info("  " + "-" * (20 + 16 * len(ordered)))

        for i, cls_name in enumerate(CLASS_NAMES):
            row = f"  {cls_name:<20s}"
            for lbl in ordered:
                f1 = results[lbl].test_result.per_class_f1[i]
                row += f"  {f1:<14.4f}"
            self.logger.info(row)

        # Per-class precision/recall
        self.logger.info("")
        self.logger.info("Per-class Precision / Recall:")
        for lbl in ordered:
            self.logger.info("  %spx:", lbl)
            tr = results[lbl].test_result
            for i, cls_name in enumerate(CLASS_NAMES):
                self.logger.info(
                    "    %-20s  P=%.4f  R=%.4f  F1=%.4f",
                    cls_name,
                    tr.per_class_precision[i],
                    tr.per_class_recall[i],
                    tr.per_class_f1[i],
                )

        # Summary
        log_section(self.logger, "SUMMARY")
        best_lbl = max(ordered, key=lambda lbl: results[lbl].test_result.macro_f1)
        best_f1 = results[best_lbl].test_result.macro_f1
        self.logger.info("  Best resolution: %spx (Macro F1=%.4f)", best_lbl, best_f1)

        self.logger.info("")
        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            diff = f1 - best_f1
            if lbl == best_lbl:
                self.logger.info("  * %spx: F1=%.4f (BEST)", lbl, f1)
            else:
                self.logger.info("    %spx: F1=%.4f (%+.4f vs best)", lbl, f1, diff)

        delta_vs_baseline = best_f1 - baseline_f1
        self.logger.info("")
        if best_lbl == baseline_label:
            self.logger.info(
                "  結論: 提高解析度沒有帶來改善，維持 %spx", baseline_label
            )
        elif delta_vs_baseline > 0.005:
            self.logger.info(
                "  結論: %spx 有顯著改善 (%+.4f F1)，建議更新 constants.py",
                best_lbl,
                delta_vs_baseline,
            )
        else:
            self.logger.info(
                "  結論: %spx 改善有限 (%+.4f F1)，考量訓練時間後不建議更換",
                best_lbl,
                delta_vs_baseline,
            )
        self.logger.info("=" * 80)


if __name__ == "__main__":
    CompareResolutionCLI().main()

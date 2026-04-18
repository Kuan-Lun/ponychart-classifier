"""
比較不同輸入解析度（以倍率縮放）對訓練效果的影響。

以 production INPUT_SIZE 為基準，乘以倍率得到實驗尺寸。
修改 model_spec.py 的 INPUT_SIZE 後，所有倍率自動對應新尺寸。

此 CLI 拆成兩種模式，方便把不同解析度派到不同設備上跑：

1. 單跑一種解析度並把結果寫成 JSON（``--run KEY``）：
     uv run --extra train python -m cli.compare_resolution --run 1.00x
     uv run --extra train python -m cli.compare_resolution --run 1.40x

   KEY 必須是 ``configs.py`` 中 ``RESOLUTION_CONFIGS`` 定義的倍率之一。
   目前可用的 KEY：``1.40x``, ``1.20x``, ``1.00x``, ``0.85x``, ``0.70x``。
   其中 ``1.00x`` 即 production ``INPUT_SIZE``（基準線）。

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.compare_resolution --report

預設 results dir 是 ``results/compare_resolution/``，
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
from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    PerClassColumn,
    PerClassPrecisionRecallBlock,
    RunEnvironmentRow,
    get_transforms,
    is_significant_improvement,
    load_all_json_results,
    log_per_class_f1_matrix,
    log_per_class_precision_recall,
    log_run_environments,
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


def _fmt_size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


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
            "RESOLUTION: %s  (scale=%.2f  INPUT_SIZE=%s)",
            key,
            config.scale,
            _fmt_size(config.input_size.hw()),
            width=70,
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
            "  %-10s  %-7s  %-12s  %-8s  %-10s  %-10s  %-10s  %-10s",
            "Config",
            "Scale",
            "INPUT_SIZE",
            "Pixels",
            "Macro F1",
            "Params",
            "ONNX Size",
            "Time",
        )
        self.logger.info("  " + "-" * 90)

        best_lbl = max(ordered, key=lambda lbl: results[lbl].test_result.macro_f1)
        best_f1 = results[best_lbl].test_result.macro_f1

        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            pixels = r.input_size.height * r.input_size.width
            self.logger.info(
                "  %-10s  %-7.2f  %-12s  %-8s  %-10.4f  %-10s  %-10s  %s",
                lbl,
                r.scale,
                _fmt_size(r.input_size.hw()),
                f"{pixels // 1000}K",
                f1,
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
        log_per_class_precision_recall(
            self.logger,
            list(CLASS_NAMES),
            [
                PerClassPrecisionRecallBlock(
                    label=f"{lbl} ({results[lbl].scale:.2f}x, "
                    f"{_fmt_size(results[lbl].input_size.hw())})",
                    precision=results[lbl].test_result.per_class_precision,
                    recall=results[lbl].test_result.per_class_recall,
                )
                for lbl in ordered
            ],
        )

        # Summary
        log_section(self.logger, "SUMMARY")
        self.logger.info("  Best resolution: %s (Macro F1=%.4f)", best_lbl, best_f1)

        self.logger.info("")
        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            diff = f1 - best_f1
            if lbl == best_lbl:
                self.logger.info("  * %s: F1=%.4f (BEST)", lbl, f1)
            else:
                self.logger.info("    %s: F1=%.4f (%+.4f vs best)", lbl, f1, diff)

        # Compare against 1.00x baseline
        baseline_lbl = "1.00x"
        if baseline_lbl in results:
            baseline_f1 = results[baseline_lbl].test_result.macro_f1
            delta_vs_baseline = best_f1 - baseline_f1
            self.logger.info("")
            if best_lbl == baseline_lbl:
                self.logger.info("  結論: 放大縮小解析度沒有帶來改善，維持 1.00x")
            elif is_significant_improvement(delta_vs_baseline):
                self.logger.info(
                    "  結論: %s 有顯著改善 (%+.4f F1)，建議更新 constants.py",
                    best_lbl,
                    delta_vs_baseline,
                )
            else:
                self.logger.info(
                    "  結論: %s 改善有限 (%+.4f F1)，考量訓練時間後不建議更換",
                    best_lbl,
                    delta_vs_baseline,
                )
        self.logger.info("=" * 80)


if __name__ == "__main__":
    CompareResolutionCLI().main()

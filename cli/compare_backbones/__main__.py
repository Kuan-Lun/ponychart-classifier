"""
比較不同 backbone 架構對 PonyChart 分類效果的影響。

此 CLI 拆成兩種模式，方便把不同 backbone 派到不同設備上跑：

1. 單跑一個 backbone 並把結果寫成 JSON：
     uv run --extra train python -m cli.compare_backbones --run efficientnet_b0
     uv run --extra train python -m cli.compare_backbones --run efficientnet_b4

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.compare_backbones --report

預設 results dir 是 ``results/compare_backbones/``，
可用 ``--results-dir`` 指定其他路徑。

## 跨機器一致性

JSON 檔名包含「資料集 hash」(``<backbone>__<hash12>.json``)，
hash 是從所有 sample (path + labels) 計算出來的指紋。
- 同一份 ``rawimage`` + ``labels.json`` → 同 hash → 同檔名 → 重跑會覆蓋舊版
- 任何新增 / 刪除 / 標籤異動 → 不同 hash → 不同檔名 → 兩者並存
- ``--report`` 偵測到 results dir 同時存在多個 hash 時會直接報錯，
  避免不小心比較不同資料快照下的數字。

要加新的 backbone，只要在 ``BACKBONE_CONFIGS`` 補一筆即可。
"""

from __future__ import annotations

from pathlib import Path

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from cli.training_runner import prepare_training_setup, run_training_experiment
from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    get_transforms,
    load_all_json_results,
    log_section,
    select_consistent_results,
)

from .configs import BACKBONE_CONFIGS
from .result import (
    ExperimentResult,
    measurements_to_result,
    parse_result_file,
    save_result,
)


def _ordered_backbones(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded backbone names in canonical (config) order, then extras."""
    canonical = [name for name in BACKBONE_CONFIGS if name in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


class CompareBackbonesCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Train backbones one-at-a-time and merge results across machines. "
            "Use --run to train one backbone (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "compare_backbones"

    def available_keys(self) -> list[str]:
        return list(BACKBONE_CONFIGS.keys())

    def run_one(self, key: str, results_dir: Path) -> None:
        if key not in BACKBONE_CONFIGS:
            available = ", ".join(BACKBONE_CONFIGS.keys())
            msg = (
                f"Backbone {key!r} not configured. Add it to "
                f"BACKBONE_CONFIGS first. Currently configured: {available}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        config = BACKBONE_CONFIGS[key]

        setup = prepare_training_setup(self.logger)

        backbone_meta = BACKBONE_REGISTRY[key]
        log_section(self.logger, "BACKBONE: %s", backbone_meta.description, width=70)
        self.logger.info(
            "  Resolution: input=%s  pre_resize=%s  batch_size=%d",
            config.input_size,
            config.pre_resize,
            config.batch_size,
        )
        self.logger.info("  Data hash: %s", setup.data_hash[:HASH_PREFIX_LEN])

        measurements = run_training_experiment(
            train_samples=setup.split.train,
            val_samples=setup.split.val,
            test_samples=setup.split.test,
            device=setup.device,
            num_workers=setup.num_workers,
            run_label=key,
            backbone=key,
            input_size=config.input_size,
            pre_resize=config.pre_resize,
            batch_size=config.batch_size,
            train_transform=get_transforms(is_train=True, input_size=config.input_size),
        )

        self.logger.info(
            ">> %s @ %s: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB  time=%.0fs",
            key,
            config.input_size,
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
            key=lambda r: r.backbone_name,
            key_label="backbone",
            logger=self.logger,
        )
        ordered_names = _ordered_backbones(results)
        data_hash = next(iter(results.values())).data_hash

        log_section(self.logger, "BACKBONE COMPARISON RESULTS")
        self.logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
        self.logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

        self.logger.info("")
        self.logger.info(
            "  %-22s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s",
            "Backbone",
            "Input",
            "Macro F1",
            "Params",
            "ONNX Size",
            "Time",
            "Thresholds",
        )
        self.logger.info("  " + "-" * 92)

        for name in ordered_names:
            r = results[name]
            tr = r.test_result
            thr_str = " ".join(f"{t:.2f}" for t in r.thresholds)
            inp_str = f"{r.input_size.height}x{r.input_size.width}"
            self.logger.info(
                "  %-22s  %-8s  %-10.4f  %-10s  %-10s  %-10s  %s",
                name,
                inp_str,
                tr.macro_f1,
                f"{r.param_count / 1e6:.1f}M",
                f"{r.onnx_size_mb:.1f}MB",
                f"{r.train_time_s:.0f}s",
                thr_str,
            )

        # Per-backbone environment
        self.logger.info("")
        self.logger.info("Run environment:")
        for name in ordered_names:
            r = results[name]
            self.logger.info("  %-22s  host=%s  device=%s", name, r.hostname, r.device)

        # Per-class F1 table
        self.logger.info("")
        self.logger.info("Per-class F1:")
        header = f"  {'Class':<20s}"
        for name in ordered_names:
            header += f"  {name:<22s}"
        self.logger.info(header)
        self.logger.info("  " + "-" * (20 + 24 * len(ordered_names)))

        for i, cls_name in enumerate(CLASS_NAMES):
            row = f"  {cls_name:<20s}"
            for name in ordered_names:
                f1 = results[name].test_result.per_class_f1[i]
                row += f"  {f1:<22.4f}"
            self.logger.info(row)

        # Per-class precision/recall
        self.logger.info("")
        self.logger.info("Per-class Precision / Recall:")
        for name in ordered_names:
            self.logger.info("  %s:", name)
            tr = results[name].test_result
            for i, cls_name in enumerate(CLASS_NAMES):
                self.logger.info(
                    "    %-20s  P=%.4f  R=%.4f  F1=%.4f",
                    cls_name,
                    tr.per_class_precision[i],
                    tr.per_class_recall[i],
                    tr.per_class_f1[i],
                )

        # Recommendation
        log_section(self.logger, "RECOMMENDATION")

        best_name = max(
            ordered_names,
            key=lambda n: results[n].test_result.macro_f1,
        )
        best_f1 = results[best_name].test_result.macro_f1

        self.logger.info("  Best backbone: %s (Macro F1=%.4f)", best_name, best_f1)
        self.logger.info("")

        for name in ordered_names:
            r = results[name]
            f1 = r.test_result.macro_f1
            diff = f1 - best_f1
            if name == best_name:
                self.logger.info("  * %s: F1=%.4f (BEST)", name, f1)
            else:
                self.logger.info("    %s: F1=%.4f (%+.4f vs best)", name, f1, diff)

        # Efficiency analysis
        self.logger.info("")
        self.logger.info("  Efficiency analysis:")
        baseline_name = ordered_names[0]
        baseline = results[baseline_name]
        baseline_f1 = baseline.test_result.macro_f1
        for name in ordered_names:
            r = results[name]
            f1 = r.test_result.macro_f1
            gain = f1 - baseline_f1
            size_ratio = r.onnx_size_mb / baseline.onnx_size_mb
            time_ratio = r.train_time_s / baseline.train_time_s
            self.logger.info(
                "    %s: F1 %+.4f vs %s, %.1fx size, %.1fx time",
                name,
                gain,
                baseline_name,
                size_ratio,
                time_ratio,
            )
        self.logger.info(
            "  (time ratios only meaningful when both runs were on the same" " device)"
        )

        self.logger.info("")
        self.logger.info("  To use the best backbone for production training:")
        self.logger.info("    uv run python scripts/train.py --backbone %s", best_name)
        self.logger.info("=" * 90)


if __name__ == "__main__":
    CompareBackbonesCLI().main()

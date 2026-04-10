"""
比較不同輸入解析度對訓練效果的影響。

此 CLI 拆成兩種模式，方便把不同解析度派到不同設備上跑：

1. 單跑一種解析度並把結果寫成 JSON：
     uv run --extra train python -m cli.compare_resolution --run 320
     uv run --extra train python -m cli.compare_resolution --run 448

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.compare_resolution --report

預設 results dir 是 ``cli/compare_resolution/resolution_results/``，
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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from cli.training_runner import (
    SplitDict,
    TestResultDict,
    TrainingMeasurements,
    TrainingSetup,
    eval_result_from_dict,
    eval_result_to_dict,
    prepare_training_setup,
    run_training_experiment,
)
from ponychart_classifier.training import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HASH_PREFIX_LEN,
    SEED,
    EnvDict,
    EvalResult,
    load_all_json_results,
    log_section,
    select_consistent_results,
)

# ---------------------------------------------------------------------------
# Resolution configs — ordered from highest to lowest so OOM surfaces early
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionConfig:
    """Per-resolution training/eval settings."""

    pre_resize: int
    input_size: int


RESOLUTION_CONFIGS: dict[str, ResolutionConfig] = {
    "448": ResolutionConfig(pre_resize=512, input_size=448),
    "380": ResolutionConfig(pre_resize=448, input_size=380),
    "320": ResolutionConfig(pre_resize=384, input_size=320),
    "288": ResolutionConfig(pre_resize=320, input_size=288),
    "224": ResolutionConfig(pre_resize=256, input_size=224),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a single resolution experiment."""

    label: str
    pre_resize: int
    input_size: int
    test_result: EvalResult
    thresholds: list[float]
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    train_size: int
    val_size: int
    test_size: int
    seed: int
    data_hash: str
    hostname: str
    device: str


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class ExperimentDict(TypedDict):
    label: str
    pre_resize: int
    input_size: int
    param_count: int
    onnx_size_mb: float
    train_time_s: float
    thresholds: list[float]
    test_result: TestResultDict
    split: SplitDict
    seed: int
    data_hash: str
    env: EnvDict


def experiment_to_dict(exp: ExperimentResult) -> ExperimentDict:
    return ExperimentDict(
        label=exp.label,
        pre_resize=exp.pre_resize,
        input_size=exp.input_size,
        param_count=exp.param_count,
        onnx_size_mb=exp.onnx_size_mb,
        train_time_s=exp.train_time_s,
        thresholds=list(exp.thresholds),
        test_result=eval_result_to_dict(exp.test_result),
        split=SplitDict(
            train_size=exp.train_size,
            val_size=exp.val_size,
            test_size=exp.test_size,
        ),
        seed=exp.seed,
        data_hash=exp.data_hash,
        env=EnvDict(hostname=exp.hostname, device=exp.device),
    )


def experiment_from_dict(data: ExperimentDict) -> ExperimentResult:
    split = data["split"]
    env = data["env"]
    return ExperimentResult(
        label=data["label"],
        pre_resize=data["pre_resize"],
        input_size=data["input_size"],
        test_result=eval_result_from_dict(data["test_result"]),
        thresholds=list(data["thresholds"]),
        param_count=data["param_count"],
        onnx_size_mb=data["onnx_size_mb"],
        train_time_s=data["train_time_s"],
        train_size=split["train_size"],
        val_size=split["val_size"],
        test_size=split["test_size"],
        seed=data["seed"],
        data_hash=data["data_hash"],
        hostname=env["hostname"],
        device=env["device"],
    )


def _parse_experiment_json(raw: str) -> ExperimentDict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(ExperimentDict, parsed)


def result_filename(label: str, data_hash: str) -> str:
    return f"{label}px__{data_hash[:HASH_PREFIX_LEN]}.json"


def save_result(exp: ExperimentResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / result_filename(exp.label, exp.data_hash)
    out_path.write_text(json.dumps(experiment_to_dict(exp), indent=2))
    return out_path


def _parse_result_file(raw: str) -> ExperimentResult:
    return experiment_from_dict(_parse_experiment_json(raw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measurements_to_result(
    label: str,
    config: ResolutionConfig,
    m: TrainingMeasurements,
    setup: TrainingSetup,
) -> ExperimentResult:
    return ExperimentResult(
        label=label,
        pre_resize=config.pre_resize,
        input_size=config.input_size,
        test_result=m.test_result,
        thresholds=m.thresholds,
        param_count=m.param_count,
        onnx_size_mb=m.onnx_size_mb,
        train_time_s=m.train_time_s,
        train_size=len(setup.split.train),
        val_size=len(setup.split.val),
        test_size=len(setup.split.test),
        seed=SEED,
        data_hash=setup.data_hash,
        hostname=m.hostname,
        device=m.device_label,
    )


def _ordered_labels(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded labels in canonical (config) order, then extras."""
    canonical = [lbl for lbl in RESOLUTION_CONFIGS if lbl in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

        experiment = _measurements_to_result(key, config, measurements, setup)
        out_path = save_result(experiment, results_dir)
        self.logger.info("Saved result to %s", out_path)

    def format_report(self, results_dir: Path) -> None:
        raw_results = load_all_json_results(
            results_dir, _parse_result_file, self.logger
        )
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

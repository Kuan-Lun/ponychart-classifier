"""
分析空間增強（水平翻轉、垂直翻轉、旋轉角度）對訓練效果的影響。

透過 ablation study 逐一啟用各空間增強，比較與 baseline 的 F1 差異，
判斷哪些增強有正面效果、最佳旋轉角度為何。

此 CLI 拆成兩種模式，方便把不同實驗派到不同設備上跑：

1. 單跑一組增強設定並把結果寫成 JSON：
     uv run --extra train python -m cli.analyze_augmentations --run hflip
     uv run --extra train python -m cli.analyze_augmentations --run rot90

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run --extra train python -m cli.analyze_augmentations --report

預設 results dir 是 ``results/analyze_augmentations/``，
可用 ``--results-dir`` 指定其他路徑。

七組實驗（僅空間增強不同，其餘 augmentation 皆相同）：
  1. none    — 無翻轉、無旋轉（baseline）
  2. hflip   — + 水平翻轉
  3. vflip   — + 垂直翻轉
  4. rot15   — + 旋轉 15 度
  5. rot45   — + 旋轉 45 度
  6. rot90   — + 旋轉 90 度
  7. current — 水平翻轉 + 垂直翻轉 + 旋轉 90 度（目前 train.py 設定）
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
    INPUT_SIZE,
    load_all_json_results,
    log_section,
    select_consistent_results,
)

from .configs import AUG_CONFIGS, build_train_transform
from .result import (
    ExperimentResult,
    measurements_to_result,
    parse_result_file,
    save_result,
)


def _ordered_labels(loaded: dict[str, ExperimentResult]) -> list[str]:
    """Return loaded labels in canonical (config) order, then extras."""
    canonical = [lbl for lbl in AUG_CONFIGS if lbl in loaded]
    extras = sorted(set(loaded) - set(canonical))
    return canonical + extras


class AnalyzeAugmentationsCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Ablation study for spatial augmentations (flip / rotation). "
            "Use --run to train one config (writes JSON), then --report to "
            "print a comparison from all collected JSON files."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "analyze_augmentations"

    def available_keys(self) -> list[str]:
        return list(AUG_CONFIGS.keys())

    def run_one(self, key: str, results_dir: Path) -> None:
        if key not in AUG_CONFIGS:
            available = ", ".join(AUG_CONFIGS.keys())
            msg = (
                f"Augmentation config {key!r} not configured. "
                f"Currently configured: {available}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        config = AUG_CONFIGS[key]

        setup = prepare_training_setup(self.logger)

        log_section(
            self.logger,
            "AUGMENTATION: %s (%s)",
            key,
            config.description,
            width=70,
        )
        self.logger.info("  Data hash: %s", setup.data_hash[:HASH_PREFIX_LEN])

        train_tf = build_train_transform(config)
        measurements = run_training_experiment(
            train_samples=setup.split.train,
            val_samples=setup.split.val,
            test_samples=setup.split.test,
            device=setup.device,
            num_workers=setup.num_workers,
            run_label=key,
            backbone=BACKBONE,
            input_size=INPUT_SIZE,
            batch_size=BATCH_SIZE,
            train_transform=train_tf,
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
            key_label="augmentation",
            logger=self.logger,
        )
        ordered = _ordered_labels(results)
        data_hash = next(iter(results.values())).data_hash
        test_size = next(iter(results.values())).test_size

        # ── Macro F1 overview ──
        log_section(
            self.logger,
            "AUGMENTATION ABLATION RESULTS (test set, %d images)",
            test_size,
        )
        self.logger.info("  Loaded %d result(s) from %s", len(results), results_dir)
        self.logger.info("  Data snapshot: %s", data_hash[:HASH_PREFIX_LEN])

        baseline_f1 = results["none"].test_result.macro_f1 if "none" in results else 0.0
        baseline_per_class = (
            results["none"].test_result.per_class_f1 if "none" in results else []
        )

        self.logger.info("")
        self.logger.info(
            "  %-12s  %-10s  %-10s  %s",
            "Experiment",
            "Macro F1",
            "Delta",
            "Config",
        )
        self.logger.info("  " + "-" * 75)
        for lbl in ordered:
            r = results[lbl]
            f1 = r.test_result.macro_f1
            delta = f1 - baseline_f1
            delta_str = f"{delta:+.4f}" if lbl != "none" else "baseline"
            cfg = AUG_CONFIGS.get(lbl)
            desc = cfg.description if cfg else lbl
            self.logger.info("  %-12s  %-10.4f  %-10s  %s", lbl, f1, delta_str, desc)

        # Per-class F1 table
        self.logger.info("")
        self.logger.info("  Per-class F1 (delta vs baseline):")
        header = "    %-20s" + "  %-12s" * len(ordered)
        self.logger.info(header, "Class", *ordered)
        self.logger.info("    " + "-" * (20 + 14 * len(ordered)))
        for i, name in enumerate(CLASS_NAMES):
            row_parts = [f"    {name:<20s}"]
            for lbl in ordered:
                f1 = results[lbl].test_result.per_class_f1[i]
                if lbl == "none" or not baseline_per_class:
                    row_parts.append(f"  {f1:<12.4f}")
                else:
                    delta = f1 - baseline_per_class[i]
                    row_parts.append(f"  {f1:.4f}{delta:+.3f}")
            self.logger.info("".join(row_parts))

        # Run environment
        self.logger.info("")
        self.logger.info("  Run environment:")
        for lbl in ordered:
            r = results[lbl]
            self.logger.info("    %-12s  host=%s  device=%s", lbl, r.hostname, r.device)

        # ── Flip analysis ──
        self._report_flip_analysis(results, baseline_f1, baseline_per_class)

        # ── Rotation analysis ──
        best_rot_name, best_rot_f1 = self._report_rotation_analysis(
            results, baseline_f1, baseline_per_class
        )

        # ── Combined effect analysis ──
        self._report_combined_analysis(results, baseline_f1, best_rot_f1)

        # ── Recommendation ──
        self._report_recommendation(
            results, ordered, baseline_f1, best_rot_name, best_rot_f1
        )

    # ------------------------------------------------------------------
    # Report sub-sections
    # ------------------------------------------------------------------

    def _report_flip_analysis(
        self,
        results: dict[str, ExperimentResult],
        baseline_f1: float,
        baseline_per_class: list[float],
    ) -> None:
        log_section(self.logger, "FLIP ANALYSIS")

        hflip_delta = (
            results["hflip"].test_result.macro_f1 - baseline_f1
            if "hflip" in results
            else 0.0
        )
        vflip_delta = (
            results["vflip"].test_result.macro_f1 - baseline_f1
            if "vflip" in results
            else 0.0
        )
        self.logger.info("  HFlip effect:  %+.4f", hflip_delta)
        self.logger.info("  VFlip effect:  %+.4f", vflip_delta)

        for label, delta in [("水平翻轉", hflip_delta), ("垂直翻轉", vflip_delta)]:
            if delta > 0.005:
                self.logger.info("  >> %s有正面效果，建議保留", label)
            elif delta < -0.005:
                self.logger.info("  >> %s有負面效果，建議移除", label)
            else:
                self.logger.info(
                    "  >> %s效果不明顯 (%.4f)，可考慮移除以簡化 pipeline",
                    label,
                    delta,
                )

        if baseline_per_class and "hflip" in results and "vflip" in results:
            self.logger.info("")
            self.logger.info("  Per-class flip impact (F1 delta vs baseline):")
            self.logger.info("  %-20s  %-12s  %-12s", "Class", "HFlip", "VFlip")
            for i, name in enumerate(CLASS_NAMES):
                hd = (
                    results["hflip"].test_result.per_class_f1[i] - baseline_per_class[i]
                )
                vd = (
                    results["vflip"].test_result.per_class_f1[i] - baseline_per_class[i]
                )
                self.logger.info("  %-20s  %+.4f       %+.4f", name, hd, vd)

    def _report_rotation_analysis(
        self,
        results: dict[str, ExperimentResult],
        baseline_f1: float,
        baseline_per_class: list[float],
    ) -> tuple[str, float]:
        log_section(self.logger, "ROTATION ANALYSIS")

        rot_configs = [("rot15", 15), ("rot45", 45), ("rot90", 90)]
        best_rot_name = "none"
        best_rot_f1 = baseline_f1
        for rname, deg in rot_configs:
            if rname not in results:
                continue
            rot_f1 = results[rname].test_result.macro_f1
            delta = rot_f1 - baseline_f1
            self.logger.info("  Rotation %3d: %+.4f (F1=%.4f)", deg, delta, rot_f1)
            if rot_f1 > best_rot_f1:
                best_rot_f1 = rot_f1
                best_rot_name = rname

        if best_rot_name == "none":
            self.logger.info("  >> 所有旋轉角度皆無正面效果，建議移除旋轉增強")
        else:
            best_deg = {"rot15": 15, "rot45": 45, "rot90": 90}[best_rot_name]
            delta = best_rot_f1 - baseline_f1
            self.logger.info("  >> 最佳旋轉角度: %d (%+.4f F1)", best_deg, delta)
            if delta > 0.005:
                self.logger.info("  >> 建議使用 %d 旋轉", best_deg)
            else:
                self.logger.info("  >> 效果有限 (%.4f)，旋轉非必要", delta)

        if baseline_per_class:
            self.logger.info("")
            self.logger.info("  Per-class rotation impact (F1 delta vs baseline):")
            self.logger.info("  %-20s  %-10s  %-10s  %-10s", "Class", "15", "45", "90")
            for i, name in enumerate(CLASS_NAMES):
                deltas = []
                for rname, _ in rot_configs:
                    if rname in results:
                        d = (
                            results[rname].test_result.per_class_f1[i]
                            - baseline_per_class[i]
                        )
                        deltas.append(d)
                    else:
                        deltas.append(0.0)
                self.logger.info("  %-20s  %+.4f     %+.4f     %+.4f", name, *deltas)

        return best_rot_name, best_rot_f1

    def _report_combined_analysis(
        self,
        results: dict[str, ExperimentResult],
        baseline_f1: float,
        best_rot_f1: float,
    ) -> None:
        if "current" not in results:
            return

        log_section(self.logger, "COMBINED EFFECT ANALYSIS")

        hflip_delta = (
            results["hflip"].test_result.macro_f1 - baseline_f1
            if "hflip" in results
            else 0.0
        )
        vflip_delta = (
            results["vflip"].test_result.macro_f1 - baseline_f1
            if "vflip" in results
            else 0.0
        )

        current_delta = results["current"].test_result.macro_f1 - baseline_f1
        sum_individual = hflip_delta + vflip_delta + (best_rot_f1 - baseline_f1)
        interaction = current_delta - sum_individual

        self.logger.info("  Current config (HFlip+VFlip+Rot90): %+.4f", current_delta)
        self.logger.info("  Sum of individual effects:           %+.4f", sum_individual)
        self.logger.info("  Interaction effect:                  %+.4f", interaction)
        if interaction > 0.005:
            self.logger.info("  >> 組合使用有正向交互作用，建議同時啟用")
        elif interaction < -0.005:
            self.logger.info("  >> 組合使用有負向交互作用，建議精簡增強組合")
        else:
            self.logger.info("  >> 交互作用微小，各增強可獨立決定是否啟用")

    def _report_recommendation(
        self,
        results: dict[str, ExperimentResult],
        ordered: list[str],
        baseline_f1: float,
        best_rot_name: str,
        best_rot_f1: float,
    ) -> None:
        log_section(self.logger, "RECOMMENDATION")

        best_name = max(ordered, key=lambda k: results[k].test_result.macro_f1)
        best_result = results[best_name]
        self.logger.info(
            "  最佳實驗: %s (Macro F1=%.4f)",
            best_name,
            best_result.test_result.macro_f1,
        )
        self.logger.info("")

        hflip_delta = (
            results["hflip"].test_result.macro_f1 - baseline_f1
            if "hflip" in results
            else 0.0
        )
        vflip_delta = (
            results["vflip"].test_result.macro_f1 - baseline_f1
            if "vflip" in results
            else 0.0
        )

        recommended_parts: list[str] = []
        if hflip_delta > 0.003:
            recommended_parts.append("RandomHorizontalFlip(p=0.5)")
        if vflip_delta > 0.003:
            recommended_parts.append("RandomVerticalFlip(p=0.5)")
        if best_rot_name != "none" and (best_rot_f1 - baseline_f1) > 0.003:
            best_deg = {"rot15": 15, "rot45": 45, "rot90": 90}[best_rot_name]
            recommended_parts.append(f"RandomAffine(degrees={best_deg})")

        if recommended_parts:
            self.logger.info("  建議在 train.py 中使用的空間增強:")
            for part in recommended_parts:
                self.logger.info("    - %s", part)
        else:
            self.logger.info("  建議移除所有空間增強（翻轉與旋轉），僅保留:")
            self.logger.info("    - RandomCrop")
            self.logger.info("    - ColorJitter")
            self.logger.info("    - GaussianBlur")
            self.logger.info("    - RandomErasing")

        # Compare with current config
        if "current" in results:
            self.logger.info("")
            diff = (
                best_result.test_result.macro_f1
                - results["current"].test_result.macro_f1
            )
            if abs(diff) < 0.003:
                self.logger.info("  目前設定 (current) 已接近最佳，無需調整")
            elif diff > 0:
                self.logger.info("  切換至 '%s' 可提升 F1 約 %+.4f", best_name, diff)
            else:
                self.logger.info("  目前設定 (current) 即為最佳或接近最佳")

        self.logger.info("=" * 90)


if __name__ == "__main__":
    AnalyzeAugmentationsCLI().main()

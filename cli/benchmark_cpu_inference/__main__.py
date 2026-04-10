"""
比較不同 backbone 在 CPU 上的單張推論延遲。

使用 random-init 模型 + onnxruntime CPU 後端，測量單張前向傳播的延遲。
權重值不影響 FLOPs 或 latency，所以不需要載入訓練好的 checkpoint——
這個 benchmark 量的是「架構在這台 CPU 上跑多快」，純架構成本。

使用方式：

1. 單跑一個 backbone 並把結果寫成 JSON：
     uv run python -m cli.benchmark_cpu_inference --run efficientnet_b0
     uv run python -m cli.benchmark_cpu_inference --run efficientnet_b4

2. 讀取 results dir 內的 JSON 並印對照表：
     uv run python -m cli.benchmark_cpu_inference --report

額外選項 (僅 ``--run`` 時有效)：
     --warmup 20 --iters 200 --threads 4 --input-size 320
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    INPUT_SIZE,
)

from .benchmark import print_comparison, run_benchmark
from .configs import BACKBONE_INPUT_SIZE
from .result import BenchmarkResult, parse_result_file, save_result


class BenchmarkCpuCLI(ExperimentCLI):
    def experiment_name(self) -> str:
        return (
            "Benchmark single-image CPU inference latency for one or more "
            "backbones using a random-init model exported to ONNX."
        )

    def default_results_dir(self) -> Path:
        return RESULTS_ROOT / "benchmark_cpu_inference"

    def available_keys(self) -> list[str]:
        return list(BACKBONE_REGISTRY.keys())

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--warmup",
            type=int,
            default=5,
            help="Warmup iterations before timing (default: 5)",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=50,
            help="Timed iterations (default: 50)",
        )
        parser.add_argument(
            "--input-size",
            type=int,
            default=None,
            help=(
                "Override input size for the backbone. "
                "Default: per-backbone native (see BACKBONE_INPUT_SIZE)."
            ),
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=1,
            help=(
                "onnxruntime intra-op threads (default: 1 for"
                " reproducibility). Use 0 to let onnxruntime decide."
            ),
        )

    def run_one(self, key: str, results_dir: Path) -> None:
        args = self._args
        threads: int | None = args.threads if args.threads > 0 else None
        input_size: int = args.input_size or BACKBONE_INPUT_SIZE.get(key, INPUT_SIZE)

        self.logger.info("CPU inference benchmark")
        self.logger.info(
            "  warmup=%d  iters=%d  threads=%s",
            args.warmup,
            args.iters,
            threads if threads is not None else "auto",
        )

        result = run_benchmark(
            key,
            input_size=input_size,
            warmup=args.warmup,
            iters=args.iters,
            intra_op_threads=threads,
            logger=self.logger,
        )

        out_path = save_result(result, results_dir)
        self.logger.info("Saved result to %s", out_path)

    def format_report(self, results_dir: Path) -> None:
        if not results_dir.is_dir():
            msg = f"Results directory not found: {results_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        result_files = sorted(results_dir.glob("*.json"))
        if not result_files:
            msg = f"No result JSONs found in {results_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        results: list[BenchmarkResult] = []
        for path in result_files:
            try:
                results.append(parse_result_file(path.read_text()))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                self.logger.warning("Skipping %s: %s", path.name, exc)

        if not results:
            msg = f"No valid results found in {results_dir}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Sort by canonical BACKBONE_REGISTRY order, then alphabetically
        registry_order = list(BACKBONE_REGISTRY.keys())

        def sort_key(r: BenchmarkResult) -> tuple[int, str]:
            idx = (
                registry_order.index(r.backbone_name)
                if r.backbone_name in registry_order
                else len(registry_order)
            )
            return (idx, r.backbone_name)

        results.sort(key=sort_key)
        print_comparison(results, self.logger)


if __name__ == "__main__":
    BenchmarkCpuCLI().main()

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
import logging
import socket
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from cli.experiment import RESULTS_ROOT, ExperimentCLI
from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    INPUT_SIZE,
    build_model,
    export_onnx,
)

# 每個 backbone 預設使用的推論輸入解析度。
# 跟 cli/compare_backbones 的 BACKBONE_CONFIGS 一致：
# B0 用 production 設定 320，其他用各自原生 (B4=380)。
BACKBONE_INPUT_SIZE: dict[str, int] = {
    "mobilenet_v3_small": 224,
    "mobilenet_v3_large": 224,
    "efficientnet_b0": 320,
    "efficientnet_b2": 260,
    "efficientnet_b4": 380,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency stats for one backbone."""

    backbone_name: str
    input_size: int
    onnx_size_mb: float
    n_iters: int
    warmup: int
    threads: int | None
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    hostname: str


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class BenchmarkResultDict(TypedDict):
    backbone_name: str
    input_size: int
    onnx_size_mb: float
    n_iters: int
    warmup: int
    threads: int | None
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    hostname: str


def result_to_dict(r: BenchmarkResult) -> BenchmarkResultDict:
    return BenchmarkResultDict(
        backbone_name=r.backbone_name,
        input_size=r.input_size,
        onnx_size_mb=r.onnx_size_mb,
        n_iters=r.n_iters,
        warmup=r.warmup,
        threads=r.threads,
        mean_ms=r.mean_ms,
        p50_ms=r.p50_ms,
        p95_ms=r.p95_ms,
        p99_ms=r.p99_ms,
        hostname=r.hostname,
    )


def result_from_dict(data: BenchmarkResultDict) -> BenchmarkResult:
    return BenchmarkResult(
        backbone_name=data["backbone_name"],
        input_size=data["input_size"],
        onnx_size_mb=data["onnx_size_mb"],
        n_iters=data["n_iters"],
        warmup=data["warmup"],
        threads=data["threads"],
        mean_ms=data["mean_ms"],
        p50_ms=data["p50_ms"],
        p95_ms=data["p95_ms"],
        p99_ms=data["p99_ms"],
        hostname=data["hostname"],
    )


def _parse_result_json(raw: str) -> BenchmarkResultDict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return cast(BenchmarkResultDict, parsed)


def _parse_result_file(raw: str) -> BenchmarkResult:
    return result_from_dict(_parse_result_json(raw))


def save_result(r: BenchmarkResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{r.backbone_name}.json"
    out_path.write_text(json.dumps(result_to_dict(r), indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def export_to_temp_onnx(backbone_name: str, input_size: int) -> Path:
    """Build a fresh model and export it to a temp ONNX file."""
    model = build_model(backbone=backbone_name, pretrained=False)
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        out_path = Path(f.name)
    export_onnx(model, out_path, input_size=input_size)
    return out_path


def cleanup_onnx(onnx_path: Path) -> None:
    """Remove an ONNX file plus any external-data sidecar."""
    onnx_path.unlink(missing_ok=True)
    Path(str(onnx_path) + ".data").unlink(missing_ok=True)


def benchmark_onnx(
    onnx_path: Path,
    input_size: int,
    *,
    warmup: int,
    iters: int,
    intra_op_threads: int | None,
) -> tuple[float, float, float, float]:
    """Run *iters* timed forward passes and return (mean, p50, p95, p99) ms."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    if intra_op_threads is not None:
        sess_options.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((1, 3, input_size, input_size), dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    mean = statistics.fmean(samples_ms)
    p50 = samples_ms[len(samples_ms) // 2]
    p95 = samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.95))]
    p99 = samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.99))]
    return mean, p50, p95, p99


def run_benchmark(
    backbone_name: str,
    *,
    input_size: int,
    warmup: int,
    iters: int,
    intra_op_threads: int | None,
    logger: logging.Logger,
) -> BenchmarkResult:
    """Build, export, time, and clean up one backbone."""
    if backbone_name not in BACKBONE_REGISTRY:
        available = ", ".join(BACKBONE_REGISTRY.keys())
        msg = f"Unknown backbone {backbone_name!r}. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("--- %s @ %d ---", backbone_name, input_size)
    onnx_path = export_to_temp_onnx(backbone_name, input_size)
    try:
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        mean, p50, p95, p99 = benchmark_onnx(
            onnx_path,
            input_size,
            warmup=warmup,
            iters=iters,
            intra_op_threads=intra_op_threads,
        )
    finally:
        cleanup_onnx(onnx_path)

    result = BenchmarkResult(
        backbone_name=backbone_name,
        input_size=input_size,
        onnx_size_mb=size_mb,
        n_iters=iters,
        warmup=warmup,
        threads=intra_op_threads,
        mean_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        hostname=socket.gethostname(),
    )
    logger.info(
        "  ONNX %.1fMB  mean=%.1fms  p50=%.1fms  p95=%.1fms  p99=%.1fms",
        size_mb,
        mean,
        p50,
        p95,
        p99,
    )
    return result


def _print_comparison(results: list[BenchmarkResult], logger: logging.Logger) -> None:
    """Print a side-by-side comparison table and ratios."""
    logger.info("")
    logger.info("=" * 86)
    logger.info("CPU INFERENCE LATENCY")
    logger.info("=" * 86)
    logger.info(
        "  %-22s  %-6s  %-9s  %-9s  %-9s  %-9s  %-9s",
        "Backbone",
        "Input",
        "ONNX",
        "Mean",
        "p50",
        "p95",
        "p99",
    )
    logger.info("  " + "-" * 84)
    for r in results:
        logger.info(
            "  %-22s  %-6d  %-9s  %-9s  %-9s  %-9s  %-9s",
            r.backbone_name,
            r.input_size,
            f"{r.onnx_size_mb:.1f}MB",
            f"{r.mean_ms:.1f}ms",
            f"{r.p50_ms:.1f}ms",
            f"{r.p95_ms:.1f}ms",
            f"{r.p99_ms:.1f}ms",
        )

    if len(results) > 1:
        baseline = results[0]
        logger.info("")
        logger.info("  Latency vs %s (mean):", baseline.backbone_name)
        for r in results:
            ratio = r.mean_ms / baseline.mean_ms
            marker = "*" if r is baseline else " "
            logger.info("   %s %-22s  %.2fx", marker, r.backbone_name, ratio)

    # Run environment
    logger.info("")
    logger.info("Run environment:")
    for r in results:
        logger.info(
            "  %-22s  host=%s  iters=%d  warmup=%d  threads=%s",
            r.backbone_name,
            r.hostname,
            r.n_iters,
            r.warmup,
            r.threads if r.threads is not None else "auto",
        )
    logger.info("=" * 86)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
                results.append(_parse_result_file(path.read_text()))
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
        _print_comparison(results, self.logger)


if __name__ == "__main__":
    BenchmarkCpuCLI().main()

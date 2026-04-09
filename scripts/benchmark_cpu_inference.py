"""
比較不同 backbone 在 CPU 上的單張推論延遲。

使用 random-init 模型 + onnxruntime CPU 後端，測量單張前向傳播的延遲。
權重值不影響 FLOPs 或 latency，所以不需要載入訓練好的 checkpoint——
這個 benchmark 量的是「架構在這台 CPU 上跑多快」，純架構成本。

預設量 ``efficientnet_b0`` vs ``efficientnet_b4``，對應
:mod:`scripts.compare_backbones` 主要在比較的兩個。

使用方式：
  uv run python scripts/benchmark_cpu_inference.py
  uv run python scripts/benchmark_cpu_inference.py --backbone efficientnet_b0 efficientnet_b4
  uv run python scripts/benchmark_cpu_inference.py --iters 200 --warmup 20
  uv run python scripts/benchmark_cpu_inference.py --threads 4
"""

from __future__ import annotations

import argparse
import logging
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    INPUT_SIZE,
    build_model,
    export_onnx,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BACKBONES = ["efficientnet_b0", "efficientnet_b4"]

# 每個 backbone 預設使用的推論輸入解析度。
# 跟 scripts/compare_backbones.py 的 BACKBONE_RESOLUTION 一致：
# B0 用 production 設定 320，其他用各自原生 (B4=380)。
BACKBONE_INPUT_SIZE: dict[str, int] = {
    "mobilenet_v3_small": 224,
    "mobilenet_v3_large": 224,
    "efficientnet_b0": 320,
    "efficientnet_b2": 260,
    "efficientnet_b4": 380,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency stats for one backbone."""

    backbone_name: str
    input_size: int
    onnx_size_mb: float
    n_iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


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
) -> BenchmarkResult:
    """Build, export, time, and clean up one backbone.

    Raises :class:`ValueError` for an unknown backbone so the caller
    can decide how to surface the error (the CLI ``main`` translates
    it to a non-zero exit code).
    """
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
        mean_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
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


def print_comparison(results: list[BenchmarkResult]) -> None:
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
    logger.info("=" * 86)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-image CPU inference latency for one or more "
            "backbones using a random-init model exported to ONNX. Defaults "
            "to comparing efficientnet_b0 vs efficientnet_b4."
        ),
    )
    parser.add_argument(
        "--backbone",
        nargs="+",
        default=DEFAULT_BACKBONES,
        metavar="NAME",
        help=(
            "One or more backbone names from BACKBONE_REGISTRY. "
            f"Default: {' '.join(DEFAULT_BACKBONES)}"
        ),
    )
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
            "Override input size for ALL backbones. "
            "Default: per-backbone native (see BACKBONE_INPUT_SIZE)."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "onnxruntime intra-op threads (default: 1 for reproducibility). "
            "Use 0 to let onnxruntime decide."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threads: int | None = args.threads if args.threads > 0 else None

    logger.info("CPU inference benchmark")
    logger.info(
        "  warmup=%d  iters=%d  threads=%s",
        args.warmup,
        args.iters,
        threads if threads is not None else "auto",
    )

    results: list[BenchmarkResult] = []
    for name in args.backbone:
        input_size = args.input_size or BACKBONE_INPUT_SIZE.get(name, INPUT_SIZE)
        results.append(
            run_benchmark(
                name,
                input_size=input_size,
                warmup=args.warmup,
                iters=args.iters,
                intra_op_threads=threads,
            )
        )

    print_comparison(results)


if __name__ == "__main__":
    main()

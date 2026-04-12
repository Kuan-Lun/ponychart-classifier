"""ONNX export and CPU inference timing logic."""

from __future__ import annotations

import logging
import socket
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from ponychart_classifier.model_spec import ImageSize
from ponychart_classifier.training import (
    BACKBONE_REGISTRY,
    build_model,
    export_onnx,
)

from .result import BenchmarkResult


def export_to_temp_onnx(backbone_name: str, input_size: ImageSize) -> Path:
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
    input_size: ImageSize,
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
    dummy = rng.standard_normal((1, 3, *input_size.hw()), dtype=np.float32)

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
    input_size: ImageSize,
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

    logger.info("--- %s @ %s ---", backbone_name, input_size)
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


def print_comparison(results: list[BenchmarkResult], logger: logging.Logger) -> None:
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
        inp_str = f"{r.input_size.height}x{r.input_size.width}"
        logger.info(
            "  %-22s  %-6s  %-9s  %-9s  %-9s  %-9s  %-9s",
            r.backbone_name,
            inp_str,
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

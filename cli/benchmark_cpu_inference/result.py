"""BenchmarkResult type and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


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


def parse_result_file(raw: str) -> BenchmarkResult:
    return result_from_dict(_parse_result_json(raw))


def save_result(r: BenchmarkResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{r.backbone_name}.json"
    out_path.write_text(json.dumps(result_to_dict(r), indent=2))
    return out_path

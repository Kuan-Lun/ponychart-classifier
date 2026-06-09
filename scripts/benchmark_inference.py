"""推論效能 benchmark：量測各階段耗時並比較 ONNX provider。

Usage:
    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --n 100 --batch-sizes 1 8 16
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort

from ponychart_classifier.inference import artifacts
from ponychart_classifier.inference.preprocessing import preprocess_bgr
from ponychart_classifier.model_spec import ImageSize

# ── 參數 ─────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_DIR = Path(__file__).parent.parent / "rawimage"
DEFAULT_N = 50
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
WARMUP = 3


def _find_images(image_dir: Path, n: int) -> list[Path]:
    paths = [
        Path(p)
        for p in glob.glob(str(image_dir / "**" / "*"), recursive=True)
        if Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        and Path(p).is_file()
    ]
    if not paths:
        raise RuntimeError(f"找不到圖片：{image_dir}")
    paths.sort()
    return paths[:n]


def _get_input_size(session: ort.InferenceSession) -> ImageSize:
    shape = session.get_inputs()[0].shape
    return ImageSize(height=int(shape[2]), width=int(shape[3]))


# ── 計時工具 ──────────────────────────────────────────────────────────────────


class _Timer:
    def __init__(self) -> None:
        self._samples: list[float] = []

    def __enter__(self) -> _Timer:
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self._samples.append(time.perf_counter() - self._t)

    @property
    def mean_ms(self) -> float:
        return (
            1000.0 * sum(self._samples) / len(self._samples) if self._samples else 0.0
        )

    @property
    def total_ms(self) -> float:
        return 1000.0 * sum(self._samples)


# ── 單張逐一推論（baseline） ──────────────────────────────────────────────────


def _bench_sequential(
    session: ort.InferenceSession,
    images: list[np.ndarray[object, np.dtype[np.uint8]]],
    input_size: ImageSize,
) -> dict[str, float]:
    """逐張推論，回傳各階段平均耗時（ms）。"""
    input_name: str = session.get_inputs()[0].name
    t_pre = _Timer()
    t_inf = _Timer()

    for _ in range(WARMUP):
        tensor = preprocess_bgr(images[0], input_size=input_size)
        session.run(None, {input_name: tensor})

    for img in images:
        with t_pre:
            tensor = preprocess_bgr(img, input_size=input_size)
        with t_inf:
            session.run(None, {input_name: tensor})

    return {
        "preprocess_ms": t_pre.mean_ms,
        "inference_ms": t_inf.mean_ms,
        "total_ms": t_pre.mean_ms + t_inf.mean_ms,
    }


# ── 批次推論 ──────────────────────────────────────────────────────────────────


def _bench_batch(
    session: ort.InferenceSession,
    images: list[np.ndarray[object, np.dtype[np.uint8]]],
    input_size: ImageSize,
    batch_size: int,
) -> dict[str, float]:
    """批次推論，回傳每張平均耗時（ms）。"""
    input_name: str = session.get_inputs()[0].name
    t_pre = _Timer()
    t_inf = _Timer()

    batches: list[np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        tensors = np.concatenate(
            [preprocess_bgr(img, input_size=input_size) for img in chunk], axis=0
        )
        batches.append(tensors)

    for _ in range(WARMUP):
        session.run(None, {input_name: batches[0]})

    for batch in batches:
        n = batch.shape[0]
        with t_pre:
            pass  # 預處理已在上面做完，這裡只量 inference
        with t_inf:
            session.run(None, {input_name: batch})
        # 手動把時間除以張數以得到 per-image 值
        t_inf._samples[-1] /= n

    return {
        "inference_ms_per_image": t_inf.mean_ms,
    }


# ── IO 讀檔 ───────────────────────────────────────────────────────────────────


def _bench_io(paths: list[Path]) -> float:
    t = _Timer()
    for p in paths[:WARMUP]:
        cv.imread(str(p), cv.IMREAD_COLOR)
    for p in paths:
        with t:
            cv.imread(str(p), cv.IMREAD_COLOR)
    return t.mean_ms


# ── 主程式 ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="測試圖片數量")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        metavar="B",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    args = parser.parse_args()

    print(f"載入 {args.n} 張圖片...")
    paths = _find_images(args.image_dir, args.n)

    print("讀取圖片到記憶體（排除 IO 時間影響）...")
    t_io = _bench_io(paths)
    loaded = [cv.imread(str(p), cv.IMREAD_COLOR) for p in paths]
    images: list[np.ndarray[object, np.dtype[np.uint8]]] = [
        img for img in loaded if img is not None
    ]
    print(f"  cv.imread 平均耗時：{t_io:.2f} ms/張\n")

    artifacts.ensure_artifact(artifacts.DEFAULT_MODEL_PATH, artifacts.MODEL_FILENAME)
    model_path = str(artifacts.DEFAULT_MODEL_PATH)

    available = ort.get_available_providers()
    providers_to_test: list[tuple[str, list[str]]] = []
    if "CoreMLExecutionProvider" in available:
        providers_to_test.append(
            ("CoreML", ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        )
    if "CUDAExecutionProvider" in available:
        providers_to_test.append(
            ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        )
    providers_to_test.append(("CPU only", ["CPUExecutionProvider"]))

    print(f"可用 providers：{available}\n")

    # 先偵測 model 是否支援 dynamic batch（N != 1）
    _probe_session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    _probe_shape = _probe_session.get_inputs()[0].shape
    model_supports_batch = not isinstance(_probe_shape[0], int) or _probe_shape[0] != 1
    del _probe_session

    if not model_supports_batch:
        print("注意：目前 ONNX model 的 batch size 硬編為 1，跳過批次測試。")
        print("      重新 export 時使用 dynamic_axes={'input': {0: 'N'}} 即可啟用。\n")

    print(f"{'Provider':<14} {'preprocess':>12} {'inference(b=1)':>16}", end="")
    if model_supports_batch:
        for bs in args.batch_sizes:
            if bs > 1:
                print(f"  {'inf/img(b=' + str(bs) + ')':>14}", end="")
    print()
    print("-" * 44)

    for label, providers in providers_to_test:
        session = ort.InferenceSession(model_path, providers=providers)
        input_size = _get_input_size(session)

        seq = _bench_sequential(session, images, input_size)
        row = (
            f"{label:<14}"
            f"  {seq['preprocess_ms']:>10.2f} ms"
            f"  {seq['inference_ms']:>12.2f} ms"
        )

        if model_supports_batch:
            for bs in args.batch_sizes:
                if bs == 1:
                    continue
                if bs > len(images):
                    row += f"  {'(skip)':>14}"
                    continue
                bat = _bench_batch(session, images, input_size, bs)
                row += f"  {bat['inference_ms_per_image']:>12.2f} ms"

        print(row)

    print()
    print(f"測試張數：{len(images)}，warmup：{WARMUP} 張")
    print("inference 時間已從記憶體讀取的圖片開始量，不含 cv.imread 的 IO 時間。")


if __name__ == "__main__":
    main()

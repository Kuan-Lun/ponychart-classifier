"""ONNX model export utilities."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .constants import INPUT_SIZE

logger = logging.getLogger(__name__)

# External loggers that emit verbose messages during ONNX export.
_NOISY_LOGGERS = (
    "onnxruntime",
    "onnx",
    "onnx_ir",
    "onnxscript",
    "torch.onnx",
)


def export_onnx(model: nn.Module, output_path: Path) -> None:
    """Export a PyTorch model to ONNX format."""
    import warnings

    import onnx

    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    saved_levels: dict[str, int] = {}
    for name in _NOISY_LOGGERS:
        ext_logger = logging.getLogger(name)
        saved_levels[name] = ext_logger.level
        ext_logger.setLevel(logging.WARNING)
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.filterwarnings(
                "ignore", message="Missing annotation for parameter"
            )
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.onnx.export(
                model_cpu,
                (dummy,),
                str(output_path),
                input_names=["input"],
                output_names=["logits"],
                opset_version=18,
            )
    finally:
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
    # Merge external data into single file if needed
    external_data = Path(str(output_path) + ".data")
    if external_data.exists():
        onnx_model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(
            onnx_model,
            str(output_path),
            save_as_external_data=False,
        )
        external_data.unlink()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported: %s (%.1f MB)", output_path, size_mb)

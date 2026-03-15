# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PonyChart Classifier is a multi-label image classifier for HentaiVerse PonyChart characters (6 classes). It uses transfer learning with ImageNet-pretrained backbones, trains in PyTorch, and exports to ONNX for CPU-only inference.

## Commands

```bash
# Install (runtime only)
uv pip install .

# Install for development (with training dependencies)
uv pip install -e ".[train]"

# Run training
uv run scripts/train.py
uv run scripts/train.py --from-scratch  # ignore existing checkpoint

# Label images (Tkinter GUI)
uv run scripts/label_images.py

# Type checking
uv run mypy src/ scripts/

# Linting
uv run ruff check src/ scripts/

# Build for PyPI
pip install ".[publish]"
python -m build
twine upload dist/*
```

There is no test suite. Quality is enforced via MyPy (strict mode) and Ruff.

## Architecture

### Inference (`src/ponychart_classifier/`)

- `model_spec.py` — Shared constants: 6 classes, input size 320×320, pre-resize 384×384, ImageNet normalization stats
- `inference.py` — `PonyChartClassifier` class with lazy-loaded ONNX session. Public API: `predict(img_path)` and `preload()`
- Ships bundled `model.onnx` and `thresholds.json` as package data

### Training (`src/ponychart_classifier/training/`)

- `constants.py` — **Single source of truth** for all hyperparameters (backbone, LR, epochs, patience, thresholds)
- `model.py` — `BACKBONE_REGISTRY` with 4 architectures (MobileNetV3-Small/Large, EfficientNet-B0/B2); replaces final classifier for NUM_CLASSES
- `training.py` — Two-phase pipeline: Phase 1 freezes backbone (head-only), Phase 2 fine-tunes everything with early stopping
- `dataset.py` — Memory-aware image caching with shared memory for DataLoader workers
- `splitting.py` — MD5 hash-based stable train/val split that prevents data leakage across resume sessions
- `sampling.py` — Separates original vs. cropped images by filename pattern; balances crops to match original class distribution
- `export.py` — ONNX export (opset 18)

### Key Design Decisions

- **Resume training** auto-detects checkpoint compatibility and triggers from-scratch retraining if new data ratio exceeds 5% or validation set size increases
- **Hash-based splitting** groups samples by timestamp so related crops stay together; assignment is deterministic regardless of dataset size
- **Threshold optimization** tunes per-class sigmoid thresholds on validation data for multi-label F1

## Design Principles

- Follow SOLID principles: single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion

## Code Style

- **Formatter:** Black (line-length 88)
- **Linter:** Ruff (rules: E, F, I, UP)
- **Type checker:** MyPy strict mode; external stubs ignored for numpy, torch, cv2, onnxruntime, etc. (see `.mypy.ini`)
- Python 3.11+

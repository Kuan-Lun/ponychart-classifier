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

# Label images (Tkinter GUI)
uv run python -m app.label_images

# Type checking
uv run mypy src/ app/

# Linting
uv run ruff check src/ app/

```

There is no test suite. Quality is enforced via MyPy (strict mode) and Ruff.

## Project Structure

When you need to understand the directory layout, run `tree -I '__pycache__|*.egg-info|.venv|rawimage|checkpoints' -L 3` instead of maintaining a static listing here.

## Key Design Decisions

- `training/constants.py` is the **single source of truth** for all hyperparameters
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

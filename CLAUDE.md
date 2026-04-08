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

## Running Python

- Always use `uv run python` to run scripts, tests, or ad-hoc snippets (use `uv run --extra train python` when training dependencies are needed).

## Code Style

- **Sync obligation for tooling configuration:** the IDE save pipeline and the Stop hook pipeline are kept in lockstep across the locations below. Any change to one of them requires matching updates to the others in the same change.
  - Python formatting/lint/type-check: [.vscode/settings.json](.vscode/settings.json) (`[python]` block), the `[tool.ruff]` section of [pyproject.toml](pyproject.toml), [mypy.ini](mypy.ini), and [.claude/hooks/finalize-python.sh](.claude/hooks/finalize-python.sh).
  - Markdown formatting: [.vscode/settings.json](.vscode/settings.json) (`[markdown]` block) and [.claude/hooks/finalize-markdown.sh](.claude/hooks/finalize-markdown.sh).
  - Tool versions: the `dev` group of `[project.optional-dependencies]` in [pyproject.toml](pyproject.toml) pins `black`, `ruff`, `mypy`, and `pymarkdownlnt`. Both the IDE pipeline (when invoked via `uv run`) and the Stop hooks resolve to these venv-installed versions, so bumping any of them must be done here — not via Homebrew or any other system-wide install.
- Python version range: refer to `requires-python` in [pyproject.toml](pyproject.toml)

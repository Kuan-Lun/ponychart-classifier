"""Cross-machine experiment result I/O.

Shared infrastructure for scripts that train one config at a time on
different machines and merge the results from JSON files. Used by both
``scripts/compare_backbones.py`` (keyed by backbone name) and
``scripts/search_batch_lr.py`` (keyed by batch size).

Responsibilities:

- Stable data fingerprinting via :func:`hash_samples`, so results from
  the same data snapshot are grouped and incompatible snapshots are
  refused.
- Generic JSON loading with fail-fast on parse errors
  (:func:`load_all_json_results`).
- Consistency checks: single data hash, no duplicate keys
  (:func:`select_consistent_results`).
- Human-readable device labels (:func:`describe_device`) for
  cross-machine result tables where wall-clock numbers must be
  interpreted alongside the hardware they came from.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar

import torch

from .sampling import Sample

# Short hash for filenames; 48 bits, ample for our scale.
HASH_PREFIX_LEN = 12


class EnvDict(TypedDict):
    """JSON shape for run environment metadata."""

    hostname: str
    device: str


class HasDataHash(Protocol):
    """Protocol for results that carry a data-snapshot fingerprint.

    The attribute is exposed as a read-only ``@property`` so the protocol
    matches both regular attributes and ``frozen=True`` dataclass fields.
    """

    @property
    def data_hash(self) -> str: ...


T = TypeVar("T")
R = TypeVar("R", bound=HasDataHash)
K = TypeVar("K", bound=Hashable)


def hash_samples(samples: list[Sample]) -> str:
    """Return a stable SHA-256 hex digest of the sample set.

    Two machines that load the exact same ``rawimage`` + ``labels.json``
    will produce the same hash. Any addition, deletion, or label change
    flips the hash, which makes it safe to use as a "data snapshot ID"
    for grouping experiment results.
    """
    h = hashlib.sha256()
    for path, labels in sorted(samples):
        h.update(path.encode("utf-8"))
        h.update(b"\x00")
        h.update(",".join(str(label) for label in labels).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def describe_device(device: torch.device) -> str:
    """Return a human-readable device label, e.g. ``cuda:0 (NVIDIA L4)``."""
    device_type = str(device.type)
    if device_type == "cuda":
        idx = device.index if device.index is not None else 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return device_type


def load_all_json_results(
    results_dir: Path,
    parser: Callable[[str], T],
    logger: logging.Logger,
) -> list[T]:
    """Load every ``*.json`` in *results_dir* and parse with *parser*.

    Fails fast on parse errors rather than skipping bad files: a partial
    report would silently omit results and might push the reader to a
    wrong conclusion.
    """
    if not results_dir.is_dir():
        msg = f"Results directory does not exist: {results_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    results: list[T] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            results.append(parser(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            msg = (
                f"Failed to load result file {path}: {exc}\n"
                "  This may be a stale result from an older script version. "
                "Delete it (or move it elsewhere) and re-run the corresponding "
                "--run, then try --report again."
            )
            for line in msg.splitlines():
                logger.error(line)
            raise RuntimeError(msg) from exc
    return results


def select_consistent_results(
    results: list[R],
    *,
    key: Callable[[R], K],
    key_label: str,
    logger: logging.Logger,
) -> dict[K, R]:
    """Group *results* by *key* after enforcing a single data hash.

    Aborts the program if results come from more than one data snapshot,
    or if any *key* value appears more than once. *key_label* is used in
    error messages (e.g. ``"backbone"`` or ``"batch_size"``).
    """
    hashes_to_keys: dict[str, list[K]] = {}
    for r in results:
        hashes_to_keys.setdefault(r.data_hash, []).append(key(r))

    if len(hashes_to_keys) > 1:
        lines = [
            f"Loaded results were trained on {len(hashes_to_keys)} "
            "different data snapshots — refusing to produce a misleading "
            "comparison.",
        ]
        for h, keys in sorted(hashes_to_keys.items()):
            keys_str = ", ".join(str(k) for k in sorted(keys, key=str))
            lines.append(f"  {h[:HASH_PREFIX_LEN]}: {key_label}={keys_str}")
        lines.append(
            "Resolution: keep only the JSONs from one data snapshot in the "
            "results dir, then re-run --report."
        )
        for line in lines:
            logger.error(line)
        raise RuntimeError("\n".join(lines))

    by_key: dict[K, R] = {}
    for r in results:
        k = key(r)
        if k in by_key:
            msg = (
                f"Duplicate result for {key_label}={k} at the same data "
                "hash. This shouldn't happen via --run; remove one of the "
                "JSON files."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        by_key[k] = r
    return by_key

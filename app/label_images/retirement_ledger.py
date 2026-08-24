"""Persistent once-only receipts for post-cutoff source retirement."""

import dataclasses
import json
from pathlib import Path

from ponychart_classifier.image_names import parse_image_name

from .atomic_io import atomic_write, restore_file

_LEDGER_VERSION = 1


@dataclasses.dataclass(frozen=True)
class RetirementLedgerSnapshot:
    """In-memory and persisted state used for ordinary-exception rollback."""

    source_stems: frozenset[str]
    usable: bool
    persisted: bytes | None


class RetirementReceiptLedger:
    """Record source stems whose first post-cutoff save already completed.

    A malformed existing ledger fails closed: all retirement is disabled instead of
    risking a second deletion. Normal label saving remains available.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._source_stems: set[str] = set()
        self._usable = True
        self._load()

    @property
    def file_path(self) -> Path:
        """Metadata path used by the crash-recovery transaction journal."""
        return self._path

    @property
    def usable(self) -> bool:
        return self._usable

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: object = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != _LEDGER_VERSION:
                raise ValueError("Unsupported retirement ledger")
            values = raw.get("completed_source_stems")
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError("Invalid retirement receipts")
            sources = set(values)
            if not all(self._is_source_stem(source) for source in sources):
                raise ValueError("Invalid retirement source stem")
            self._source_stems = sources
        except OSError, ValueError, TypeError, json.JSONDecodeError:
            self._source_stems.clear()
            self._usable = False

    @staticmethod
    def _is_source_stem(value: str) -> bool:
        parsed = parse_image_name(value)
        return parsed is not None and parsed.is_original and parsed.source_stem == value

    def contains(self, source_stem: str) -> bool:
        """Return True after completion; unusable ledgers conservatively return True."""
        return not self._usable or source_stem in self._source_stems

    def record(self, source_stem: str) -> None:
        if not self._usable:
            raise RuntimeError("Retirement receipt ledger is malformed")
        if not self._is_source_stem(source_stem):
            raise ValueError(f"Invalid source stem: {source_stem}")
        self._source_stems.add(source_stem)

    def snapshot(self) -> RetirementLedgerSnapshot:
        return RetirementLedgerSnapshot(
            source_stems=frozenset(self._source_stems),
            usable=self._usable,
            persisted=self._path.read_bytes() if self._path.exists() else None,
        )

    def restore(self, snapshot: RetirementLedgerSnapshot) -> None:
        self.restore_memory(snapshot)
        restore_file(self._path, snapshot.persisted)

    def restore_memory(self, snapshot: RetirementLedgerSnapshot) -> None:
        """Restore state after journal recovery has already restored disk bytes."""
        self._source_stems = set(snapshot.source_stems)
        self._usable = snapshot.usable

    def save(self) -> None:
        if not self._usable:
            raise RuntimeError("Retirement receipt ledger is malformed")
        payload = json.dumps(
            {
                "version": _LEDGER_VERSION,
                "completed_source_stems": sorted(self._source_stems),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        atomic_write(self._path, payload)

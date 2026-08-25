"""Durable write-ahead journal for one-for-one sample retirement."""

import base64
import binascii
import dataclasses
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

from ponychart_classifier.image_names import parse_image_name

from .atomic_io import atomic_write, durable_mkdir, fsync_directory, restore_file
from .source_identity import parse_key_identity, parse_path_identity

_JOURNAL_VERSION = 2
_TRANSACTION_ID_RE = re.compile(r"[0-9a-f]{32}")
_PHASE_PREPARED = "prepared"
_PHASE_COMMITTED = "committed"


class RetirementRecoveryError(RuntimeError):
    """A pending journal cannot be trusted or safely recovered."""


@dataclasses.dataclass(frozen=True)
class RetirementJournal:
    """Validated on-disk retirement transaction state."""

    transaction_id: str
    source_stem: str
    candidate_paths: tuple[PurePosixPath, ...]
    labels_before: bytes | None
    phase: str


def journal_path_for(image_dir: Path) -> Path:
    """Return the fixed journal path outside the scanned image tree."""
    return image_dir.parent / f".{image_dir.name}-retirement-journal.json"


def staging_path_for(image_dir: Path, transaction_id: str) -> Path:
    """Return the staging path derived solely from a validated transaction ID."""
    if _TRANSACTION_ID_RE.fullmatch(transaction_id) is None:
        raise RetirementRecoveryError("Invalid retirement transaction ID")
    return image_dir.parent / f".{image_dir.name}-retirement-{transaction_id}"


def _encode_snapshot(snapshot: bytes | None) -> str | None:
    if snapshot is None:
        return None
    return base64.b64encode(snapshot).decode("ascii")


def _decode_snapshot(value: object, field: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RetirementRecoveryError(f"Invalid {field} journal snapshot")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise RetirementRecoveryError(f"Invalid {field} journal snapshot") from error


def _validate_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RetirementRecoveryError("Invalid retirement candidate path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RetirementRecoveryError("Unsafe retirement candidate path")
    if path.as_posix() != value:
        raise RetirementRecoveryError("Non-canonical retirement candidate path")
    return path


def _payload(journal: RetirementJournal, image_dir: Path) -> bytes:
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    return json.dumps(
        {
            "version": _JOURNAL_VERSION,
            "transaction_id": journal.transaction_id,
            "staging_dir": staging_dir.name,
            "source_stem": journal.source_stem,
            "candidate_paths": [path.as_posix() for path in journal.candidate_paths],
            "labels_before": _encode_snapshot(journal.labels_before),
            "phase": journal.phase,
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def _load(image_dir: Path) -> RetirementJournal:
    path = journal_path_for(image_dir)
    if path.is_symlink() or not path.is_file():
        raise RetirementRecoveryError("Unsafe retirement journal file")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetirementRecoveryError("Unreadable retirement journal") from error
    expected_fields = {
        "version",
        "transaction_id",
        "staging_dir",
        "source_stem",
        "candidate_paths",
        "labels_before",
        "phase",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise RetirementRecoveryError("Malformed retirement journal")
    if raw.get("version") != _JOURNAL_VERSION:
        raise RetirementRecoveryError("Unsupported retirement journal version")
    transaction_id = raw.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or _TRANSACTION_ID_RE.fullmatch(transaction_id) is None
    ):
        raise RetirementRecoveryError("Invalid retirement transaction ID")
    expected_staging = staging_path_for(image_dir, transaction_id).name
    if raw.get("staging_dir") != expected_staging:
        raise RetirementRecoveryError("Inconsistent retirement staging path")
    source_stem = raw.get("source_stem")
    parsed_source = (
        parse_image_name(source_stem) if isinstance(source_stem, str) else None
    )
    if (
        parsed_source is None
        or not parsed_source.is_original
        or parsed_source.source_stem != source_stem
    ):
        raise RetirementRecoveryError("Invalid retirement source stem")
    raw_paths = raw.get("candidate_paths")
    if not isinstance(raw_paths, list):
        raise RetirementRecoveryError("Invalid retirement candidate list")
    candidate_paths = tuple(_validate_relative_path(value) for value in raw_paths)
    if len(set(candidate_paths)) != len(candidate_paths) or candidate_paths != tuple(
        sorted(candidate_paths, key=lambda item: item.as_posix())
    ):
        raise RetirementRecoveryError("Duplicate or unordered candidate paths")
    parsed_candidates = tuple(
        parse_path_identity(Path(relative.as_posix())) for relative in candidate_paths
    )
    if any(parsed is None for parsed in parsed_candidates):
        raise RetirementRecoveryError("Unrecognized retirement candidate")
    recognized_candidates = tuple(
        parsed for parsed in parsed_candidates if parsed is not None
    )
    candidate_sources = {parsed.source_stem for parsed in recognized_candidates}
    if len(candidate_sources) > 1 or source_stem in candidate_sources:
        raise RetirementRecoveryError("Inconsistent retirement candidate sources")
    if candidate_paths and not any(
        parsed.is_original for parsed in recognized_candidates
    ):
        raise RetirementRecoveryError("Retirement candidate has no original image")
    phase = raw.get("phase")
    if phase not in {_PHASE_PREPARED, _PHASE_COMMITTED}:
        raise RetirementRecoveryError("Invalid retirement journal phase")
    return RetirementJournal(
        transaction_id=transaction_id,
        source_stem=source_stem,
        candidate_paths=candidate_paths,
        labels_before=_decode_snapshot(raw.get("labels_before"), "labels"),
        phase=phase,
    )


def _ensure_path_beneath(base: Path, path: Path) -> None:
    """Reject lexical traversal and symlink-parent escapes before mutation."""
    try:
        path.parent.resolve(strict=False).relative_to(base.resolve(strict=False))
    except (OSError, ValueError) as error:
        raise RetirementRecoveryError(f"Path escapes recovery root: {path}") from error


def _candidate_locations(
    image_dir: Path, journal: RetirementJournal
) -> list[tuple[Path, Path]]:
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    locations: list[tuple[Path, Path]] = []
    for relative in journal.candidate_paths:
        original = image_dir.joinpath(*relative.parts)
        staged = staging_dir.joinpath(*relative.parts)
        _ensure_path_beneath(image_dir, original)
        _ensure_path_beneath(staging_dir, staged)
        locations.append((original, staged))
    return locations


def _validate_staging_contents(
    staging_dir: Path,
    locations: list[tuple[Path, Path]],
) -> None:
    if staging_dir.is_symlink():
        raise RetirementRecoveryError("Retirement staging path is a symlink")
    if staging_dir.exists() and not staging_dir.is_dir():
        raise RetirementRecoveryError("Retirement staging path is not a directory")
    expected = {staged for _, staged in locations}
    if not staging_dir.exists():
        return
    for path in staging_dir.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and path not in expected):
            raise RetirementRecoveryError("Unexpected retirement staging content")


def _validate_metadata_paths(image_dir: Path, label_file: Path) -> None:
    expected_parent = image_dir.resolve(strict=True)
    try:
        parent = label_file.parent.resolve(strict=True)
    except OSError as error:
        raise RetirementRecoveryError("Metadata directory is unavailable") from error
    if parent != expected_parent or label_file.is_symlink():
        raise RetirementRecoveryError("Unsafe retirement metadata path")


def _validate_committed_metadata(journal: RetirementJournal, label_file: Path) -> None:
    try:
        labels: object = json.loads(label_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetirementRecoveryError(
            "Committed retirement metadata is unreadable"
        ) from error
    if not isinstance(labels, dict):
        raise RetirementRecoveryError("Committed labels metadata is invalid")
    candidate_sources = {
        parsed.source_stem
        for relative in journal.candidate_paths
        if (parsed := parse_path_identity(Path(relative.as_posix()))) is not None
    }
    for key in labels:
        parsed = parse_key_identity(str(key))
        if parsed is not None and parsed.source_stem in candidate_sources:
            raise RetirementRecoveryError("Committed labels retain retired source")


def _best_effort_remove_unowned_staging(staging_dir: Path) -> None:
    """Clean a root created before journaling without masking the first failure."""
    try:
        if staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
    except OSError:
        pass
    try:
        fsync_directory(staging_dir.parent)
    except OSError:
        pass


def prepare_retirement_journal(
    image_dir: Path,
    source_stem: str,
    candidate_paths: tuple[Path, ...],
    labels_before: bytes | None,
) -> RetirementJournal:
    """Durably record prior state before staging any candidate file."""
    journal_path = journal_path_for(image_dir)
    if journal_path.exists() or journal_path.is_symlink():
        raise RetirementRecoveryError(
            "Pending retirement journal must be recovered before another save"
        )
    relative_paths: list[PurePosixPath] = []
    for path in candidate_paths:
        try:
            relative = path.relative_to(image_dir)
        except ValueError as error:
            raise RetirementRecoveryError(
                "Candidate is outside image directory"
            ) from error
        relative_paths.append(_validate_relative_path(relative.as_posix()))
    journal = RetirementJournal(
        transaction_id=uuid.uuid4().hex,
        source_stem=source_stem,
        candidate_paths=tuple(sorted(relative_paths, key=lambda item: item.as_posix())),
        labels_before=labels_before,
        phase=_PHASE_PREPARED,
    )
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    staging_created = False
    try:
        # Exclusive mkdir is the ownership claim: an exact-path directory or
        # symlink must never be reused as a destination for candidate moves.
        staging_dir.mkdir(mode=0o700)
        staging_created = True
        fsync_directory(staging_dir)
        fsync_directory(staging_dir.parent)
    except FileExistsError as error:
        raise RetirementRecoveryError(
            "Retirement staging root already exists"
        ) from error
    except OSError as error:
        if staging_created:
            _best_effort_remove_unowned_staging(staging_dir)
        raise RetirementRecoveryError(
            "Cannot durably create retirement staging root"
        ) from error
    # Round-trip validation catches a mismatched candidate identity before moving.
    try:
        atomic_write(journal_path, _payload(journal, image_dir))
    except BaseException:
        if not journal_path.exists() and not journal_path.is_symlink():
            _best_effort_remove_unowned_staging(staging_dir)
        raise
    try:
        return _load(image_dir)
    except RetirementRecoveryError:
        # Keep a written-but-invalid journal in place: fail closed for inspection.
        raise


def ensure_no_pending_retirement(image_dir: Path) -> None:
    """Fail closed before candidate discovery if prior recovery is unfinished."""
    path = journal_path_for(image_dir)
    if path.exists() or path.is_symlink():
        raise RetirementRecoveryError(
            "Pending retirement journal must be recovered before another save"
        )


def mark_retirement_committed(
    image_dir: Path, journal: RetirementJournal
) -> RetirementJournal:
    """Durably mark that both metadata files have committed."""
    current = _load(image_dir)
    if current != journal or current.phase != _PHASE_PREPARED:
        raise RetirementRecoveryError("Retirement journal changed before commit")
    committed = dataclasses.replace(current, phase=_PHASE_COMMITTED)
    atomic_write(journal_path_for(image_dir), _payload(committed, image_dir))
    return committed


def committed_retirement_is_valid(image_dir: Path, label_file: Path) -> bool:
    """Confirm the irreversible commit boundary without changing any state."""
    journal = _load(image_dir)
    if journal.phase != _PHASE_COMMITTED:
        return False
    _validate_metadata_paths(image_dir, label_file)
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    locations = _candidate_locations(image_dir, journal)
    _validate_staging_contents(staging_dir, locations)
    for original, _staged in locations:
        if original.exists() or original.is_symlink():
            raise RetirementRecoveryError(
                "Committed retirement candidate unexpectedly exists"
            )
    _validate_committed_metadata(journal, label_file)
    return True


def _remove_staging(staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
        fsync_directory(staging_dir.parent)


def clear_retirement_journal(image_dir: Path) -> None:
    """Remove the durable journal only after recovery/cleanup is complete."""
    path = journal_path_for(image_dir)
    path.unlink()
    fsync_directory(path.parent)


def recover_retirement_transaction(image_dir: Path, label_file: Path) -> bool:
    """Recover one pending transaction, returning whether a journal existed.

    Prepared transactions are rolled back exactly. Committed transactions retain
    metadata and only finish deleting the staging tree. Any structural or state
    inconsistency aborts without clearing the journal.
    """
    journal_path = journal_path_for(image_dir)
    if not journal_path.exists() and not journal_path.is_symlink():
        return False
    journal = _load(image_dir)
    _validate_metadata_paths(image_dir, label_file)
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    locations = _candidate_locations(image_dir, journal)
    _validate_staging_contents(staging_dir, locations)

    if journal.phase == _PHASE_PREPARED:
        for original, staged in locations:
            if original.is_symlink():
                raise RetirementRecoveryError(
                    "Prepared retirement candidate is a symlink"
                )
            original_exists = original.is_file() and not original.is_symlink()
            staged_exists = staged.is_file() and not staged.is_symlink()
            if original_exists == staged_exists:
                raise RetirementRecoveryError(
                    "Prepared retirement has missing or duplicate candidate data"
                )
        for original, staged in reversed(locations):
            if not staged.exists():
                continue
            durable_mkdir(original.parent)
            staged.replace(original)
            fsync_directory(staged.parent)
            fsync_directory(original.parent)
        restore_file(label_file, journal.labels_before)
    else:
        for original, _staged in locations:
            if original.exists() or original.is_symlink():
                raise RetirementRecoveryError(
                    "Committed retirement candidate unexpectedly exists"
                )
        _validate_committed_metadata(journal, label_file)

    _remove_staging(staging_dir)
    clear_retirement_journal(image_dir)
    return True

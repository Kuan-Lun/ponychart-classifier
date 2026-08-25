"""以新 canonical 原圖按 N:1 比例漸進取代最舊的已標註來源樣本。"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

from ponychart_classifier.image_names import ParsedImageName

from .atomic_io import durable_mkdir, fsync_directory
from .constants import NEW_SOURCES_PER_RETIREMENT, RETIREMENT_CUTOFF
from .label_store import LabelStore
from .retirement_journal import (
    RetirementRecoveryError,
    committed_retirement_is_valid,
    ensure_no_pending_retirement,
    mark_retirement_committed,
    prepare_retirement_journal,
    recover_retirement_transaction,
    staging_path_for,
)
from .source_identity import parse_key_identity, parse_path_identity


@dataclasses.dataclass(frozen=True)
class RetiredSample:
    """一次成功淘汰的完整來源樣本。"""

    source_stem: str
    paths: tuple[Path, ...]
    keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SourceSavePlan:
    """One retirement due when this first-time original save succeeds."""

    source_stem: str


@dataclasses.dataclass(frozen=True)
class _ImageEntry:
    path: Path
    parsed: ParsedImageName


@dataclasses.dataclass(frozen=True)
class _RetirementCandidate:
    source_stem: str
    paths: tuple[Path, ...]
    label_keys: tuple[str, ...]
    all_keys: tuple[str, ...]


def _capture_second(parsed: ParsedImageName) -> str:
    """回傳檔名中的秒級 timestamp，刻意忽略 microseconds。"""
    return parsed.captured_at.strftime("%Y%m%d_%H%M%S")


def plan_source_save(
    path: Path,
    labels: list[int],
    store: LabelStore,
) -> SourceSavePlan | None:
    """Plan a retirement for every Nth first-time post-cutoff original save.

    The tally of labeled post-cutoff sources (including this save) is derived from
    the label store, so no counter is persisted. Repeat saves of an already labeled
    source never trigger, and crop labels never count.
    """
    parsed = parse_path_identity(path)
    if (
        parsed is None
        or not parsed.is_original
        or _capture_second(parsed) <= RETIREMENT_CUTOFF
        or not labels
    ):
        return None

    labeled_new_sources = {
        existing.source_stem
        for key in store.all_keys()
        if (existing := parse_key_identity(key)) is not None
        and existing.is_original
        and _capture_second(existing) > RETIREMENT_CUTOFF
        and bool(store.get(key))
    }
    if parsed.source_stem in labeled_new_sources:
        return None
    if (len(labeled_new_sources) + 1) % NEW_SOURCES_PER_RETIREMENT != 0:
        return None
    return SourceSavePlan(parsed.source_stem)


def _find_candidate(
    image_dir: Path,
    store: LabelStore,
    excluded_source_stem: str,
) -> _RetirementCandidate | None:
    labeled_original_sources = {
        parsed.source_stem
        for key in store.all_keys()
        if (parsed := parse_key_identity(key)) is not None
        and parsed.is_original
        and bool(store.get(key))
    }
    groups: dict[str, list[_ImageEntry]] = {}
    for path in image_dir.rglob("*"):
        if path.is_symlink():
            if parse_path_identity(path) is not None:
                raise RetirementRecoveryError(
                    f"Retirement candidate path is a symlink: {path}"
                )
            continue
        if not path.is_file():
            continue
        parsed = parse_path_identity(path)
        if parsed is None:
            continue
        groups.setdefault(parsed.source_stem, []).append(_ImageEntry(path, parsed))

    eligible: list[tuple[str, str]] = []
    for source_stem, entries in groups.items():
        if source_stem == excluded_source_stem:
            continue
        old_originals = [
            entry
            for entry in entries
            if entry.parsed.is_original
            and _capture_second(entry.parsed) < RETIREMENT_CUTOFF
        ]
        if source_stem in labeled_original_sources and old_originals:
            eligible.append(
                (
                    min(_capture_second(entry.parsed) for entry in old_originals),
                    source_stem,
                )
            )

    if not eligible:
        return None

    _, source_stem = min(eligible)
    paths = tuple(sorted((entry.path for entry in groups[source_stem]), key=str))
    label_keys = tuple(
        sorted(
            key
            for key in store.all_keys()
            if (
                (parsed := parse_key_identity(key)) is not None
                and parsed.source_stem == source_stem
            )
        )
    )
    path_keys = {store.path_to_key(path) for path in paths}
    return _RetirementCandidate(
        source_stem=source_stem,
        paths=paths,
        label_keys=label_keys,
        all_keys=tuple(sorted(path_keys | set(label_keys))),
    )


def save_and_retire_oldest_sample(
    image_dir: Path,
    store: LabelStore,
    plan: SourceSavePlan,
    *,
    before_retire: Callable[[str], object] | None = None,
) -> RetiredSample | None:
    """Atomically save labels and retire the oldest labeled source, if any.

    Old files are renamed outside ``image_dir`` before the label file commits.
    An ordinary exception restores staged files, exact previous label bytes, and
    the in-memory store. Without a candidate this is an ordinary label save.
    """
    # Candidate discovery is read-only but must never advance while a prior journal
    # remains unresolved, even when that journal is malformed.
    ensure_no_pending_retirement(image_dir)
    candidate = _find_candidate(image_dir, store, plan.source_stem)
    if candidate is None:
        store.save()
        return None
    store_snapshot = store.snapshot()
    persisted_store = store.persisted_snapshot()
    journal = prepare_retirement_journal(
        image_dir,
        plan.source_stem,
        candidate.paths,
        persisted_store,
    )
    staging_dir = staging_path_for(image_dir, journal.transaction_id)
    try:
        # Tombstone active/background analysis keys while every source file
        # still exists. A prediction that starts after this point can then
        # safely treat FileNotFoundError from the move as a retired sample.
        if before_retire is not None:
            before_retire(candidate.source_stem)
        if staging_dir.is_symlink() or not staging_dir.is_dir():
            raise RetirementRecoveryError("Unsafe retirement staging root")
        for original_path in candidate.paths:
            relative_path = original_path.relative_to(image_dir)
            staged_path = staging_dir / relative_path
            durable_mkdir(staged_path.parent)
            original_path.replace(staged_path)
            fsync_directory(original_path.parent)
            fsync_directory(staged_path.parent)

        for key in candidate.label_keys:
            store.delete(key)
        store.save()
        journal = mark_retirement_committed(image_dir, journal)
    except Exception as error:
        try:
            crossed_commit_boundary = committed_retirement_is_valid(
                image_dir,
                store.file_path,
            )
        except Exception:
            crossed_commit_boundary = False
        if crossed_commit_boundary:
            # The committed marker may already have replaced the journal even when
            # its directory fsync reported an error. Never expose that as a failure
            # to LabelActions: its rollback would resurrect old metadata after the
            # retired files crossed the irreversible boundary.
            try:
                recover_retirement_transaction(image_dir, store.file_path)
            except Exception:
                pass
        else:
            store.restore(store_snapshot)
            rollback_errors: list[BaseException] = []
            try:
                recover_retirement_transaction(image_dir, store.file_path)
            except Exception as restore_error:
                rollback_errors.append(restore_error)
            if rollback_errors:
                raise RuntimeError(
                    "Failed to restore retirement transaction; "
                    f"recovery files may remain in {staging_dir}"
                ) from error
            raise

    # A process death or cleanup failure after the committed marker is harmless:
    # startup keeps metadata and only finishes deleting the staging tree.  Ordinary
    # cleanup errors likewise leave the journal in place to fail closed next time.
    try:
        recover_retirement_transaction(image_dir, store.file_path)
    except Exception:
        pass
    return RetiredSample(
        source_stem=candidate.source_stem,
        paths=candidate.paths,
        keys=candidate.all_keys,
    )

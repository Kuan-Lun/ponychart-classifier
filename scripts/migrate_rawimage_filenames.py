"""Migrate ``rawimage`` to the canonical PonyChart filename schema.

The command is a dry-run by default. Use ``--execute`` only after reviewing the
complete preflight summary::

    uv run python -m scripts.migrate_rawimage_filenames
    uv run python -m scripts.migrate_rawimage_filenames --execute
    uv run python -m scripts.migrate_rawimage_filenames --recover

Legacy parsing intentionally lives only in this one-time migration. Runtime code
accepts canonical names exclusively. ``--execute`` requires the label app,
training, and image-download processes to be closed; derived analysis/checkpoint
state is backed up and deactivated rather than rewritten.
"""

import argparse
import base64
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from app.label_images.constants import ANALYSIS_CACHE_FILE, IMAGE_DIR, LABEL_FILE
from ponychart_classifier.image_names import (
    SUPPORTED_IMAGE_SUFFIXES,
    parse_image_name,
)

CHECKPOINT_PATH = IMAGE_DIR / "checkpoint.pt"

_LEGACY_SOURCE_RE = re.compile(
    r"^(?P<source>pony_chart_(?P<date>\d{8})_(?P<time>\d{6}))$"
)
_LEGACY_CROP_RE = re.compile(
    r"^(?P<source>pony_chart_(?P<date>\d{8})_(?P<time>\d{6}))_"
    r"(?:crop)?(?P<crop>[1-9]\d*)$"
)


class MigrationError(RuntimeError):
    """Raised when preflight finds an unsafe or inconsistent migration state."""


@dataclasses.dataclass(frozen=True)
class RenameOperation:
    """One image rename with its pre-migration content identity."""

    source: Path
    target: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class MigrationPlan:
    """Fully validated image rename plan."""

    image_dir: Path
    operations: tuple[RenameOperation, ...]
    current_count: int
    original_relpaths: frozenset[str]
    final_relpaths: frozenset[str]


@dataclasses.dataclass(frozen=True)
class PreparedMetadata:
    """Validated authoritative metadata ready for atomic replacement."""

    labels: dict[str, Any]
    labels_payload: bytes
    labels_changed: bool
    labels_sha256: str


@dataclasses.dataclass(frozen=True)
class _LegacyName:
    source_stem: str
    crop_index: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_legacy(stem: str) -> _LegacyName | None:
    match = _LEGACY_SOURCE_RE.fullmatch(stem)
    crop_index: int | None = None
    if match is None:
        match = _LEGACY_CROP_RE.fullmatch(stem)
        if match is None:
            return None
        crop_index = int(match.group("crop"))
    try:
        dt.datetime.strptime(
            match.group("date") + match.group("time"),
            "%Y%m%d%H%M%S",
        )
    except ValueError:
        return None
    return _LegacyName(match.group("source"), crop_index)


def _content_token(sha256: str) -> str:
    """Encode the first 40 SHA-256 bits as the schema's 8-character token."""
    digest_prefix = bytes.fromhex(sha256)[:5]
    return base64.b32encode(digest_prefix).decode("ascii").lower()


def _collision_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return unicodedata.normalize("NFC", relative).casefold()


def build_plan(image_dir: Path) -> MigrationPlan:
    """Hash every image and return a collision-free canonical rename plan."""
    image_dir = image_dir.resolve()
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not paths:
        raise MigrationError(f"No images found under {image_dir}")
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise MigrationError(f"Image symlinks are not supported: {symlinks[0]}")

    hashes: dict[Path, str] = {}
    paths_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        digest = _sha256(path)
        hashes[path] = digest
        paths_by_hash[digest].append(path)
    duplicate_groups = [group for group in paths_by_hash.values() if len(group) > 1]
    if duplicate_groups:
        examples = ", ".join(str(path) for path in duplicate_groups[0])
        raise MigrationError(
            f"Duplicate image content must be resolved first: {examples}"
        )

    legacy_sources: dict[str, Path] = {}
    legacy_crops: list[tuple[Path, _LegacyName]] = []
    current_sources: dict[str, Path] = {}
    current_crops: list[tuple[Path, str, int]] = []
    unknown: list[Path] = []

    for path in paths:
        parsed = parse_image_name(path.name)
        if parsed is not None:
            if parsed.is_original:
                if parsed.source_stem in current_sources:
                    raise MigrationError(
                        f"Duplicate canonical source stem: {parsed.source_stem}"
                    )
                current_sources[parsed.source_stem] = path
            else:
                assert parsed.crop_index is not None
                current_crops.append((path, parsed.source_stem, parsed.crop_index))
            continue

        legacy = _parse_legacy(path.stem)
        if legacy is None:
            unknown.append(path)
        elif legacy.crop_index is None:
            if legacy.source_stem in legacy_sources:
                raise MigrationError(
                    f"Duplicate legacy source stem: {legacy.source_stem}"
                )
            legacy_sources[legacy.source_stem] = path
        else:
            legacy_crops.append((path, legacy))

    if unknown:
        raise MigrationError(f"Unsupported image filename: {unknown[0].name}")

    canonical_stems: dict[str, str] = {}
    operations: list[RenameOperation] = []
    for source_stem, path in sorted(legacy_sources.items()):
        canonical_stem = f"{source_stem}_000000_{_content_token(hashes[path])}"
        if canonical_stem in current_sources:
            raise MigrationError(
                f"Migrated source identity already exists: {canonical_stem}"
            )
        canonical_stems[source_stem] = canonical_stem
        operations.append(
            RenameOperation(
                source=path,
                target=path.with_stem(canonical_stem),
                sha256=hashes[path],
            )
        )

    seen_crops: set[tuple[str, int]] = set()
    for path, source_stem, crop_index in current_crops:
        if source_stem not in current_sources:
            raise MigrationError(f"Canonical crop has no source image: {path}")
        identity = (source_stem, crop_index)
        if identity in seen_crops:
            raise MigrationError(
                f"Duplicate crop index for {source_stem}: {crop_index}"
            )
        seen_crops.add(identity)

    for path, legacy in legacy_crops:
        legacy_canonical_stem = canonical_stems.get(legacy.source_stem)
        if legacy_canonical_stem is None:
            raise MigrationError(f"Legacy crop has no source image: {path}")
        assert legacy.crop_index is not None
        identity = (legacy_canonical_stem, legacy.crop_index)
        if identity in seen_crops:
            raise MigrationError(
                f"Duplicate crop index for {legacy.source_stem}: {legacy.crop_index}"
            )
        seen_crops.add(identity)
        operations.append(
            RenameOperation(
                source=path,
                target=path.with_stem(
                    f"{legacy_canonical_stem}_crop{legacy.crop_index}"
                ),
                sha256=hashes[path],
            )
        )

    targets_by_source = {operation.source: operation.target for operation in operations}
    sources = set(targets_by_source)
    targets = {operation.target for operation in operations}
    if not sources.isdisjoint(targets):
        raise MigrationError("Migration sources and targets must be disjoint")
    if len(targets) != len(operations):
        raise MigrationError("Multiple images would be renamed to the same target")
    occupied_targets = [
        target for target in targets if target.exists() and target not in sources
    ]
    if occupied_targets:
        raise MigrationError(f"Migration target already exists: {occupied_targets[0]}")

    final_paths = [targets_by_source.get(path, path) for path in paths]
    collision_keys: dict[str, Path] = {}
    for path in final_paths:
        key = _collision_key(path, image_dir)
        previous = collision_keys.get(key)
        if previous is not None:
            raise MigrationError(
                f"Case/Unicode-normalized target collision: {previous} and {path}"
            )
        collision_keys[key] = path

    return MigrationPlan(
        image_dir=image_dir,
        operations=tuple(sorted(operations, key=lambda op: str(op.source))),
        current_count=len(paths) - len(operations),
        original_relpaths=frozenset(
            path.relative_to(image_dir).as_posix() for path in paths
        ),
        final_relpaths=frozenset(
            path.relative_to(image_dir).as_posix() for path in final_paths
        ),
    )


def _load_json_object_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value: object = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"Cannot read JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"Expected a JSON object: {path}")
    return value, raw


def _load_json_object(path: Path) -> dict[str, Any]:
    value, _ = _load_json_object_bytes(path)
    return value


def _safe_metadata_key(key: object) -> str:
    if not isinstance(key, str) or "\\" in key:
        raise MigrationError(f"Unsafe metadata key: {key!r}")
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise MigrationError(f"Unsafe metadata key: {key!r}")
    return path.as_posix()


def _rename_mapping(plan: MigrationPlan) -> dict[str, str]:
    return {
        operation.source.relative_to(plan.image_dir)
        .as_posix(): operation.target.relative_to(plan.image_dir)
        .as_posix()
        for operation in plan.operations
    }


def _remap_exact_keys(
    values: dict[str, Any],
    *,
    plan: MigrationPlan,
    metadata_name: str,
) -> dict[str, Any]:
    mapping = _rename_mapping(plan)
    remapped: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = _safe_metadata_key(raw_key)
        if key not in plan.original_relpaths:
            raise MigrationError(f"{metadata_name} references a missing image: {key}")
        target = mapping.get(key, key)
        if target in remapped:
            raise MigrationError(f"{metadata_name} key collision: {target}")
        remapped[target] = value
    return remapped


def prepare_metadata(
    plan: MigrationPlan,
    labels_path: Path,
) -> PreparedMetadata:
    """Validate and stage the authoritative label-key rewrite."""
    labels, labels_bytes = _load_json_object_bytes(labels_path)
    remapped_labels = _remap_exact_keys(
        labels,
        plan=plan,
        metadata_name="labels.json",
    )

    labels_payload = _json_bytes(remapped_labels, indent=2)
    return PreparedMetadata(
        labels=remapped_labels,
        labels_payload=labels_payload,
        labels_changed=remapped_labels != labels,
        labels_sha256=hashlib.sha256(labels_bytes).hexdigest(),
    )


def _json_bytes(value: dict[str, Any], *, indent: int | None) -> bytes:
    text = json.dumps(value, ensure_ascii=False, indent=indent) + "\n"
    return text.encode("utf-8")


def _fsync_directories(directories: set[Path]) -> None:
    for directory in sorted(directories):
        file_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)


def _fsync_files(paths: set[Path]) -> None:
    for path in sorted(paths):
        with path.open("rb") as file:
            os.fsync(file.fileno())


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temp_path.chmod(mode)
        temp_path.replace(path)
        _fsync_directories({path.parent})
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(
    path: Path, value: dict[str, Any], *, indent: int | None
) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    _atomic_write_bytes(path, _json_bytes(value, indent=indent), mode=mode)


def _atomic_restore_file(backup: Path, target: Path) -> None:
    """Atomically restore *target* from a same-directory temporary copy."""
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".restore",
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(backup, temp_path)
        with temp_path.open("rb") as file:
            os.fsync(file.fileno())
        temp_path.replace(target)
        _fsync_directories({target.parent})
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _backup_dir(image_dir: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return image_dir / ".filename-migration-backups" / timestamp


def _mkdir_durable(path: Path) -> None:
    """Create a new directory tree, persisting every parent entry."""
    if path.exists():
        raise FileExistsError(path)
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor == cursor.parent:
            raise MigrationError(f"Cannot find existing parent for: {path}")
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directories({directory.parent})


def _metadata_hash(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _manifest(
    plan: MigrationPlan,
    metadata: PreparedMetadata,
    *,
    pre_metadata_hashes: dict[str, str | None],
    status: str,
) -> dict[str, Any]:
    return {
        "version": 2,
        "status": status,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "image_dir": str(plan.image_dir),
        "pre_migration_metadata_sha256": pre_metadata_hashes,
        "migrated_labels_sha256": hashlib.sha256(metadata.labels_payload).hexdigest(),
        "renames": [
            {
                "old": operation.source.relative_to(plan.image_dir).as_posix(),
                "new": operation.target.relative_to(plan.image_dir).as_posix(),
                "sha256": operation.sha256,
            }
            for operation in plan.operations
        ],
    }


def _manifest_renames(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    raw_renames = manifest.get("renames")
    if not isinstance(raw_renames, list):
        raise MigrationError("Migration manifest has no rename list")
    renames: list[tuple[str, str, str]] = []
    for raw_entry in raw_renames:
        if not isinstance(raw_entry, dict):
            raise MigrationError("Migration manifest has an invalid rename entry")
        old = _safe_metadata_key(raw_entry.get("old"))
        new = _safe_metadata_key(raw_entry.get("new"))
        sha256 = raw_entry.get("sha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise MigrationError("Migration manifest has an invalid image hash")
        renames.append((old, new, sha256))
    return renames


def _pending_manifests(image_dir: Path) -> list[Path]:
    backup_root = image_dir / ".filename-migration-backups"
    if not backup_root.exists():
        return []
    pending: list[Path] = []
    for manifest_path in sorted(backup_root.glob("*/manifest.json")):
        manifest = _load_json_object(manifest_path)
        if manifest.get("status") in {"prepared", "rolling_back"}:
            pending.append(manifest_path)
    return pending


@contextmanager
def _migration_lock(image_dir: Path) -> Iterator[None]:
    """Prevent concurrent migration/recovery without creating lock files."""
    try:
        file_descriptor = os.open(image_dir, os.O_RDONLY)
    except OSError as exc:
        raise MigrationError(f"Cannot open image directory: {image_dir}") from exc
    try:
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MigrationError(
                "Another filename migration is already running"
            ) from exc
        yield
    finally:
        os.close(file_descriptor)


def recover_migration(
    manifest_path: Path,
    *,
    labels_path: Path,
    analysis_cache_path: Path,
    checkpoint_path: Path,
) -> None:
    """Roll an interrupted migration back from any recoverable disk state."""
    manifest = _load_json_object(manifest_path)
    if manifest.get("version") != 2:
        raise MigrationError(f"Unsupported migration manifest: {manifest_path}")
    if manifest.get("status") not in {"prepared", "rolling_back"}:
        raise MigrationError(f"Migration is not pending: {manifest_path}")

    raw_image_dir = manifest.get("image_dir")
    if not isinstance(raw_image_dir, str):
        raise MigrationError("Migration manifest has no image directory")
    image_dir = Path(raw_image_dir).resolve()
    if image_dir != labels_path.parent.resolve():
        raise MigrationError(
            "Migration manifest belongs to a different image directory"
        )

    renames = _manifest_renames(manifest)
    image_states: list[tuple[Path, Path, str, str]] = []
    for old, new, image_hash in renames:
        source = image_dir / old
        target = image_dir / new
        source_exists = source.exists()
        target_exists = target.exists()
        if source_exists and target_exists:
            if (
                source.is_symlink()
                or target.is_symlink()
                or not os.path.samefile(source, target)
                or _sha256(source) != image_hash
            ):
                raise MigrationError(
                    f"Cannot recover ambiguous image paths: {source} and {target}"
                )
            state = "linked"
        elif target_exists:
            if target.is_symlink() or _sha256(target) != image_hash:
                raise MigrationError(f"Cannot recover changed image: {target}")
            state = "moved"
        elif source_exists:
            if source.is_symlink() or _sha256(source) != image_hash:
                raise MigrationError(f"Cannot recover changed image: {source}")
            state = "original"
        else:
            raise MigrationError(f"Cannot recover missing image: {source}")
        image_states.append((source, target, image_hash, state))

    raw_hashes = manifest.get("pre_migration_metadata_sha256")
    if not isinstance(raw_hashes, dict):
        raise MigrationError("Migration manifest has no metadata hashes")
    migrated_labels_hash = manifest.get("migrated_labels_sha256")
    if not isinstance(migrated_labels_hash, str):
        raise MigrationError("Migration manifest has no migrated labels hash")
    metadata_paths = {
        "labels.json": labels_path,
        "analysis_cache.json": analysis_cache_path,
        "checkpoint.pt": checkpoint_path,
    }
    pre_hashes: dict[str, str | None] = {}
    current_hashes: dict[str, str | None] = {}
    for name, active_path in metadata_paths.items():
        raw_expected_hash = raw_hashes.get(name)
        if raw_expected_hash is not None and not isinstance(raw_expected_hash, str):
            raise MigrationError(f"Migration manifest has an invalid {name} hash")
        pre_hash: str | None = raw_expected_hash
        if name == "labels.json" and pre_hash is None:
            raise MigrationError("Migration manifest has no labels.json hash")
        pre_hashes[name] = pre_hash
        if pre_hash is not None:
            backup_path = manifest_path.parent / name
            if not backup_path.exists() or _sha256(backup_path) != pre_hash:
                raise MigrationError(
                    f"Cannot recover invalid metadata backup: {backup_path}"
                )
        current_hash = _metadata_hash(active_path)
        accepted_hashes: set[str | None] = {pre_hash}
        if name == "labels.json":
            accepted_hashes.add(migrated_labels_hash)
        else:
            accepted_hashes.add(None)
        if current_hash not in accepted_hashes:
            raise MigrationError(
                f"Cannot overwrite externally changed metadata: {active_path}"
            )
        current_hashes[name] = current_hash

    manifest["status"] = "rolling_back"
    _atomic_write_json(manifest_path, manifest, indent=2)

    changed_directories = {
        source.parent
        for source, _target, _image_hash, state in image_states
        if state in {"linked", "moved"}
    }

    # Recovery mirrors the forward two-phase protocol. First restore every
    # missing legacy name as a no-clobber hard link; only after those links are
    # durable may the canonical names be removed.
    for source, target, _image_hash, state in image_states:
        if state != "moved":
            continue
        try:
            os.link(target, source, follow_symlinks=False)
        except FileExistsError as exc:
            raise MigrationError(f"Recovery target already exists: {source}") from exc
        except OSError as exc:
            raise MigrationError(f"Cannot restore migration source: {source}") from exc
    if changed_directories:
        _fsync_directories(changed_directories)
    for source, target, _image_hash, state in image_states:
        if state == "moved" and not os.path.samefile(source, target):
            raise MigrationError(f"Recovery link verification failed: {source}")

    for _source, target, _image_hash, state in reversed(image_states):
        if state in {"linked", "moved"}:
            target.unlink()
    if changed_directories:
        _fsync_directories(changed_directories)

    for name, active_path in metadata_paths.items():
        pre_hash = pre_hashes[name]
        current_hash = current_hashes[name]
        if pre_hash is not None and current_hash != pre_hash:
            _atomic_restore_file(manifest_path.parent / name, active_path)

    for source, target, image_hash, _state in image_states:
        if (
            not source.exists()
            or target.exists()
            or source.is_symlink()
            or _sha256(source) != image_hash
        ):
            raise MigrationError(f"Migration recovery postcheck failed: {source}")
    for name, active_path in metadata_paths.items():
        if _metadata_hash(active_path) != pre_hashes[name]:
            raise MigrationError(f"Metadata recovery postcheck failed: {active_path}")

    manifest["status"] = "rolled_back"
    manifest["rolled_back_at"] = dt.datetime.now(dt.UTC).isoformat()
    _atomic_write_json(manifest_path, manifest, indent=2)


def execute_migration(
    plan: MigrationPlan,
    metadata: PreparedMetadata,
    *,
    labels_path: Path,
    analysis_cache_path: Path,
    checkpoint_path: Path,
    backup_dir: Path | None = None,
) -> Path:
    """Apply a prepared plan transactionally and return the backup directory."""
    if not plan.operations and not metadata.labels_changed:
        raise MigrationError("Nothing to migrate")
    if _sha256(labels_path) != metadata.labels_sha256:
        raise MigrationError("labels.json changed after preflight; run dry-run again")
    for operation in plan.operations:
        if not operation.source.exists():
            raise MigrationError(f"Migration source disappeared: {operation.source}")
        if operation.target.exists():
            raise MigrationError(f"Migration target appeared: {operation.target}")
        if _sha256(operation.source) != operation.sha256:
            raise MigrationError(f"Migration source changed: {operation.source}")

    destination = backup_dir or _backup_dir(plan.image_dir)
    _mkdir_durable(destination)
    backups: dict[Path, Path] = {}
    for path in (labels_path, analysis_cache_path, checkpoint_path):
        if path.exists():
            backup_path = destination / path.name
            shutil.copy2(path, backup_path)
            backups[path] = backup_path
    _fsync_files(set(backups.values()))
    _fsync_directories({destination})

    if _sha256(backups[labels_path]) != metadata.labels_sha256:
        raise MigrationError("labels.json changed while creating its backup")
    pre_metadata_hashes = {
        name: _metadata_hash(destination / name)
        for name in ("labels.json", "analysis_cache.json", "checkpoint.pt")
    }

    manifest_path = destination / "manifest.json"
    _atomic_write_json(
        manifest_path,
        _manifest(
            plan,
            metadata,
            pre_metadata_hashes=pre_metadata_hashes,
            status="prepared",
        ),
        indent=2,
    )

    try:
        active_metadata = {
            "labels.json": labels_path,
            "analysis_cache.json": analysis_cache_path,
            "checkpoint.pt": checkpoint_path,
        }
        for name, active_path in active_metadata.items():
            if _metadata_hash(active_path) != pre_metadata_hashes[name]:
                raise MigrationError(f"{name} changed while preparing the migration")

        changed_directories = {operation.source.parent for operation in plan.operations}

        # Phase 1: create every target as an exclusive hard link. Existing
        # targets are never overwritten, and recovery can identify this state
        # because source/target resolve to the same inode.
        for operation in plan.operations:
            try:
                os.link(operation.source, operation.target, follow_symlinks=False)
            except FileExistsError as exc:
                raise MigrationError(
                    f"Migration target appeared: {operation.target}"
                ) from exc
            except OSError as exc:
                raise MigrationError(
                    f"Cannot create migration target: {operation.target}"
                ) from exc
        if changed_directories:
            _fsync_directories(changed_directories)
        for operation in plan.operations:
            if (
                not os.path.samefile(operation.source, operation.target)
                or _sha256(operation.target) != operation.sha256
            ):
                raise MigrationError(
                    f"Migration link verification failed: {operation.target}"
                )

        # Phase 2: after every target is durable and verified, remove the old
        # directory entries. Image bytes are always reachable by at least one
        # of the two names.
        for operation in plan.operations:
            operation.source.unlink()
        if changed_directories:
            _fsync_directories(changed_directories)

        for operation in plan.operations:
            if _sha256(operation.target) != operation.sha256:
                raise MigrationError(
                    f"Post-migration hash mismatch: {operation.target}"
                )
            if parse_image_name(operation.target.name) is None:
                raise MigrationError(
                    f"Non-canonical migration target: {operation.target}"
                )
        actual_relpaths = frozenset(
            path.relative_to(plan.image_dir).as_posix()
            for path in plan.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if actual_relpaths != plan.final_relpaths:
            raise MigrationError("Image set changed during migration")

        if metadata.labels_changed:
            if _sha256(labels_path) != metadata.labels_sha256:
                raise MigrationError("labels.json changed during migration")
            mode = stat.S_IMODE(labels_path.stat().st_mode)
            _atomic_write_bytes(labels_path, metadata.labels_payload, mode=mode)
            if (
                _sha256(labels_path)
                != hashlib.sha256(metadata.labels_payload).hexdigest()
            ):
                raise MigrationError("labels.json migration verification failed")
        # Both files are derived state keyed by the old image names. The next
        # model/input-size training run will regenerate them, so retain only
        # the backup instead of rewriting stale contents.
        if plan.operations:
            for derived_path in (analysis_cache_path, checkpoint_path):
                backup = backups.get(derived_path)
                if backup is None:
                    continue
                if _sha256(derived_path) != _sha256(backup):
                    raise MigrationError(
                        f"{derived_path.name} changed during migration"
                    )
                derived_path.unlink()
            _fsync_directories({plan.image_dir})
        if analysis_cache_path.exists() or checkpoint_path.exists():
            raise MigrationError("Derived state was not deactivated")

        manifest = _load_json_object(manifest_path)
        manifest["status"] = "committed"
        manifest["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
        manifest["post_migration_metadata_sha256"] = {
            "labels.json": _metadata_hash(labels_path),
            "analysis_cache.json": _metadata_hash(analysis_cache_path),
            "checkpoint.pt": _metadata_hash(checkpoint_path),
        }
        manifest["checkpoint_backup"] = (
            str(backups[checkpoint_path]) if checkpoint_path in backups else None
        )
        _atomic_write_json(manifest_path, manifest, indent=2)
        return destination
    except BaseException:
        try:
            recover_migration(
                manifest_path,
                labels_path=labels_path,
                analysis_cache_path=analysis_cache_path,
                checkpoint_path=checkpoint_path,
            )
        except BaseException as recovery_error:
            raise MigrationError(
                "Migration failed and automatic recovery is incomplete; "
                f"inspect {manifest_path} before retrying"
            ) from recovery_error
        raise


def _run_locked(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Run preflight/execute/recovery while holding the migration mutex."""
    pending = _pending_manifests(IMAGE_DIR)
    if len(pending) > 1:
        parser.error("偵測到多份未完成的 migration，請先人工檢查 backup manifests")
    if args.recover:
        if not pending:
            print("沒有需要回復的 migration。")
            return
        manifest_path = pending[0]
        recover_migration(
            manifest_path,
            labels_path=LABEL_FILE,
            analysis_cache_path=ANALYSIS_CACHE_FILE,
            checkpoint_path=CHECKPOINT_PATH,
        )
        print(f"已回復：{manifest_path}")
        return
    if pending:
        parser.error("偵測到未完成的 migration；請先關閉標註／訓練程式並執行 --recover")

    plan = build_plan(IMAGE_DIR)
    metadata = prepare_metadata(plan, LABEL_FILE)

    print(f"圖片總數: {len(plan.original_relpaths):,}")
    print(f"需要改名: {len(plan.operations):,}")
    print(f"已是目前格式: {plan.current_count:,}")
    print(f"labels.json 需要更新: {'是' if metadata.labels_changed else '否'}")
    if plan.operations and ANALYSIS_CACHE_FILE.exists():
        print("analysis_cache.json: 備份後停用（不搬移舊模型快取）")
    if plan.operations and CHECKPOINT_PATH.exists():
        print("checkpoint.pt: 備份後停用（下一次訓練從頭開始）")
    for operation in plan.operations[:20]:
        old = operation.source.relative_to(plan.image_dir)
        new = operation.target.relative_to(plan.image_dir)
        print(f"  {old} -> {new}")
    if len(plan.operations) > 20:
        print("  ...")

    if not plan.operations and not metadata.labels_changed:
        print("資料已全部符合目前格式，不需遷移。")
        return
    if not args.execute:
        print("\n此為 dry-run；確認後加上 --execute 才會修改檔案。")
        return

    print("\n開始前請確認 label app、training 與圖片下載程序均已關閉。")
    try:
        backup = execute_migration(
            plan,
            metadata,
            labels_path=LABEL_FILE,
            analysis_cache_path=ANALYSIS_CACHE_FILE,
            checkpoint_path=CHECKPOINT_PATH,
        )
    except Exception as exc:
        raise SystemExit(f"遷移失敗，已嘗試回復原狀：{exc}") from exc
    print(f"\n遷移完成；manifest 與衍生狀態備份於：{backup}")
    print("analysis_cache.json / checkpoint.pt 已停用，之後會重新產生。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將 rawimage 圖片與 metadata 一次性遷移到目前命名格式"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--execute",
        action="store_true",
        help="關閉標註／訓練程式後實際執行（預設只做完整 dry-run）",
    )
    actions.add_argument(
        "--recover",
        action="store_true",
        help="回復先前被中斷且仍為 prepared 狀態的 migration",
    )
    args = parser.parse_args()

    try:
        with _migration_lock(IMAGE_DIR):
            _run_locked(parser, args)
    except MigrationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

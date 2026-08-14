import json
import os
import shutil
from pathlib import Path

import pytest

from ponychart_classifier.image_names import ParsedImageName, parse_image_name
from scripts import migrate_rawimage_filenames as migration


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _metadata_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "labels.json",
        root / "analysis_cache.json",
        root / "checkpoint.pt",
    )


def test_build_plan_migrates_legacy_parent_and_crop_with_content_token(
    tmp_path: Path,
) -> None:
    source = tmp_path / "1" / "pony_chart_20260101_010203.png"
    crop = tmp_path / "unlabeled" / "pony_chart_20260101_010203_1.png"
    current = tmp_path / "2" / "pony_chart_20260102_010203_123456_abcdefgh.png"
    source.parent.mkdir()
    crop.parent.mkdir()
    current.parent.mkdir()
    source.write_bytes(b"source-content")
    crop.write_bytes(b"crop-content")
    current.write_bytes(b"current-content")

    plan = migration.build_plan(tmp_path)

    assert len(plan.operations) == 2
    assert plan.current_count == 1
    source_operation = next(op for op in plan.operations if op.source == source)
    crop_operation = next(op for op in plan.operations if op.source == crop)
    parsed_source = parse_image_name(source_operation.target.name)
    parsed_crop = parse_image_name(crop_operation.target.name)
    assert parsed_source is not None
    assert parsed_crop is not None
    assert parsed_source.source_stem == parsed_crop.source_stem
    assert parsed_source.crop_index is None
    assert parsed_crop.crop_index == 1
    assert "_000000_" in parsed_source.source_stem


def test_prepare_metadata_remaps_label_paths(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "1" / "pony_chart_20260101_010203.png"
    current = tmp_path / "2" / "pony_chart_20260102_010203_123456_abcdefgh.png"
    legacy.parent.mkdir()
    current.parent.mkdir()
    legacy.write_bytes(b"legacy-content")
    current.write_bytes(b"current-content")
    labels_path, _, _ = _metadata_paths(tmp_path)
    _write_json(
        labels_path,
        {
            "1/pony_chart_20260101_010203.png": [1],
            "2/pony_chart_20260102_010203_123456_abcdefgh.png": [2],
        },
    )
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)

    target_key = next(iter(plan.final_relpaths - plan.original_relpaths))
    assert metadata.labels_changed
    assert set(metadata.labels) == {
        target_key,
        "2/pony_chart_20260102_010203_123456_abcdefgh.png",
    }


def test_execute_migration_is_transactional_and_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / "1" / "pony_chart_20260101_010203.png"
    current = tmp_path / "2" / "pony_chart_20260102_010203_123456_abcdefgh.png"
    legacy.parent.mkdir()
    current.parent.mkdir()
    legacy.write_bytes(b"legacy-content")
    current.write_bytes(b"current-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(
        labels_path,
        {
            "1/pony_chart_20260101_010203.png": [1],
            "2/pony_chart_20260102_010203_123456_abcdefgh.png": [2],
        },
    )
    _write_json(
        cache_path,
        {
            "model_sha256": "abc",
            "probs": {
                "1/pony_chart_20260101_010203.png": [0.1],
                "pony_chart_20260102_010203_123456_abcdefgh.png": [0.2],
            },
        },
    )
    checkpoint_path.write_bytes(b"stale-checkpoint")

    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    backup_dir = tmp_path / ".backups" / "test"
    result = migration.execute_migration(
        plan,
        metadata,
        labels_path=labels_path,
        analysis_cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        backup_dir=backup_dir,
    )

    assert result == backup_dir
    assert not legacy.exists()
    assert not cache_path.exists()
    assert not checkpoint_path.exists()
    assert _read_json(backup_dir / "analysis_cache.json")["model_sha256"] == "abc"
    assert (backup_dir / "checkpoint.pt").read_bytes() == b"stale-checkpoint"
    assert _read_json(backup_dir / "manifest.json")["status"] == "committed"
    labels = _read_json(labels_path)
    assert set(labels) == plan.final_relpaths

    second_plan = migration.build_plan(tmp_path)
    second_metadata = migration.prepare_metadata(second_plan, labels_path)
    assert second_plan.operations == ()
    assert not second_metadata.labels_changed


def test_build_plan_rejects_unknown_and_orphan_crop_names(tmp_path: Path) -> None:
    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()
    (unknown_dir / "image.png").write_bytes(b"unknown")

    with pytest.raises(migration.MigrationError, match="Unsupported image filename"):
        migration.build_plan(tmp_path)

    (unknown_dir / "image.png").unlink()
    (unknown_dir / "pony_chart_20260101_010203_crop1.png").write_bytes(b"orphan")

    with pytest.raises(migration.MigrationError, match="has no source image"):
        migration.build_plan(tmp_path)


def test_build_plan_rejects_existing_target(tmp_path: Path) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    digest = migration._sha256(legacy)
    token = migration._content_token(digest)
    target = tmp_path / f"pony_chart_20260101_010203_000000_{token}.png"
    target.write_bytes(b"different-content")

    with pytest.raises(migration.MigrationError, match="already exists"):
        migration.build_plan(tmp_path)


def test_execute_migration_rolls_back_on_postcheck_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    _write_json(cache_path, {"model_sha256": "abc", "probs": {legacy.name: [0.1]}})
    checkpoint_path.write_bytes(b"checkpoint")
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    target = plan.operations[0].target
    real_parse = parse_image_name

    def reject_target(filename: str) -> ParsedImageName | None:
        if filename == target.name:
            return None
        return real_parse(filename)

    monkeypatch.setattr(migration, "parse_image_name", reject_target)
    backup_dir = tmp_path / ".backups" / "rollback"

    with pytest.raises(migration.MigrationError, match="Non-canonical"):
        migration.execute_migration(
            plan,
            metadata,
            labels_path=labels_path,
            analysis_cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            backup_dir=backup_dir,
        )

    assert legacy.exists()
    assert not target.exists()
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert _read_json(cache_path)["model_sha256"] == "abc"
    assert set(_read_json(labels_path)) == {legacy.name}
    assert _read_json(backup_dir / "manifest.json")["status"] == "rolled_back"


def test_execute_migration_rolls_back_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    real_link = os.link

    def interrupt_after_link(
        source: Path,
        target: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        real_link(source, target, follow_symlinks=follow_symlinks)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupt_after_link)
    backup_dir = tmp_path / ".backups" / "interrupt"

    with pytest.raises(KeyboardInterrupt):
        migration.execute_migration(
            plan,
            metadata,
            labels_path=labels_path,
            analysis_cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            backup_dir=backup_dir,
        )

    assert legacy.exists()
    assert not plan.operations[0].target.exists()
    assert set(_read_json(labels_path)) == {legacy.name}
    assert _read_json(backup_dir / "manifest.json")["status"] == "rolled_back"


def test_execute_migration_rejects_labels_changed_after_preflight(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    _write_json(labels_path, {legacy.name: [2]})

    with pytest.raises(migration.MigrationError, match="changed after preflight"):
        migration.execute_migration(
            plan,
            metadata,
            labels_path=labels_path,
            analysis_cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            backup_dir=tmp_path / ".backups" / "stale-labels",
        )

    assert legacy.exists()
    assert not plan.operations[0].target.exists()
    assert _read_json(labels_path) == {legacy.name: [2]}


@pytest.mark.parametrize(
    "crash_state",
    ["original", "linked", "moved", "metadata_committed"],
)
def test_recover_migration_handles_each_durable_crash_state(
    tmp_path: Path,
    crash_state: str,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    original_labels = {legacy.name: [1]}
    _write_json(labels_path, original_labels)
    cache_path.write_bytes(b"cache")
    checkpoint_path.write_bytes(b"checkpoint")
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    operation = plan.operations[0]

    backup_dir = tmp_path / ".filename-migration-backups" / crash_state
    backup_dir.mkdir(parents=True)
    for metadata_path in (labels_path, cache_path, checkpoint_path):
        shutil.copy2(metadata_path, backup_dir / metadata_path.name)
    pre_hashes: dict[str, str | None] = {
        name: migration._sha256(backup_dir / name)
        for name in ("labels.json", "analysis_cache.json", "checkpoint.pt")
    }
    manifest_path = backup_dir / "manifest.json"
    migration._atomic_write_json(
        manifest_path,
        migration._manifest(
            plan,
            metadata,
            pre_metadata_hashes=pre_hashes,
            status="rolling_back" if crash_state == "moved" else "prepared",
        ),
        indent=2,
    )

    if crash_state != "original":
        os.link(operation.source, operation.target, follow_symlinks=False)
        if crash_state in {"moved", "metadata_committed"}:
            operation.source.unlink()
    if crash_state == "metadata_committed":
        labels_path.write_bytes(metadata.labels_payload)
        cache_path.unlink()
        checkpoint_path.unlink()

    migration.recover_migration(
        manifest_path,
        labels_path=labels_path,
        analysis_cache_path=cache_path,
        checkpoint_path=checkpoint_path,
    )

    assert legacy.read_bytes() == b"legacy-content"
    assert not operation.target.exists()
    assert _read_json(labels_path) == original_labels
    assert cache_path.read_bytes() == b"cache"
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert _read_json(manifest_path)["status"] == "rolled_back"


def test_recover_migration_refuses_to_overwrite_external_labels(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    backup_dir = tmp_path / ".filename-migration-backups" / "external-labels"
    backup_dir.mkdir(parents=True)
    shutil.copy2(labels_path, backup_dir / labels_path.name)
    manifest_path = backup_dir / "manifest.json"
    migration._atomic_write_json(
        manifest_path,
        migration._manifest(
            plan,
            metadata,
            pre_metadata_hashes={
                "labels.json": migration._sha256(backup_dir / labels_path.name),
                "analysis_cache.json": None,
                "checkpoint.pt": None,
            },
            status="prepared",
        ),
        indent=2,
    )
    external_labels = {legacy.name: [6]}
    _write_json(labels_path, external_labels)

    with pytest.raises(migration.MigrationError, match="externally changed"):
        migration.recover_migration(
            manifest_path,
            labels_path=labels_path,
            analysis_cache_path=cache_path,
            checkpoint_path=checkpoint_path,
        )

    assert _read_json(labels_path) == external_labels
    assert legacy.read_bytes() == b"legacy-content"
    assert not plan.operations[0].target.exists()
    assert _read_json(manifest_path)["status"] == "prepared"


def test_execute_migration_never_overwrites_target_created_after_preflight(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    target = plan.operations[0].target
    target.write_bytes(b"concurrent-image")

    with pytest.raises(migration.MigrationError, match="target appeared"):
        migration.execute_migration(
            plan,
            metadata,
            labels_path=labels_path,
            analysis_cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            backup_dir=tmp_path / ".backups" / "target-race",
        )

    assert legacy.read_bytes() == b"legacy-content"
    assert target.read_bytes() == b"concurrent-image"


def test_migration_lock_is_exclusive_and_creates_no_files(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())

    with migration._migration_lock(tmp_path):
        with pytest.raises(migration.MigrationError, match="already running"):
            with migration._migration_lock(tmp_path):
                pass

    assert set(tmp_path.iterdir()) == before


def test_recovery_is_reentrant_when_unlink_interrupts_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "pony_chart_20260101_010203.png"
    legacy.write_bytes(b"legacy-content")
    labels_path, cache_path, checkpoint_path = _metadata_paths(tmp_path)
    _write_json(labels_path, {legacy.name: [1]})
    plan = migration.build_plan(tmp_path)
    metadata = migration.prepare_metadata(plan, labels_path)
    operation = plan.operations[0]
    backup_dir = tmp_path / ".filename-migration-backups" / "unlink-interrupt"
    backup_dir.mkdir(parents=True)
    shutil.copy2(labels_path, backup_dir / labels_path.name)
    manifest_path = backup_dir / "manifest.json"
    migration._atomic_write_json(
        manifest_path,
        migration._manifest(
            plan,
            metadata,
            pre_metadata_hashes={
                "labels.json": migration._sha256(backup_dir / labels_path.name),
                "analysis_cache.json": None,
                "checkpoint.pt": None,
            },
            status="prepared",
        ),
        indent=2,
    )
    os.link(operation.source, operation.target, follow_symlinks=False)
    operation.source.unlink()
    real_unlink = Path.unlink

    def unlink_then_interrupt(path: Path, missing_ok: bool = False) -> None:
        real_unlink(path, missing_ok=missing_ok)
        if path == operation.target:
            raise KeyboardInterrupt

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", unlink_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            migration.recover_migration(
                manifest_path,
                labels_path=labels_path,
                analysis_cache_path=cache_path,
                checkpoint_path=checkpoint_path,
            )

    assert legacy.read_bytes() == b"legacy-content"
    assert not operation.target.exists()
    assert _read_json(manifest_path)["status"] == "rolling_back"

    migration.recover_migration(
        manifest_path,
        labels_path=labels_path,
        analysis_cache_path=cache_path,
        checkpoint_path=checkpoint_path,
    )
    assert legacy.read_bytes() == b"legacy-content"
    assert _read_json(manifest_path)["status"] == "rolled_back"

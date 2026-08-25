import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import ponychart_classifier
from app.label_images.analysis import AnalysisManager
from app.label_images.app import _LabelActions
from app.label_images.atomic_io import atomic_write as durable_atomic_write
from app.label_images.atomic_io import durable_mkdir
from app.label_images.file_ops import organize_single
from app.label_images.label_store import LabelStore
from app.label_images.mutation_guard import RawImageMutationGuard
from app.label_images.navigator import ImageNavigator
from app.label_images.retirement_journal import (
    RetirementRecoveryError,
    journal_path_for,
    prepare_retirement_journal,
    recover_retirement_transaction,
    staging_path_for,
)
from app.label_images.sample_retirement import (
    RetiredSample,
    SourceSavePlan,
    plan_source_save,
    save_and_retire_oldest_sample,
)
from ponychart_classifier.image_names import parse_image_name
from ponychart_classifier.training.sampling import Sample


def _image(image_dir: Path, relative: str) -> Path:
    path = image_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())
    return path


def _store(
    image_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LabelStore:
    monkeypatch.setattr("app.label_images.label_store.IMAGE_DIR", image_dir)
    return LabelStore(image_dir / "labels.json", "rawimage")


def test_trigger_uses_strict_second_precision_and_canonical_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    # 先標一張新來源，讓下一張首次標註成為第 2 張（2 mod 2 == 0 才排定淘汰）。
    prior = image_dir / "pony_chart_20260825_000002_000000_prior001.png"
    store.set(store.path_to_key(prior), [1])
    equal = image_dir / "pony_chart_20260825_000000_999999_equal001.png"
    after = image_dir / "pony_chart_20260825_000001_000000_after001.png"
    crop = image_dir / "pony_chart_20260825_000001_000000_after001_crop1.png"

    assert plan_source_save(equal, [1], store) is None
    assert plan_source_save(after, [], store) is None
    assert plan_source_save(crop, [1], store) is None
    assert plan_source_save(image_dir / "not-a-pony.png", [1], store) is None
    assert plan_source_save(after, [1], store) == SourceSavePlan(after.stem)


def test_every_second_first_time_new_source_save_plans_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    originals = [
        image_dir / f"pony_chart_20260825_00000{i}_000000_new0000{i}.png"
        for i in range(1, 5)
    ]

    expected = [
        None,
        SourceSavePlan(originals[1].stem),
        None,
        SourceSavePlan(originals[3].stem),
    ]
    for original, plan in zip(originals, expected, strict=True):
        assert plan_source_save(original, [1], store) == plan
        store.set(store.path_to_key(original), [1])


def test_crop_label_does_not_trigger_or_count_toward_tally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    prior = image_dir / "pony_chart_20260825_000001_000000_prior001.png"
    store.set(store.path_to_key(prior), [1])
    source = "pony_chart_20260825_000002_000000_after001"
    original = image_dir / f"{source}.png"
    crop = image_dir / f"{source}_crop1.png"

    # 已標註的 crop 不計入 tally，也不讓來源被視為已標註。
    store.set(store.path_to_key(crop), [2])
    assert plan_source_save(crop, [2], store) is None
    assert plan_source_save(original, [2], store) == SourceSavePlan(source)

    # 首次標註後的重複儲存不再觸發。
    store.set(store.path_to_key(original), [2])
    assert plan_source_save(original, [3], store) is None


def test_loaded_empty_original_label_neither_counts_nor_blocks_first_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    labeled_source = "pony_chart_20260825_000001_000000_new00001"
    empty_source = "pony_chart_20260825_000002_000000_new00002"
    (image_dir / "labels.json").write_text(
        json.dumps(
            {
                f"{labeled_source}.png": [1],
                f"{empty_source}.png": [],
            }
        ),
        encoding="utf-8",
    )
    store = _store(image_dir, monkeypatch)

    # 空標籤不算已標註：empty_source 是第 2 個首次標註 → 觸發。
    assert plan_source_save(
        image_dir / f"{empty_source}.png", [1], store
    ) == SourceSavePlan(empty_source)


def test_loaded_empty_original_label_is_not_a_retirement_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    old_key = str(old.relative_to(image_dir))
    (image_dir / "labels.json").write_text(
        json.dumps({old_key: []}),
        encoding="utf-8",
    )
    store = _store(image_dir, monkeypatch)
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(store.path_to_key(new), [2])

    retired = save_and_retire_oldest_sample(
        image_dir,
        store,
        SourceSavePlan(new_source),
    )

    assert retired is None
    assert old.exists()
    assert store.has(old_key)
    assert store.get(old_key) == []


def test_retirement_removes_oldest_labeled_source_and_all_its_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)

    unlabeled_older = _image(
        image_dir,
        "unlabeled/pony_chart_20240101_000000_000000_nolabel1.png",
    )
    oldest_source = "pony_chart_20250101_000000_000000_old00001"
    oldest_original = _image(image_dir, f"1/twilight/{oldest_source}.png")
    oldest_crop1 = _image(image_dir, f"1/twilight/{oldest_source}_crop1.png")
    oldest_crop2 = _image(image_dir, f"2/twilight+rarity/{oldest_source}_crop2.png")
    legacy_conflict = _image(
        image_dir,
        f"_conflicts/{oldest_source}_crop4_1.png",
    )
    canonical_conflict = _image(
        image_dir,
        f"_conflicts/{oldest_source}_crop5_conflict2.png",
    )
    manual_similar = _image(
        image_dir,
        f"unlabeled/{oldest_source}_crop6_1.png",
    )
    malformed = _image(image_dir, f"unlabeled/{oldest_source}_crop0.png")
    malformed_key = store.path_to_key(malformed)
    orphan_crop_key = f"3/twilight+rarity+fluttershy/{oldest_source}_crop3.png"
    newer_source = "pony_chart_20260101_000000_000000_old00002"
    newer_original = _image(image_dir, f"1/rarity/{newer_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new_original = _image(image_dir, f"1/fluttershy/{new_source}.png")

    for path, labels in (
        (oldest_original, [1]),
        (oldest_crop1, [1]),
        (legacy_conflict, [1]),
        (canonical_conflict, [1]),
        (manual_similar, [1]),
        (newer_original, [2]),
        (new_original, [3]),
    ):
        store.set(store.path_to_key(path), labels)
    store.set(orphan_crop_key, [1, 2, 3])
    store.set(malformed_key, [6])

    retired = save_and_retire_oldest_sample(
        image_dir,
        store,
        SourceSavePlan(new_source),
    )

    assert retired is not None
    assert retired.source_stem == oldest_source
    assert not oldest_original.exists()
    assert not oldest_crop1.exists()
    assert not oldest_crop2.exists()
    assert not legacy_conflict.exists()
    assert not canonical_conflict.exists()
    assert manual_similar.exists()
    assert malformed.exists()
    assert unlabeled_older.exists()
    assert newer_original.exists()
    assert new_original.exists()
    assert not store.has(store.path_to_key(oldest_original))
    assert not store.has(store.path_to_key(oldest_crop1))
    assert not store.has(orphan_crop_key)
    assert not store.has(store.path_to_key(legacy_conflict))
    assert not store.has(store.path_to_key(canonical_conflict))
    assert store.get(store.path_to_key(manual_similar)) == [1]
    assert store.get(malformed_key) == [6]
    saved = json.loads((image_dir / "labels.json").read_text(encoding="utf-8"))
    assert saved == {key: store.get(key) for key in store.all_keys()}
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


def test_no_candidate_only_commits_new_label_without_deleting_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    unlabeled_old = _image(
        image_dir,
        "unlabeled/pony_chart_20240101_000000_000000_nolabel1.png",
    )
    equal_source = "pony_chart_20260825_000000_999999_equal001"
    equal = _image(image_dir, f"1/twilight/{equal_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(store.path_to_key(equal), [1])
    store.set(store.path_to_key(new), [2])

    retired = save_and_retire_oldest_sample(
        image_dir,
        store,
        SourceSavePlan(new_source),
    )

    assert retired is None
    assert unlabeled_old.exists()
    assert equal.exists()
    assert new.exists()
    saved = json.loads((image_dir / "labels.json").read_text(encoding="utf-8"))
    assert store.path_to_key(new) in saved
    assert not journal_path_for(image_dir).exists()
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


def test_retirement_rolls_back_files_and_labels_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    crop = _image(image_dir, f"1/twilight/{old_source}_crop1.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    old_key = store.path_to_key(old)
    crop_key = store.path_to_key(crop)
    new_key = store.path_to_key(new)
    store.set(old_key, [1])
    store.set(crop_key, [1])
    store.save()
    before = (image_dir / "labels.json").read_bytes()
    store.set(new_key, [2])

    def fail_save() -> None:
        raise OSError("simulated labels commit failure")

    monkeypatch.setattr(store, "save", fail_save)
    with pytest.raises(OSError, match="simulated labels commit failure"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )

    assert old.exists()
    assert crop.exists()
    assert new.exists()
    assert store.get(old_key) == [1]
    assert store.get(crop_key) == [1]
    assert store.get(new_key) == [2]
    assert (image_dir / "labels.json").read_bytes() == before
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


def test_startup_recovery_restores_partial_file_move_and_exact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    crop = _image(image_dir, f"1/twilight/{old_source}_crop1.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(store.path_to_key(old), [1])
    store.set(store.path_to_key(crop), [1])
    store.save()
    labels_before = store.file_path.read_bytes()
    store.set(store.path_to_key(new), [2])
    original_replace = Path.replace

    def crash_before_second_move(path: Path, target: Path) -> Path:
        if path == crop:
            raise SystemExit("simulated process death")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", crash_before_second_move)
    with pytest.raises(SystemExit, match="simulated process death"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )

    assert not old.exists()
    assert crop.exists()
    assert journal_path_for(image_dir).exists()
    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        RawImageMutationGuard(image_dir).ensure_allowed()

    assert recover_retirement_transaction(image_dir, store.file_path)
    assert old.exists()
    assert crop.exists()
    assert store.file_path.read_bytes() == labels_before
    assert not journal_path_for(image_dir).exists()
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


def test_startup_recovery_rolls_back_partial_metadata_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(store.path_to_key(old), [1])
    store.save()
    labels_before = store.file_path.read_bytes()
    store.set(store.path_to_key(new), [2])
    original_save = store.save

    def crash_after_labels_save() -> None:
        original_save()
        raise SystemExit("simulated process death")

    monkeypatch.setattr(store, "save", crash_after_labels_save)
    with pytest.raises(SystemExit, match="simulated process death"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )

    assert not old.exists()
    assert store.file_path.read_bytes() != labels_before

    assert recover_retirement_transaction(image_dir, store.file_path)
    assert old.exists()
    assert store.file_path.read_bytes() == labels_before


def test_startup_recovery_keeps_committed_metadata_and_finishes_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    old_key = store.path_to_key(old)
    new_key = store.path_to_key(new)
    store.set(old_key, [1])
    store.save()
    store.set(new_key, [2])

    def crash_before_staging_cleanup(path: Path) -> None:
        del path
        raise SystemExit("simulated process death")

    monkeypatch.setattr(
        "app.label_images.retirement_journal._remove_staging",
        crash_before_staging_cleanup,
    )
    with pytest.raises(SystemExit, match="simulated process death"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )

    journal_path = journal_path_for(image_dir)
    assert json.loads(journal_path.read_text(encoding="utf-8"))["phase"] == "committed"
    committed_labels = store.file_path.read_bytes()
    assert not old.exists()

    monkeypatch.undo()
    assert recover_retirement_transaction(image_dir, store.file_path)
    assert not old.exists()
    assert store.file_path.read_bytes() == committed_labels
    assert not journal_path.exists()
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


def test_committed_journal_replace_error_never_exposes_rollback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    old_key = store.path_to_key(old)
    new_key = store.path_to_key(new)
    store.set(old_key, [1])
    store.save()
    store.set(new_key, [2])

    def replace_then_report_fsync_failure(path: Path, payload: bytes) -> None:
        durable_atomic_write(path, payload)
        if json.loads(payload)["phase"] == "committed":
            raise OSError("simulated directory fsync failure after replace")

    monkeypatch.setattr(
        "app.label_images.retirement_journal.atomic_write",
        replace_then_report_fsync_failure,
    )

    retired = save_and_retire_oldest_sample(
        image_dir,
        store,
        SourceSavePlan(new_source),
    )

    assert retired is not None
    assert not old.exists()
    assert not store.has(old_key)
    assert store.get(new_key) == [2]
    assert not journal_path_for(image_dir).exists()


def test_pre_move_tombstone_turns_background_missing_file_into_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    old_key = store.path_to_key(old)
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(old_key, [1])
    store.save()
    store.set(store.path_to_key(new), [2])
    manager = AnalysisManager()
    manager._active_keys = {old_key}
    at_predict = threading.Event()
    continue_predict = threading.Event()
    monkeypatch.setattr(ponychart_classifier, "update", lambda: None)
    monkeypatch.setattr(
        ponychart_classifier,
        "get_thresholds",
        lambda: SimpleNamespace(as_list=lambda: [0.5] * 6),
    )

    def predict_after_retirement(path: str) -> SimpleNamespace:
        at_predict.set()
        assert continue_predict.wait(timeout=2)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return SimpleNamespace(
            twilight_sparkle=0.1,
            rarity=0.2,
            fluttershy=0.3,
            rainbow_dash=0.4,
            pinkie_pie=0.5,
            applejack=0.6,
        )

    monkeypatch.setattr(ponychart_classifier, "predict", predict_after_retirement)
    worker = threading.Thread(
        target=manager._run,
        args=([Sample(str(old), [])], [old_key]),
    )
    worker.start()
    assert at_predict.wait(timeout=2)
    callback_saw_existing_file = False

    def tombstone_before_move(source_stem: str) -> None:
        nonlocal callback_saw_existing_file
        assert source_stem == old_source
        callback_saw_existing_file = old.exists()
        manager.purge_source(source_stem)

    save_and_retire_oldest_sample(
        image_dir,
        store,
        SourceSavePlan(new_source),
        before_retire=tombstone_before_move,
    )
    assert not old.exists()
    continue_predict.set()
    worker.join(timeout=2)

    assert callback_saw_existing_file
    assert not worker.is_alive()
    assert manager._error is None
    assert manager._result == ({}, [0.5] * 6)


def test_malformed_or_inconsistent_journal_fails_closed_before_candidate_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    new_source = "pony_chart_20260825_000001_000000_new00001"
    journal_path = journal_path_for(image_dir)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"keep")
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "transaction_id": "a" * 32,
                "staging_dir": f".{image_dir.name}-retirement-{'a' * 32}",
                "source_stem": new_source,
                "candidate_paths": ["../outside.png"],
                "labels_before": None,
                "phase": "prepared",
            }
        ),
        encoding="utf-8",
    )
    scanned = False

    def unexpected_scan(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal scanned
        scanned = True

    monkeypatch.setattr(
        "app.label_images.sample_retirement._find_candidate", unexpected_scan
    )
    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )
    with pytest.raises(RetirementRecoveryError, match="candidate path"):
        recover_retirement_transaction(image_dir, store.file_path)

    assert scanned is False
    assert outside.read_bytes() == b"keep"
    assert journal_path.exists()


def test_symlink_candidate_is_rejected_before_journal_or_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    outside = tmp_path / "outside-image.png"
    outside.write_bytes(b"outside")
    old_source = "pony_chart_20250101_000000_000000_old00001"
    linked_old = image_dir / "1" / "twilight" / f"{old_source}.png"
    linked_old.parent.mkdir(parents=True)
    linked_old.symlink_to(outside)
    store = _store(image_dir, monkeypatch)
    store.set(store.path_to_key(linked_old), [1])
    store.save()
    labels_before = store.file_path.read_bytes()
    new_source = "pony_chart_20260825_000001_000000_new00001"
    new = _image(image_dir, f"1/rarity/{new_source}.png")
    store.set(store.path_to_key(new), [2])

    with pytest.raises(RetirementRecoveryError, match="candidate path is a symlink"):
        save_and_retire_oldest_sample(
            image_dir,
            store,
            SourceSavePlan(new_source),
        )

    assert linked_old.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert new.exists()
    assert store.file_path.read_bytes() == labels_before
    assert not journal_path_for(image_dir).exists()
    assert list(tmp_path.glob(".rawimage-retirement-*")) == []


@pytest.mark.parametrize("existing_kind", ["directory", "symlink"])
def test_preexisting_exact_staging_root_is_never_reused_or_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    transaction_id = "a" * 32
    staging = tmp_path / f".rawimage-retirement-{transaction_id}"
    if existing_kind == "directory":
        protected_dir = staging
        protected_dir.mkdir()
    else:
        protected_dir = tmp_path / "outside-staging"
        protected_dir.mkdir()
        staging.symlink_to(protected_dir, target_is_directory=True)
    sentinel = protected_dir / "do-not-touch"
    sentinel.write_bytes(b"protected")
    monkeypatch.setattr(
        "app.label_images.retirement_journal.uuid.uuid4",
        lambda: SimpleNamespace(hex=transaction_id),
    )

    with pytest.raises(RetirementRecoveryError, match="staging root already exists"):
        prepare_retirement_journal(
            image_dir,
            "pony_chart_20260826_000001_000000_journal1",
            (),
            None,
        )

    assert sentinel.read_bytes() == b"protected"
    assert staging.exists()
    assert staging.is_symlink() is (existing_kind == "symlink")
    assert not journal_path_for(image_dir).exists()


@pytest.mark.parametrize("failed_fsync_call", [1, 2])
def test_prejournal_staging_fsync_failure_removes_unowned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_fsync_call: int,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    transaction_id = "b" * 32
    staging = tmp_path / f".rawimage-retirement-{transaction_id}"
    monkeypatch.setattr(
        "app.label_images.retirement_journal.uuid.uuid4",
        lambda: SimpleNamespace(hex=transaction_id),
    )
    fsync_calls = 0

    def fail_one_durability_barrier(path: Path) -> None:
        del path
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == failed_fsync_call:
            raise OSError("simulated pre-journal directory fsync failure")

    monkeypatch.setattr(
        "app.label_images.retirement_journal.fsync_directory",
        fail_one_durability_barrier,
    )

    with pytest.raises(
        RetirementRecoveryError,
        match="Cannot durably create retirement staging root",
    ):
        prepare_retirement_journal(
            image_dir,
            "pony_chart_20260826_000001_000000_journal1",
            (),
            None,
        )

    assert fsync_calls >= failed_fsync_call
    assert not staging.exists()
    assert not staging.is_symlink()
    assert not journal_path_for(image_dir).exists()


def test_inconsistent_prepared_journal_keeps_files_and_journal_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    new_source = "pony_chart_20260825_000001_000000_new00001"
    journal = prepare_retirement_journal(
        image_dir,
        new_source,
        (old,),
        None,
    )
    staged = staging_path_for(image_dir, journal.transaction_id) / old.relative_to(
        image_dir
    )
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"duplicate")

    with pytest.raises(RetirementRecoveryError, match="missing or duplicate"):
        recover_retirement_transaction(image_dir, store.file_path)

    assert old.exists()
    assert staged.exists()
    assert journal_path_for(image_dir).exists()


def test_symlink_journal_fails_closed_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    outside = tmp_path / "outside-journal.json"
    outside.write_text('{"do_not_touch": true}', encoding="utf-8")
    journal = journal_path_for(image_dir)
    journal.symlink_to(outside)

    with pytest.raises(RetirementRecoveryError, match="Unsafe retirement journal"):
        recover_retirement_transaction(image_dir, store.file_path)

    assert journal.is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"do_not_touch": true}'


def test_label_store_atomic_save_creates_missing_parent(tmp_path: Path) -> None:
    label_file = tmp_path / "missing" / "rawimage" / "labels.json"
    store = LabelStore(label_file, "rawimage")
    store.set("sample.png", [1])

    store.save()

    assert json.loads(label_file.read_text(encoding="utf-8")) == {"sample.png": [1]}
    assert list(label_file.parent.glob(".labels.json.*.tmp")) == []


def test_durable_mkdir_syncs_each_nested_directory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        "app.label_images.atomic_io.fsync_directory",
        calls.append,
    )
    first = tmp_path / "staging"
    second = first / "1"
    deepest = second / "twilight"

    durable_mkdir(deepest)

    assert deepest.is_dir()
    assert tmp_path in calls
    assert first in calls
    assert second in calls
    assert deepest in calls


def test_label_store_atomic_save_preserves_previous_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_file = tmp_path / "rawimage" / "labels.json"
    store = LabelStore(label_file, "rawimage")
    store.set("old.png", [1])
    store.save()
    before = label_file.read_bytes()
    store.set("new.png", [2])

    def fail_dump(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated serialization failure")

    monkeypatch.setattr("app.label_images.label_store.json.dumps", fail_dump)
    with pytest.raises(OSError, match="simulated serialization failure"):
        store.save()

    assert label_file.read_bytes() == before
    assert list(label_file.parent.glob(".labels.json.*.tmp")) == []


def test_label_store_set_copies_mutable_ui_labels(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "rawimage" / "labels.json", "rawimage")
    ui_labels = [1]

    store.set("sample.png", ui_labels)
    ui_labels.append(2)

    assert store.get("sample.png") == [1]


def test_organize_conflict_name_preserves_canonical_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    source = "pony_chart_20250101_000000_000000_old00001"
    current = _image(image_dir, f"unlabeled/{source}_crop1.png")
    target = _image(image_dir, f"1/twilight/{source}_crop1.png")
    current.write_bytes(b"new")
    target.write_bytes(b"existing")
    monkeypatch.setattr("app.label_images.file_ops.IMAGE_DIR", image_dir)

    conflict = organize_single(current, [1])

    assert conflict == image_dir / "_conflicts" / f"{source}_crop1_conflict1.png"
    parsed = parse_image_name(conflict.name)
    assert parsed is not None
    assert parsed.source_stem == source
    assert parsed.crop_index == 1
    assert parsed.conflict_index == 1


class _FakeNav:
    def __init__(self, path: Path, store: LabelStore) -> None:
        self.current_path = path
        self._store = store
        self.is_filtered = False
        self.removed: list[Path] = []

    @property
    def current_key(self) -> str:
        return self._store.path_to_key(self.current_path)

    def replace_path(self, old_path: Path, new_path: Path) -> None:
        assert self.current_path == old_path
        self.current_path = new_path

    def remove_path(self, path: Path) -> None:
        self.removed.append(path)

    def go_to_key(self, key: str) -> None:
        del key


class _FakeAnalysis:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.save_cache_fail_safe_calls = 0
        self.renamed: list[tuple[str, str]] = []

    def rename_key(self, old_key: str, new_key: str) -> None:
        self.renamed.append((old_key, new_key))

    def purge_source(self, source_stem: str) -> list[str]:
        self.deleted.append(source_stem)
        return [source_stem]

    def save_cache_fail_safe(self) -> bool:
        self.save_cache_fail_safe_calls += 1
        return True


class _FakeApp:
    def __init__(self, path: Path, store: LabelStore) -> None:
        self.store = store
        self.mutation_guard = RawImageMutationGuard(store.file_path.parent)
        self.nav = _FakeNav(path, store)
        self.analysis = _FakeAnalysis()
        self.current_labels = [1]
        self.messages: list[str] = []

    def update_display(self, extra: str = "") -> None:
        self.messages.append(extra)


@pytest.mark.parametrize("journal_kind", ["prepared", "malformed"])
@pytest.mark.parametrize("save_kind", ["pre_cutoff", "crop", "repeat"])
def test_every_non_retirement_save_is_blocked_by_pending_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_kind: str,
    save_kind: str,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    if save_kind == "pre_cutoff":
        source = "pony_chart_20260824_235959_000000_before01"
        filename = f"{source}.png"
    else:
        source = "pony_chart_20260825_000001_000000_new00001"
        filename = f"{source}_crop1.png" if save_kind == "crop" else f"{source}.png"
    current = _image(image_dir, f"unlabeled/{filename}")
    current_key = store.path_to_key(current)
    store.set(current_key, [1])
    store.save()
    journal_path = journal_path_for(image_dir)
    if journal_kind == "prepared":
        prepare_retirement_journal(
            image_dir,
            "pony_chart_20260826_000001_000000_journal1",
            (),
            store.file_path.read_bytes(),
        )
    else:
        journal_path.write_text("{malformed", encoding="utf-8")
    journal_before = journal_path.read_bytes()
    labels_before = store.file_path.read_bytes()
    image_before = current.read_bytes()
    memory_before = store.snapshot()
    app = _FakeApp(current, store)
    app.current_labels = [2]
    organized = False

    def unexpected_organize(path: Path, labels: list[int]) -> Path:
        del path, labels
        nonlocal organized
        organized = True
        raise AssertionError("organize must not run with a pending journal")

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr("app.label_images.app.organize_single", unexpected_organize)

    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        _LabelActions(app).save()  # type: ignore[arg-type]

    assert organized is False
    assert store.snapshot() == memory_before
    assert store.file_path.read_bytes() == labels_before
    assert current.read_bytes() == image_before
    assert app.nav.current_path == current
    assert app.messages == []
    assert journal_path.read_bytes() == journal_before


def test_label_action_retires_one_old_source_per_two_new_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source1 = "pony_chart_20250101_000000_000000_old00001"
    old1 = _image(image_dir, f"1/twilight/{old_source1}.png")
    old_source2 = "pony_chart_20250102_000000_000000_old00002"
    old2 = _image(image_dir, f"1/rarity/{old_source2}.png")
    new1 = _image(image_dir, "pony_chart_20260825_000001_000000_new00001.png")
    new2 = _image(image_dir, "pony_chart_20260825_000002_000000_new00002.png")
    store.set(store.path_to_key(old1), [1])
    store.set(store.path_to_key(old2), [2])
    store.save()
    app = _FakeApp(new1, store)

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for",
        lambda filename, labels: image_dir / filename,
    )
    monkeypatch.setattr(
        "app.label_images.app.organize_single", lambda path, labels: path
    )
    actions = _LabelActions(app)  # type: ignore[arg-type]

    # 第 1 張新圖：不淘汰。
    app.current_labels = [1]
    actions.save()
    assert old1.exists()
    assert old2.exists()

    # 第 2 張新圖：淘汰最舊的一組。
    app.nav.current_path = new2
    app.current_labels = [2]
    actions.save()
    assert not old1.exists()
    assert old2.exists()

    # 重複儲存同一張新圖：不再淘汰。
    app.current_labels = [3]
    actions.save()
    assert old2.exists()
    assert store.get(store.path_to_key(new2)) == [3]
    assert app.messages == ["saved", "saved", "saved"]


def test_already_labeled_original_never_plans_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    prior = _image(image_dir, "pony_chart_20260825_000001_000000_prior001.png")
    source = "pony_chart_20260825_000002_000000_new00001"
    original = _image(image_dir, f"{source}.png")
    key = store.path_to_key(original)
    store.set(store.path_to_key(prior), [1])
    store.set(key, [1])
    store.save()
    app = _FakeApp(original, store)
    app.current_labels = [2]
    calls: list[SourceSavePlan] = []

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: original
    )
    monkeypatch.setattr(
        "app.label_images.app.organize_single", lambda path, labels: path
    )

    def unexpected_retire(
        received_dir: Path,
        received_store: LabelStore,
        plan: SourceSavePlan,
        *,
        before_retire: Callable[[str], object] | None = None,
    ) -> None:
        del received_dir, received_store, before_retire
        calls.append(plan)

    monkeypatch.setattr(
        "app.label_images.app.save_and_retire_oldest_sample", unexpected_retire
    )

    _LabelActions(app).save()  # type: ignore[arg-type]

    assert calls == []
    assert store.get(key) == [2]


def test_label_action_crop_save_never_triggers_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    prior = _image(image_dir, "pony_chart_20260825_000001_000000_prior001.png")
    store.set(store.path_to_key(prior), [1])
    source = "pony_chart_20260825_000002_000000_new00001"
    crop = _image(image_dir, f"{source}_crop1.png")
    app = _FakeApp(crop, store)
    calls: list[str] = []

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: crop
    )
    monkeypatch.setattr(
        "app.label_images.app.organize_single", lambda path, labels: path
    )
    monkeypatch.setattr(
        "app.label_images.app.save_and_retire_oldest_sample",
        lambda image_dir, label_store, plan, before_retire=None: calls.append(
            plan.source_stem
        ),
    )

    _LabelActions(app).save()  # type: ignore[arg-type]

    assert calls == []
    assert store.get(store.path_to_key(crop)) == [1]


def test_label_action_syncs_analysis_cache_and_cleans_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    prior = _image(image_dir, "pony_chart_20260825_000001_000000_prior001.png")
    store.set(store.path_to_key(prior), [1])
    new_source = "pony_chart_20260825_000002_000000_new00001"
    new = _image(image_dir, f"{new_source}.png")
    old_dir = image_dir / "1" / "twilight"
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"1/twilight/{old_source}.png")
    old_key = store.path_to_key(old)
    app = _FakeApp(new, store)

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: new
    )
    monkeypatch.setattr(
        "app.label_images.app.organize_single", lambda path, labels: path
    )

    def retire(
        received_dir: Path,
        received_store: LabelStore,
        plan: SourceSavePlan,
        *,
        before_retire: Callable[[str], object] | None = None,
    ) -> RetiredSample:
        if before_retire is not None:
            before_retire(old_source)
        assert received_dir == image_dir
        assert plan.source_stem == new_source
        old.unlink()
        received_store.save()
        return RetiredSample(old_source, (old,), (old_key,))

    monkeypatch.setattr("app.label_images.app.save_and_retire_oldest_sample", retire)

    _LabelActions(app).save()  # type: ignore[arg-type]

    assert app.nav.removed == [old]
    assert app.analysis.deleted == [old_source]
    assert app.analysis.save_cache_fail_safe_calls == 1
    assert not old_dir.exists()


def test_label_action_rolls_back_new_file_and_memory_when_retirement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    prior = _image(image_dir, "pony_chart_20260825_000001_000000_prior001.png")
    prior_key = store.path_to_key(prior)
    store.set(prior_key, [1])
    store.save()
    labels_before = (image_dir / "labels.json").read_bytes()
    source = "pony_chart_20260825_000002_000000_new00001"
    original = _image(image_dir, f"unlabeled/{source}.png")
    target = image_dir / "1" / "twilight" / original.name
    app = _FakeApp(original, store)

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: target
    )

    def organize(path: Path, labels: list[int]) -> Path:
        del labels
        target.parent.mkdir(parents=True)
        path.replace(target)
        return target

    def fail_retirement(
        received_dir: Path,
        received_store: LabelStore,
        plan: SourceSavePlan,
        *,
        before_retire: Callable[[str], object] | None = None,
    ) -> None:
        del received_dir, received_store, plan, before_retire
        raise OSError("simulated retirement failure")

    monkeypatch.setattr("app.label_images.app.organize_single", organize)
    monkeypatch.setattr(
        "app.label_images.app.save_and_retire_oldest_sample", fail_retirement
    )

    with pytest.raises(OSError, match="simulated retirement failure"):
        _LabelActions(app).save()  # type: ignore[arg-type]

    assert original.exists()
    assert not target.exists()
    assert store.all_keys() == [prior_key]
    assert (image_dir / "labels.json").read_bytes() == labels_before


def test_label_action_retirement_keeps_current_new_image_with_real_navigator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    store = _store(image_dir, monkeypatch)
    old_source = "pony_chart_20250101_000000_000000_old00001"
    old = _image(image_dir, f"{old_source}.png")
    prior = _image(image_dir, "pony_chart_20260825_000001_000000_prior001.png")
    new_source = "pony_chart_20260825_000002_000000_new00001"
    new = _image(image_dir, f"{new_source}.png")
    later = _image(
        image_dir,
        "pony_chart_20260825_000003_000000_new00002.png",
    )
    store.set(store.path_to_key(old), [1])
    store.set(store.path_to_key(prior), [1])
    store.save()
    nav = ImageNavigator([old, prior, new, later], store)
    nav.go_to_key(store.path_to_key(new))
    app = _FakeApp(new, store)
    app.nav = nav  # type: ignore[assignment]

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: new
    )
    monkeypatch.setattr(
        "app.label_images.app.organize_single", lambda path, labels: path
    )

    _LabelActions(app).save()  # type: ignore[arg-type]

    assert nav.current_path == new
    assert nav.all_paths == [prior, new, later]


def test_label_action_organize_persists_analysis_and_cleans_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    source_dir = image_dir / "unlabeled"
    source_dir.mkdir(parents=True)
    store = _store(image_dir, monkeypatch)
    source = "pony_chart_20260824_235959_000000_before01"
    original = _image(image_dir, f"unlabeled/{source}.png")
    target = image_dir / "1" / "twilight" / original.name
    app = _FakeApp(original, store)

    monkeypatch.setattr("app.label_images.app.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.app.target_path_for", lambda filename, labels: target
    )

    def organize(path: Path, labels: list[int]) -> Path:
        del labels
        target.parent.mkdir(parents=True)
        path.replace(target)
        return target

    monkeypatch.setattr("app.label_images.app.organize_single", organize)

    _LabelActions(app).save()  # type: ignore[arg-type]

    assert app.nav.current_path == target
    assert app.analysis.renamed == [
        (store.path_to_key(original), store.path_to_key(target))
    ]
    assert app.analysis.save_cache_fail_safe_calls == 1
    assert not source_dir.exists()

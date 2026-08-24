import json
from pathlib import Path

import pytest

from app.label_images.analysis import AnalysisManager
from app.label_images.file_actions import FileActions
from app.label_images.mutation_guard import RawImageMutationGuard
from app.label_images.retirement_journal import (
    RetirementRecoveryError,
    journal_path_for,
    prepare_retirement_journal,
)


class _FakeNav:
    def __init__(self, paths: list[Path]) -> None:
        self._all_paths = list(paths)
        self._paths = list(paths)
        self._idx = 0

    @property
    def is_empty(self) -> bool:
        return not self._paths

    @property
    def current_path(self) -> Path:
        return self._paths[self._idx]

    @property
    def current_key(self) -> str:
        return self.current_path.name

    @property
    def all_paths(self) -> list[Path]:
        return self._all_paths

    def remove_path(self, path: Path) -> None:
        self._all_paths = [p for p in self._all_paths if p != path]
        self._paths = [p for p in self._paths if p != path]
        self._idx = 0

    def replace_path(self, old_path: Path, new_path: Path) -> None:
        self._all_paths = [new_path if p == old_path else p for p in self._all_paths]
        self._paths = [new_path if p == old_path else p for p in self._paths]

    def sync_with_disk(self, disk_paths: list[Path]) -> int:
        existing = set(self._all_paths)
        new_paths = [p for p in disk_paths if p not in existing]
        self._all_paths.extend(new_paths)
        self._paths.extend(new_paths)
        return len(new_paths)


class _FakeStore:
    def __init__(
        self, labels: dict[str, list[int]], orphans: list[str] | None = None
    ) -> None:
        self._labels = dict(labels)
        self._orphans = orphans or []
        self.saved = False

    def get(self, key: str) -> list[int]:
        return list(self._labels.get(key, []))

    def delete(self, key: str) -> None:
        self._labels.pop(key, None)

    def save(self) -> None:
        self.saved = True

    def purge_orphans(self, base_dir: Path) -> list[str]:
        del base_dir
        for key in self._orphans:
            self._labels.pop(key, None)
        return list(self._orphans)

    def rename_key(self, old_key: str, new_key: str) -> None:
        if old_key in self._labels:
            self._labels[new_key] = self._labels.pop(old_key)

    def path_to_key(self, path: Path) -> str:
        return path.name


def _pending_journal(image_dir: Path, journal_kind: str) -> Path:
    journal = journal_path_for(image_dir)
    if journal_kind == "prepared":
        prepare_retirement_journal(
            image_dir,
            "pony_chart_20260826_000001_000000_journal1",
            (),
            None,
            None,
        )
    else:
        journal.write_text("{malformed", encoding="utf-8")
    return journal


@pytest.mark.parametrize("journal_kind", ["prepared", "malformed"])
def test_delete_crop_is_unchanged_when_recovery_journal_is_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    journal_kind: str,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    crop = image_dir / "pony_chart_20260826_000001_000000_source01_crop1.png"
    crop.write_bytes(b"crop")
    nav = _FakeNav([crop])
    store = _FakeStore({crop.name: [1]})
    analysis = AnalysisManager()
    probs = [0.5, 0.2, 0.1, 0.1, 0.1, 0.1]
    analysis.model_probs = {crop.name: probs}
    journal = _pending_journal(image_dir, journal_kind)
    journal_before = journal.read_bytes()
    confirmation_requested = False

    def unexpected_confirmation(title: str, message: str) -> bool:
        del title, message
        nonlocal confirmation_requested
        confirmation_requested = True
        return True

    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.askyesno",
        unexpected_confirmation,
    )
    guard = RawImageMutationGuard(image_dir)
    actions = FileActions(nav, store, analysis, guard)  # type: ignore[arg-type]

    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        actions.delete_crop()

    assert confirmation_requested is False
    assert crop.read_bytes() == b"crop"
    assert nav.all_paths == [crop]
    assert nav.current_path == crop
    assert store._labels == {crop.name: [1]}
    assert store.saved is False
    assert analysis.model_probs == {crop.name: probs}
    assert journal.read_bytes() == journal_before


@pytest.mark.parametrize("journal_kind", ["prepared", "malformed"])
def test_organize_all_is_unchanged_when_recovery_journal_is_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    journal_kind: str,
) -> None:
    image_dir = tmp_path / "rawimage"
    source = image_dir / "unlabeled" / "pony_chart_20260826_000001_000000_source01.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    nav = _FakeNav([source])
    store = _FakeStore({source.name: [1]})
    analysis = AnalysisManager()
    probs = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    analysis.model_probs = {source.name: probs}
    journal = _pending_journal(image_dir, journal_kind)
    journal_before = journal.read_bytes()
    scan_started = False

    def unexpected_dedup(paths: list[Path]) -> list[tuple[Path, Path]]:
        del paths
        nonlocal scan_started
        scan_started = True
        return []

    monkeypatch.setattr(
        "app.label_images.file_actions.dedup_images",
        unexpected_dedup,
    )
    guard = RawImageMutationGuard(image_dir)
    actions = FileActions(nav, store, analysis, guard)  # type: ignore[arg-type]

    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        actions.organize_all()

    assert scan_started is False
    assert source.read_bytes() == b"source"
    assert nav.all_paths == [source]
    assert nav.current_path == source
    assert store._labels == {source.name: [1]}
    assert store.saved is False
    assert analysis.model_probs == {source.name: probs}
    assert journal.read_bytes() == journal_before


def test_reload_remains_allowed_with_pending_recovery_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    existing = image_dir / "existing.png"
    added = image_dir / "added.png"
    existing.write_bytes(b"existing")
    added.write_bytes(b"added")
    nav = _FakeNav([existing])
    store = _FakeStore({})
    analysis = AnalysisManager()
    journal = _pending_journal(image_dir, "malformed")
    journal_before = journal.read_bytes()
    monkeypatch.setattr("app.label_images.file_actions.IMAGE_DIR", image_dir)
    monkeypatch.setattr(
        "app.label_images.file_actions.scan_image_paths",
        lambda base: [existing, added],
    )
    guard = RawImageMutationGuard(image_dir)
    actions = FileActions(nav, store, analysis, guard)  # type: ignore[arg-type]

    assert actions.reload() == 1
    assert nav.all_paths == [existing, added]
    assert journal.read_bytes() == journal_before


def test_organize_all_preserves_analysis_results_after_move(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.png"
    new_path = tmp_path / "organized.png"
    old_path.write_bytes(b"img")

    nav = _FakeNav([old_path])
    store = _FakeStore({"old.png": [1]})
    analysis = AnalysisManager()
    analysis.model_probs = {"old.png": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]}

    monkeypatch.setattr("app.label_images.file_actions.dedup_images", lambda paths: [])
    monkeypatch.setattr(
        "app.label_images.file_actions.dedup_near_images", lambda paths: []
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.target_path_for",
        lambda filename, labels: new_path if filename == "old.png" else old_path,
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.organize_single",
        lambda current_path, labels: new_path,
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.cleanup_empty_dirs", lambda base: None
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.askyesno",
        lambda title, message: True,
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.showinfo",
        lambda title, message: None,
    )

    actions = FileActions(nav, store, analysis)  # type: ignore[arg-type]
    actions.organize_all()

    assert analysis.model_probs == {"organized.png": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]}
    assert store._labels == {"organized.png": [1]}
    assert nav.all_paths == [new_path]
    assert store.saved


def test_organize_all_persists_analysis_cache_after_orphan_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kept_path = tmp_path / "kept.png"
    kept_path.write_bytes(b"img")

    nav = _FakeNav([kept_path])
    store = _FakeStore({"kept.png": [1]}, orphans=["deleted.png"])
    analysis = AnalysisManager()
    analysis.model_probs = {
        "kept.png": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        "deleted.png": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
    }
    cache_path = tmp_path / "analysis_cache.json"
    analysis._cache_path = cache_path

    monkeypatch.setattr("app.label_images.file_actions.dedup_images", lambda paths: [])
    monkeypatch.setattr(
        "app.label_images.file_actions.dedup_near_images", lambda paths: []
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.target_path_for",
        lambda filename, labels: kept_path,
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.cleanup_empty_dirs", lambda base: None
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.showinfo",
        lambda title, message: None,
    )

    actions = FileActions(nav, store, analysis)  # type: ignore[arg-type]
    actions.organize_all()

    assert analysis.model_probs == {"kept.png": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]}
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved["probs"] == {"kept.png": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]}


def test_reload_adds_new_images_found_on_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "old.png"
    existing.write_bytes(b"img")
    new_image = tmp_path / "new.png"

    nav = _FakeNav([existing])
    store = _FakeStore({"old.png": [1]})
    analysis = AnalysisManager()

    monkeypatch.setattr("app.label_images.file_actions.IMAGE_DIR", tmp_path)

    actions = FileActions(nav, store, analysis)  # type: ignore[arg-type]
    assert actions.reload() == 0

    new_image.write_bytes(b"img")
    assert actions.reload() == 1
    assert new_image in nav.all_paths


def test_delete_crop_removes_analysis_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    crop_path = tmp_path / "pony_chart_20240101_000000_000000_abcdefgh_crop1.png"
    crop_path.write_bytes(b"crop")

    nav = _FakeNav([crop_path])
    store = _FakeStore({crop_path.name: [1]})
    analysis = AnalysisManager()
    analysis.model_probs = {crop_path.name: [0.5, 0.2, 0.1, 0.1, 0.1, 0.1]}

    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.askyesno",
        lambda title, message: True,
    )
    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.showwarning",
        lambda title, message: None,
    )

    actions = FileActions(nav, store, analysis)  # type: ignore[arg-type]

    assert actions.delete_crop() is True
    assert analysis.model_probs == {}
    assert store._labels == {}
    assert not crop_path.exists()


def test_delete_crop_rejects_unique_source_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "pony_chart_20260812_134140_349047_py9p_4l3.png"
    source_path.write_bytes(b"source")

    nav = _FakeNav([source_path])
    store = _FakeStore({source_path.name: [1]})
    analysis = AnalysisManager()
    analysis.model_probs = {source_path.name: [0.5, 0.2, 0.1, 0.1, 0.1, 0.1]}
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "app.label_images.file_actions.messagebox.showwarning",
        lambda title, message: warnings.append((title, message)),
    )

    actions = FileActions(nav, store, analysis)  # type: ignore[arg-type]

    assert actions.delete_crop() is False
    assert warnings == [("刪除", "只能刪除明確認出的裁切圖。")]
    assert source_path.exists()
    assert store._labels == {source_path.name: [1]}
    assert source_path.name in analysis.model_probs

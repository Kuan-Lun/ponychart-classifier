import json
from pathlib import Path

import pytest

from app.label_images.analysis import AnalysisManager
from app.label_images.file_actions import FileActions


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

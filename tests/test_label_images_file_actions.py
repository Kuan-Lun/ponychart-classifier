from __future__ import annotations

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


class _FakeStore:
    def __init__(self, labels: dict[str, list[int]]) -> None:
        self._labels = dict(labels)
        self.saved = False

    def get(self, key: str) -> list[int]:
        return list(self._labels.get(key, []))

    def delete(self, key: str) -> None:
        self._labels.pop(key, None)

    def save(self) -> None:
        self.saved = True

    def purge_orphans(self, base_dir: Path) -> list[str]:
        del base_dir
        return []

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


def test_delete_crop_removes_analysis_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    crop_path = tmp_path / "pony_chart_20240101_000000_1.png"
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

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.label_images.analysis import AnalysisManager
from app.label_images.app import _AnalysisActions


class _FakeWidget:
    def __init__(self) -> None:
        self.state = "normal"
        self.text = ""

    def configure(self, **kwargs: str) -> None:
        self.__dict__.update(kwargs)


class _FakeNav:
    def __init__(self, all_paths: list[Path]) -> None:
        self.all_paths = all_paths


class _FakeStore:
    def __init__(self, labels: dict[str, list[int]]) -> None:
        self._labels = labels

    def path_to_key(self, path: Path) -> str:
        return path.name

    def has(self, key: str) -> bool:
        return key in self._labels


class _FakeApp:
    def __init__(self, all_paths: list[Path], labels: dict[str, list[int]]) -> None:
        self.nav = _FakeNav(all_paths)
        self.store = _FakeStore(labels)
        self.analysis = AnalysisManager()
        self.analyze_btn = _FakeWidget()
        self.analyze_unlabeled_btn = _FakeWidget()
        self.analyze_status = _FakeWidget()


def test_update_button_states_reenables_after_new_unanalyzed_crop(
    tmp_path: Path,
) -> None:
    """裁切出新圖片後，即便兩個按鈕先前已停用，也要重新啟用。"""
    old = tmp_path / "old.png"
    new_crop = tmp_path / "new_crop.png"

    app = _FakeApp([old, new_crop], labels={"old.png": [1]})
    app.analysis.model_probs = {"old.png": [0.1] * 6}
    app.analyze_btn.state = "disabled"
    app.analyze_unlabeled_btn.state = "disabled"

    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions.update_button_states()

    assert app.analyze_btn.state == "normal"
    assert app.analyze_unlabeled_btn.state == "normal"


def test_update_button_states_disables_when_everything_analyzed(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"

    app = _FakeApp([a, b], labels={"a.png": [1], "b.png": [2]})
    app.analysis.model_probs = {"a.png": [0.1] * 6, "b.png": [0.2] * 6}
    app.analyze_btn.state = "normal"
    app.analyze_unlabeled_btn.state = "normal"

    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions.update_button_states()

    assert app.analyze_btn.state == "disabled"
    assert app.analyze_unlabeled_btn.state == "disabled"


def test_update_button_states_full_only_when_no_unlabeled_left(
    tmp_path: Path,
) -> None:
    """已標註但尚未分析的圖片只該啟用「自動標註」，不啟用「僅未標註」。"""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"

    app = _FakeApp([a, b], labels={"a.png": [1], "b.png": [2]})
    app.analysis.model_probs = {"a.png": [0.1] * 6}

    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions.update_button_states()

    assert app.analyze_btn.state == "normal"
    assert app.analyze_unlabeled_btn.state == "disabled"


def test_update_button_states_skipped_while_running(tmp_path: Path) -> None:
    old = tmp_path / "old.png"
    new_crop = tmp_path / "new_crop.png"

    app = _FakeApp([old, new_crop], labels={"old.png": [1]})
    app.analysis.model_probs = {"old.png": [0.1] * 6}
    app.analyze_btn.state = "disabled"
    app.analyze_unlabeled_btn.state = "disabled"

    block = threading.Event()
    thread = threading.Thread(target=block.wait, daemon=True)
    thread.start()
    app.analysis._thread = thread
    try:
        actions = _AnalysisActions(app)  # type: ignore[arg-type]
        actions.update_button_states()

        assert app.analyze_btn.state == "disabled"
        assert app.analyze_unlabeled_btn.state == "disabled"
    finally:
        block.set()
        thread.join()


def test_on_error_recomputes_button_states_instead_of_forcing_normal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"

    app = _FakeApp([a, b], labels={"a.png": [1], "b.png": [2]})
    app.analysis.model_probs = {"a.png": [0.1] * 6, "b.png": [0.2] * 6}
    app.analyze_btn.state = "disabled"
    app.analyze_unlabeled_btn.state = "disabled"

    monkeypatch.setattr(
        "app.label_images.app.messagebox.showerror",
        lambda title, message: None,
    )

    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions._on_error("boom")

    assert app.analyze_status.text == ""
    assert app.analyze_btn.state == "disabled"
    assert app.analyze_unlabeled_btn.state == "disabled"

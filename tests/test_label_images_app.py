from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.label_images.analysis import AnalysisManager
from app.label_images.app import _AnalysisActions, _CropActions
from ponychart_classifier.training.sampling import Sample


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


class _FakeCropViewer:
    def __init__(self, save_path: Path | None) -> None:
        self.save_crop_calls: list[Path] = []
        self._save_path = save_path

    def save_crop(self, current_path: Path) -> Path | None:
        self.save_crop_calls.append(current_path)
        return self._save_path


class _FakeCropNav:
    def __init__(self, current_path: Path) -> None:
        self.current_path = current_path
        self.added_paths: list[Path] = []

    def add_path(self, path: Path) -> None:
        self.added_paths.append(path)
        self.current_path = path


class _FakeCropStore:
    def path_to_key(self, p: Path) -> str:
        return p.name

    def get(self, key: str) -> list[int]:
        del key
        return []


class _FakeCropAnalysisActions:
    def __init__(self) -> None:
        self.analyze_new_crop_calls: list[tuple[Path, str]] = []

    def analyze_new_crop(self, path: Path, key: str) -> None:
        self.analyze_new_crop_calls.append((path, key))


class _FakeCropApp:
    def __init__(
        self, current_path: Path, save_path: Path | None, current_labels: list[int]
    ) -> None:
        self.viewer = _FakeCropViewer(save_path)
        self.nav = _FakeCropNav(current_path)
        self.store = _FakeCropStore()
        self.analysis_actions = _FakeCropAnalysisActions()
        self.current_labels = current_labels
        self.refresh_calls = 0
        self.update_display_calls: list[str] = []

    def refresh(self) -> None:
        self.refresh_calls += 1
        # 模擬 LabelApp.refresh()：新裁切圖在 labels.json 中尚無紀錄，故清空。
        key = self.store.path_to_key(self.nav.current_path)
        self.current_labels = self.store.get(key)

    def update_display(self, extra: str = "") -> None:
        self.update_display_calls.append(extra)


def test_crop_save_triggers_auto_analysis_for_new_crop(tmp_path: Path) -> None:
    """裁切完成後，應自動對新裁切圖觸發模型推論（寫入 analysis_cache.json）。

    不直接寫入 labels.json，新裁切圖仍維持未標註，待使用者確認。
    """
    src = tmp_path / "src.png"
    crop_path = tmp_path / "src_crop1.png"
    app = _FakeCropApp(current_path=src, save_path=crop_path, current_labels=[1, 3])

    actions = _CropActions(app)  # type: ignore[arg-type]
    actions.save()

    assert app.analysis_actions.analyze_new_crop_calls == [(crop_path, crop_path.name)]
    assert app.current_labels == []
    assert app.nav.added_paths == [crop_path]
    assert app.refresh_calls == 1
    assert app.update_display_calls == [f"已儲存裁切圖：{crop_path.name}"]


def test_crop_save_noop_when_save_crop_fails(tmp_path: Path) -> None:
    src = tmp_path / "src.png"
    app = _FakeCropApp(current_path=src, save_path=None, current_labels=[2])

    actions = _CropActions(app)  # type: ignore[arg-type]
    actions.save()

    assert app.refresh_calls == 0
    assert app.update_display_calls == []
    assert app.analysis_actions.analyze_new_crop_calls == []
    assert app.current_labels == [2]


def test_analyze_new_crop_launches_single_sample_analysis(tmp_path: Path) -> None:
    """裁切觸發的自動分析應對單張新圖啟動推論，而非整批重新分析。"""
    crop_path = tmp_path / "src_crop1.png"
    app = _FakeApp([crop_path], labels={})

    calls: list[tuple[list[Sample], list[str]]] = []
    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions._launch = lambda samples, keys: calls.append((samples, keys))  # type: ignore[method-assign]

    actions.analyze_new_crop(crop_path, "src_crop1.png")

    assert calls == [([Sample(str(crop_path), [])], ["src_crop1.png"])]

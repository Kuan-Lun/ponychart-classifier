import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import ponychart_classifier
from app.label_images import __main__ as label_images_main
from app.label_images.analysis import AnalysisManager
from app.label_images.app import _AnalysisActions, _CropActions
from app.label_images.mutation_guard import RawImageMutationGuard
from app.label_images.retirement_journal import (
    RetirementRecoveryError,
    journal_path_for,
    prepare_retirement_journal,
)
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


def test_refresh_staleness_clears_cache_when_model_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    manager.model_probs = {"a.png": [0.1] * 6}
    manager.model_thresholds = [0.5] * 6
    monkeypatch.setattr(ponychart_classifier, "has_pending_update", lambda: True)

    changed = manager.refresh_staleness()

    assert changed is True
    assert manager.model_probs is None
    assert manager.model_thresholds is None


def test_refresh_staleness_keeps_cache_when_model_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    manager.model_probs = {"a.png": [0.1] * 6}
    manager.model_thresholds = [0.5] * 6
    monkeypatch.setattr(ponychart_classifier, "has_pending_update", lambda: False)

    changed = manager.refresh_staleness()

    assert changed is False
    assert manager.model_probs == {"a.png": [0.1] * 6}


def test_save_cache_fail_safe_invalidates_stale_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    cache_path = tmp_path / "analysis_cache.json"
    cache_path.write_text("stale", encoding="utf-8")
    manager._cache_path = cache_path
    manager.model_probs = {"old/path.png": [0.1] * 6}
    manager.model_thresholds = [0.5] * 6

    def fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr("app.label_images.analysis.prob_cache.save", fail_save)

    assert manager.save_cache_fail_safe() is False
    assert manager.model_probs is None
    assert manager.model_thresholds is None
    assert not cache_path.exists()


def test_refresh_staleness_skips_network_check_when_no_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    manager.model_probs = None
    called = False

    def fake_has_pending_update() -> bool:
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(
        ponychart_classifier, "has_pending_update", fake_has_pending_update
    )

    changed = manager.refresh_staleness()

    assert changed is False
    assert called is False


def _prediction() -> SimpleNamespace:
    return SimpleNamespace(
        twilight_sparkle=0.1,
        rarity=0.2,
        fluttershy=0.3,
        rainbow_dash=0.4,
        pinkie_pie=0.5,
        applejack=0.6,
    )


def test_background_result_is_tombstoned_when_key_deleted_during_predict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    key = "1/twilight/pony_chart_20250101_000000_000000_old00001.png"
    ready = threading.Event()
    release = threading.Event()
    manager._active_keys = {key}
    monkeypatch.setattr(ponychart_classifier, "update", lambda: None)
    monkeypatch.setattr(
        ponychart_classifier,
        "get_thresholds",
        lambda: SimpleNamespace(as_list=lambda: [0.5] * 6),
    )

    def delayed_predict(path: str) -> SimpleNamespace:
        del path
        ready.set()
        assert release.wait(timeout=2)
        return _prediction()

    monkeypatch.setattr(ponychart_classifier, "predict", delayed_predict)
    worker = threading.Thread(
        target=manager._run,
        args=([Sample(str(tmp_path / "old.png"), [])], [key]),
    )
    worker.start()
    assert ready.wait(timeout=2)
    manager.delete_key(key)
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert manager._error is None
    assert manager._result == ({}, [0.5] * 6)
    assert key in manager._tombstones


def test_predict_failure_after_tombstone_is_safely_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AnalysisManager()
    key = "1/twilight/pony_chart_20250101_000000_000000_old00001.png"
    ready = threading.Event()
    release = threading.Event()
    manager._active_keys = {key}
    monkeypatch.setattr(ponychart_classifier, "update", lambda: None)
    monkeypatch.setattr(
        ponychart_classifier,
        "get_thresholds",
        lambda: SimpleNamespace(as_list=lambda: [0.5] * 6),
    )

    def missing_predict(path: str) -> SimpleNamespace:
        del path
        ready.set()
        assert release.wait(timeout=2)
        raise FileNotFoundError("retired while inference was pending")

    monkeypatch.setattr(ponychart_classifier, "predict", missing_predict)
    worker = threading.Thread(
        target=manager._run,
        args=([Sample(str(tmp_path / "old.png"), [])], [key]),
    )
    worker.start()
    assert ready.wait(timeout=2)
    manager.delete_key(key)
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert manager._error is None
    assert manager._result == ({}, [0.5] * 6)


def test_rename_updates_ready_result_and_tombstones_active_old_key() -> None:
    manager = AnalysisManager()
    old_key = "unlabeled/pony_chart_20250101_000000_000000_old00001.png"
    new_key = "1/twilight/pony_chart_20250101_000000_000000_old00001.png"
    manager._active_keys = {old_key}
    manager._result = ({old_key: [0.1] * 6}, [0.5] * 6)

    manager.rename_key(old_key, new_key)

    assert old_key in manager._tombstones
    assert manager._result == ({new_key: [0.1] * 6}, [0.5] * 6)


def test_purge_source_removes_cache_pending_and_active_scoped_keys() -> None:
    manager = AnalysisManager()
    source = "pony_chart_20250101_000000_000000_old00001"
    original = f"1/twilight/{source}.png"
    canonical_conflict = f"_conflicts/{source}_crop1_conflict2.png"
    legacy_conflict = f"_conflicts/{source}_crop2_1.png"
    active_crop = f"2/twilight+rarity/{source}_crop3.png"
    manual_similar = f"unlabeled/{source}_crop4_1.png"
    other = "1/rarity/pony_chart_20250102_000000_000000_other001.png"
    manager.model_probs = {
        original: [0.1] * 6,
        canonical_conflict: [0.2] * 6,
        manual_similar: [0.3] * 6,
        other: [0.4] * 6,
    }
    manager._result = (
        {legacy_conflict: [0.5] * 6, other: [0.4] * 6},
        [0.5] * 6,
    )
    manager._active_keys = {active_crop, manual_similar, other}

    removed = manager.purge_source(source)

    assert set(removed) == {
        original,
        canonical_conflict,
        legacy_conflict,
        active_crop,
    }
    assert manager.model_probs == {
        manual_similar: [0.3] * 6,
        other: [0.4] * 6,
    }
    assert manager._result == ({other: [0.4] * 6}, [0.5] * 6)
    assert manager._tombstones == {active_crop}


def test_startup_cache_orphan_purge_retains_any_existing_image(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "rawimage"
    labeled = image_dir / "1" / "twilight" / "labeled.png"
    unlabeled = image_dir / "unlabeled" / "unlabeled.png"
    labeled.parent.mkdir(parents=True)
    unlabeled.parent.mkdir(parents=True)
    labeled.write_bytes(b"labeled")
    unlabeled.write_bytes(b"unlabeled")
    manager = AnalysisManager()
    manager._cache_path = image_dir / "analysis_cache.json"
    manager._model_path = tmp_path / "model.onnx"
    manager._model_path.write_bytes(b"model")
    manager.model_probs = {
        "1/twilight/labeled.png": [0.1] * 6,
        "unlabeled/unlabeled.png": [0.2] * 6,
        "1/rarity/retired.png": [0.3] * 6,
        "../outside.png": [0.4] * 6,
    }

    removed = manager.purge_missing_files(image_dir)
    assert removed == ["../outside.png", "1/rarity/retired.png"]
    assert manager.save_cache_fail_safe()

    saved = json.loads(manager._cache_path.read_text(encoding="utf-8"))["probs"]
    assert saved == {
        "1/twilight/labeled.png": [0.1] * 6,
        "unlabeled/unlabeled.png": [0.2] * 6,
    }


def test_startup_recovers_before_scanning_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    events: list[str] = []
    fake_root = SimpleNamespace(mainloop=lambda: events.append("mainloop"))
    monkeypatch.setattr(label_images_main, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(label_images_main, "LABEL_FILE", image_dir / "labels.json")

    def recover(*args: object) -> bool:
        del args
        events.append("recover")
        return False

    monkeypatch.setattr(
        label_images_main,
        "recover_retirement_transaction",
        recover,
    )

    def scan(path: Path) -> list[Path]:
        del path
        events.append("scan")
        return []

    monkeypatch.setattr(
        label_images_main,
        "scan_image_paths",
        scan,
    )
    monkeypatch.setattr("app.label_images.__main__.tk.Tk", lambda: fake_root)
    monkeypatch.setattr(
        label_images_main,
        "LabelApp",
        lambda root, paths: events.append("app"),
    )

    label_images_main.main()

    assert events == ["recover", "scan", "app", "mainloop"]


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
        self.mutation_guard = RawImageMutationGuard(current_path.parent)
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


@pytest.mark.parametrize("journal_kind", ["prepared", "malformed"])
def test_crop_save_is_unchanged_when_recovery_journal_is_pending(
    tmp_path: Path,
    journal_kind: str,
) -> None:
    image_dir = tmp_path / "rawimage"
    image_dir.mkdir()
    source = image_dir / "pony_chart_20260826_000001_000000_source01.png"
    crop = image_dir / "pony_chart_20260826_000001_000000_source01_crop1.png"
    source.write_bytes(b"source")
    journal = journal_path_for(image_dir)
    if journal_kind == "prepared":
        prepare_retirement_journal(
            image_dir,
            "pony_chart_20260826_000001_000000_source01",
            (),
            None,
        )
    else:
        journal.write_text("{malformed", encoding="utf-8")
    journal_before = journal.read_bytes()
    app = _FakeCropApp(source, crop, [1, 3])

    with pytest.raises(RetirementRecoveryError, match="Pending retirement journal"):
        _CropActions(app).save()  # type: ignore[arg-type]

    assert source.read_bytes() == b"source"
    assert not crop.exists()
    assert journal.read_bytes() == journal_before
    assert app.viewer.save_crop_calls == []
    assert app.nav.current_path == source
    assert app.nav.added_paths == []
    assert app.current_labels == [1, 3]
    assert app.refresh_calls == 0
    assert app.update_display_calls == []
    assert app.analysis_actions.analyze_new_crop_calls == []


def test_analyze_new_crop_launches_single_sample_analysis(tmp_path: Path) -> None:
    """裁切觸發的自動分析應對單張新圖啟動推論，而非整批重新分析。"""
    crop_path = tmp_path / "src_crop1.png"
    app = _FakeApp([crop_path], labels={})

    calls: list[tuple[list[Sample], list[str]]] = []
    actions = _AnalysisActions(app)  # type: ignore[arg-type]
    actions._launch = lambda samples, keys: calls.append((samples, keys))  # type: ignore[method-assign]

    actions.analyze_new_crop(crop_path, "src_crop1.png")

    assert calls == [([Sample(str(crop_path), [])], ["src_crop1.png"])]

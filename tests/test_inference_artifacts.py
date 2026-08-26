from pathlib import Path
from typing import Any

import pytest

from ponychart_classifier.inference import artifacts
from ponychart_classifier.model_spec import PonyClass, parse_class_key


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _FakeOpener:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def urlopen(self, req: Any) -> _FakeResponse:
        del req
        return _FakeResponse(self._payload)


def test_clear_artifacts_deletes_only_owned_canonical_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_cache = tmp_path / "cache" / "ponychart-classifier"
    repo_artifacts = tmp_path / "artifacts"
    custom_dir = tmp_path / "custom"

    (runtime_cache / "model.onnx").parent.mkdir(parents=True)
    (runtime_cache / "model.onnx").write_bytes(b"model")
    (runtime_cache / "model.onnx.etag").write_text("model-etag", encoding="utf-8")
    (runtime_cache / "thresholds.json").write_text("{}", encoding="utf-8")
    (runtime_cache / "thresholds.json.etag").write_text(
        "thresholds-etag",
        encoding="utf-8",
    )
    (runtime_cache / "current.json").write_text("{}", encoding="utf-8")
    (runtime_cache / ".update.lock").write_bytes(b"\0")
    generation = runtime_cache / "generations" / "generation-1"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text("{}", encoding="utf-8")
    (runtime_cache / "foreign.cache").write_bytes(b"foreign")

    repo_artifacts.mkdir()
    (repo_artifacts / "model.onnx").write_bytes(b"repo-model")

    custom_dir.mkdir()
    (custom_dir / "model.onnx").write_bytes(b"custom-model")

    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_DIR", runtime_cache)

    artifacts.clear_artifacts()

    assert runtime_cache.is_dir()
    for owned_name in (
        "model.onnx",
        "model.onnx.etag",
        "thresholds.json",
        "thresholds.json.etag",
    ):
        assert not (runtime_cache / owned_name).exists()
    assert (runtime_cache / "current.json").read_text(encoding="utf-8") == "{}"
    assert (runtime_cache / ".update.lock").read_bytes() == b"\0"
    assert (generation / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert (runtime_cache / "foreign.cache").read_bytes() == b"foreign"
    assert (repo_artifacts / "model.onnx").exists()
    assert (custom_dir / "model.onnx").exists()


def test_clear_artifacts_is_idempotent_with_partial_owned_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_cache = tmp_path / "cache" / "ponychart-classifier"
    runtime_cache.mkdir(parents=True)
    (runtime_cache / "model.onnx").write_bytes(b"model")
    (runtime_cache / "thresholds.json.etag").write_text(
        "thresholds-etag",
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_DIR", runtime_cache)

    artifacts.clear_artifacts()
    artifacts.clear_artifacts()

    assert runtime_cache.is_dir()
    assert not (runtime_cache / "model.onnx").exists()
    assert not (runtime_cache / "thresholds.json.etag").exists()


def test_clear_artifacts_unlinks_owned_symlinks_without_touching_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_cache = tmp_path / "cache" / "ponychart-classifier"
    runtime_cache.mkdir(parents=True)
    external_model = tmp_path / "external-model.onnx"
    external_thresholds = tmp_path / "external-thresholds.json"
    external_model.write_bytes(b"external-model")
    external_thresholds.write_text('{"external": 0.5}', encoding="utf-8")
    try:
        (runtime_cache / "model.onnx").symlink_to(external_model)
        (runtime_cache / "thresholds.json").symlink_to(external_thresholds)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")
    monkeypatch.setattr(artifacts, "DEFAULT_ARTIFACT_DIR", runtime_cache)

    artifacts.clear_artifacts()

    assert not (runtime_cache / "model.onnx").exists()
    assert not (runtime_cache / "thresholds.json").exists()
    assert external_model.read_bytes() == b"external-model"
    assert external_thresholds.read_text(encoding="utf-8") == '{"external": 0.5}'


def test_update_artifact_skips_download_when_etag_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"current-model")
    artifacts.save_etag(path, "same-etag")

    downloaded = False

    def fake_download(download_path: Path, filename: str) -> None:
        nonlocal downloaded
        del download_path, filename
        downloaded = True

    monkeypatch.setattr(artifacts, "remote_etag", lambda filename: "same-etag")
    monkeypatch.setattr(artifacts, "download_artifact", fake_download)

    updated = artifacts.update_artifact(path, artifacts.MODEL_FILENAME)

    assert updated is False
    assert downloaded is False
    assert path.read_bytes() == b"current-model"
    assert artifacts.local_etag(path) == "same-etag"


def test_update_artifact_downloads_and_updates_etag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old-model")
    artifacts.save_etag(path, "old-etag")

    def fake_download(download_path: Path, filename: str) -> None:
        assert download_path == path
        assert filename == artifacts.MODEL_FILENAME
        download_path.write_bytes(b"new-model")

    monkeypatch.setattr(artifacts, "remote_etag", lambda filename: "new-etag")
    monkeypatch.setattr(artifacts, "download_artifact", fake_download)

    updated = artifacts.update_artifact(path, artifacts.MODEL_FILENAME)

    assert updated is True
    assert path.read_bytes() == b"new-model"
    assert artifacts.local_etag(path) == "new-etag"


def test_download_artifact_replaces_target_without_tmp_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old-bytes")

    monkeypatch.setattr(artifacts, "_opener", lambda: _FakeOpener(b"new-bytes"))

    artifacts.download_artifact(path, artifacts.MODEL_FILENAME)

    assert path.read_bytes() == b"new-bytes"
    assert list(tmp_path.glob("model.onnx.*.tmp")) == []


def test_parse_class_key_accepts_stable_id_and_display_name() -> None:
    assert parse_class_key("twilight_sparkle") is PonyClass.TWILIGHT_SPARKLE
    assert parse_class_key("Twilight Sparkle") is PonyClass.TWILIGHT_SPARKLE

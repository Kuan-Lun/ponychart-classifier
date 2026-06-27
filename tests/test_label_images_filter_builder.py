from pathlib import Path

from app.label_images.filter_builder import FilterConfig, build_filter_fn


class _FakeStore:
    def __init__(self, labels: dict[str, list[int]] | None = None) -> None:
        self._labels = labels or {}

    def path_to_key(self, path: Path) -> str:
        return path.name

    def get(self, key: str) -> list[int]:
        return list(self._labels.get(key, []))

    def has(self, key: str) -> bool:
        return key in self._labels


def test_no_filters_returns_none() -> None:
    config = FilterConfig()
    assert build_filter_fn(config, [], _FakeStore()) is None  # type: ignore[arg-type]


def test_crop_only_excludes_raw_images(tmp_path: Path) -> None:
    raw = tmp_path / "pony_chart_20260101_000000.png"
    crop = tmp_path / "pony_chart_20260101_000000_crop1.png"

    config = FilterConfig(crop_only=True)
    fn = build_filter_fn(config, [raw, crop], _FakeStore())  # type: ignore[arg-type]

    assert fn is not None
    assert fn(raw) is False
    assert fn(crop) is True

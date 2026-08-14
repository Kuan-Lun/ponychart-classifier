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
    raw = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh.png"
    crop = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh_crop1.png"

    config = FilterConfig(crop_only=True)
    fn = build_filter_fn(config, [raw, crop], _FakeStore())  # type: ignore[arg-type]

    assert fn is not None
    assert fn(raw) is False
    assert fn(crop) is True


def test_unique_source_name_is_raw_not_crop(tmp_path: Path) -> None:
    raw = tmp_path / "pony_chart_20260812_134140_349047_py9p_4l3.png"

    raw_fn = build_filter_fn(
        FilterConfig(raw_only=True),
        [raw],
        _FakeStore(),  # type: ignore[arg-type]
    )
    crop_fn = build_filter_fn(
        FilterConfig(crop_only=True),
        [raw],
        _FakeStore(),  # type: ignore[arg-type]
    )

    assert raw_fn is not None
    assert crop_fn is not None
    assert raw_fn(raw) is True
    assert crop_fn(raw) is False


def test_uncropped_filter_pairs_crop_with_full_unique_source_stem(
    tmp_path: Path,
) -> None:
    cropped_raw = tmp_path / "pony_chart_20260812_134140_349047_py9p_4l3.png"
    crop = tmp_path / "pony_chart_20260812_134140_349047_py9p_4l3_crop1.png"
    uncropped_raw = tmp_path / "pony_chart_20260812_134141_349047_abcdefgh.png"
    all_paths = [cropped_raw, crop, uncropped_raw]

    fn = build_filter_fn(
        FilterConfig(uncropped_only=True),
        all_paths,
        _FakeStore(),  # type: ignore[arg-type]
    )

    assert fn is not None
    assert fn(cropped_raw) is False
    assert fn(crop) is False
    assert fn(uncropped_raw) is True


def test_crop_only_rejects_unknown_filename(tmp_path: Path) -> None:
    unknown = tmp_path / "unrelated_crop1.png"

    fn = build_filter_fn(
        FilterConfig(crop_only=True),
        [unknown],
        _FakeStore(),  # type: ignore[arg-type]
    )

    assert fn is not None
    assert fn(unknown) is False

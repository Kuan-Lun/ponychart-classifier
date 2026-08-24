import datetime as dt

import pytest

from ponychart_classifier.image_names import (
    ParsedImageName,
    extract_source_stem,
    get_captured_at,
    get_source_stem,
    is_crop,
    is_original,
    parse_image_name,
)

_SOURCE = "pony_chart_20260812_134140_349047_py9p_4l3"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (f"{_SOURCE}.png", True),
        (f"{_SOURCE}.WEBP", True),
        (f"{_SOURCE}_conflict2.png", True),
        (f"{_SOURCE}_crop12.png", False),
        ("pony_chart_20240101_123456.png", False),
        ("unrelated.png", False),
    ],
)
def test_is_original_accepts_only_canonical_source_names(
    filename: str, expected: bool
) -> None:
    assert is_original(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (f"{_SOURCE}.png", False),
        (f"{_SOURCE}_crop12.png", True),
        (f"{_SOURCE}_crop12_conflict2.png", True),
        (f"{_SOURCE}_crop12_2.png", False),
        (f"{_SOURCE}_crop0.png", False),
        ("pony_chart_20240101_123456_crop1.png", False),
        ("pony_chart_20240101_123456_1.png", False),
        ("unrelated_crop1.png", False),
    ],
)
def test_is_crop_accepts_only_canonical_crop_names(
    filename: str, expected: bool
) -> None:
    assert is_crop(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (f"{_SOURCE}.png", _SOURCE),
        (f"{_SOURCE}_crop12.png", _SOURCE),
        (f"{_SOURCE}_conflict2.png", _SOURCE),
        (f"{_SOURCE}_crop12_conflict2.png", _SOURCE),
        ("pony_chart_20240101_123456.png", None),
        ("unrelated.png", None),
    ],
)
def test_extract_source_stem_accepts_only_canonical_names(
    filename: str, expected: str | None
) -> None:
    assert extract_source_stem(filename) == expected


def test_parse_image_name_returns_structured_identity() -> None:
    parsed = parse_image_name(f"{_SOURCE}_crop12.png")

    assert parsed is not None
    assert parsed.source_stem == _SOURCE
    assert parsed.captured_at == dt.datetime(2026, 8, 12, 13, 41, 40, 349047)
    assert parsed.crop_index == 12
    assert parsed.conflict_index is None
    assert parsed.is_original is False


def test_parse_image_name_preserves_conflict_source_identity() -> None:
    parsed = parse_image_name(f"{_SOURCE}_crop12_conflict3.png")

    assert parsed is not None
    assert parsed.source_stem == _SOURCE
    assert parsed.crop_index == 12
    assert parsed.conflict_index == 3
    assert parsed.is_original is False


def test_parsed_image_name_conflict_field_is_backward_compatible() -> None:
    parsed = ParsedImageName(
        _SOURCE,
        dt.datetime(2026, 8, 12, 13, 41, 40, 349047),
        None,
    )

    assert parsed.conflict_index is None
    assert parsed.is_original is True


@pytest.mark.parametrize(
    "filename",
    [
        "pony_chart_20260230_120000_000000_abcdefgh.png",
        "pony_chart_20260101_240000_000000_abcdefgh.png",
        "pony_chart_20260101_120000_1000000_abcdefgh.png",
        "pony_chart_20260101_120000_000000_abcdefg.png",
        "pony_chart_20260101_120000_000000_abcdefgh.txt",
    ],
)
def test_parse_image_name_rejects_invalid_current_names(filename: str) -> None:
    assert parse_image_name(filename) is None


def test_get_source_stem_keeps_full_source_identity() -> None:
    assert get_source_stem(f"{_SOURCE}_crop12.png") == _SOURCE


def test_get_source_stem_rejects_legacy_name() -> None:
    with pytest.raises(ValueError, match="Unsupported image filename"):
        get_source_stem("pony_chart_20240101_123456.png")


def test_get_captured_at_returns_full_timestamp() -> None:
    assert get_captured_at(f"{_SOURCE}.png") == dt.datetime(
        2026, 8, 12, 13, 41, 40, 349047
    )

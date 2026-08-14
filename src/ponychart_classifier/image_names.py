"""Canonical PonyChart image-name parsing.

Runtime code accepts one schema only::

    pony_chart_YYYYMMDD_HHMMSS_ffffff_<8-char-token>[_cropN].ext

Legacy names are handled exclusively by the one-time migration script.
"""

import dataclasses
import datetime as dt
import os
import re

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_SOURCE_STEM_PATTERN = (
    r"pony_chart_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<microsecond>\d{6})_"
    r"(?P<token>[a-z0-9_]{8})"
)
_IMAGE_STEM_RE = re.compile(
    rf"^(?P<source>{_SOURCE_STEM_PATTERN})(?:_crop(?P<crop>[1-9]\d*))?$"
)


@dataclasses.dataclass(frozen=True)
class ParsedImageName:
    """Structured canonical image identity."""

    source_stem: str
    captured_at: dt.datetime
    crop_index: int | None

    @property
    def is_original(self) -> bool:
        return self.crop_index is None


def parse_image_name(filename: str) -> ParsedImageName | None:
    """Parse a canonical filename or stem; return ``None`` when unsupported."""
    stem, suffix = os.path.splitext(os.path.basename(filename))
    if suffix and suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None
    match = _IMAGE_STEM_RE.fullmatch(stem)
    if match is None:
        return None
    try:
        captured_at = dt.datetime.strptime(
            "".join(
                (
                    match.group("date"),
                    match.group("time"),
                    match.group("microsecond"),
                )
            ),
            "%Y%m%d%H%M%S%f",
        )
    except ValueError:
        return None
    crop = match.group("crop")
    return ParsedImageName(
        source_stem=match.group("source"),
        captured_at=captured_at,
        crop_index=int(crop) if crop is not None else None,
    )


def is_original(filename: str) -> bool:
    """Return whether *filename* is a canonical source image."""
    parsed = parse_image_name(filename)
    return parsed is not None and parsed.is_original


def is_crop(filename: str) -> bool:
    """Return whether *filename* is a canonical crop image."""
    parsed = parse_image_name(filename)
    return parsed is not None and not parsed.is_original


def extract_source_stem(filename: str) -> str | None:
    """Return the canonical source stem shared by an original and its crops."""
    parsed = parse_image_name(filename)
    return parsed.source_stem if parsed is not None else None


def get_source_stem(filename: str) -> str:
    """Return the canonical source identity, raising for unsupported names."""
    parsed = parse_image_name(filename)
    if parsed is None:
        raise ValueError(f"Unsupported image filename: {filename}")
    return parsed.source_stem


def get_captured_at(filename: str) -> dt.datetime:
    """Return the capture timestamp encoded in a canonical image name."""
    parsed = parse_image_name(filename)
    if parsed is None:
        raise ValueError(f"Unsupported image filename: {filename}")
    return parsed.captured_at

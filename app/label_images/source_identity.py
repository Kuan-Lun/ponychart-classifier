"""Strict source-identity parsing shared by retirement and analysis cleanup."""

import os
import re
from pathlib import Path

from ponychart_classifier.image_names import ParsedImageName, parse_image_name

from .constants import CONFLICT_SUBDIR

_LEGACY_CONFLICT_RE = re.compile(r"^(?P<base>.+)_(?P<index>[1-9]\d*)$")


def _parse_identity(
    filename: str, *, allow_legacy_conflict: bool
) -> ParsedImageName | None:
    parsed = parse_image_name(filename)
    if parsed is not None or not allow_legacy_conflict:
        return parsed
    stem, suffix = os.path.splitext(Path(filename).name)
    match = _LEGACY_CONFLICT_RE.fullmatch(stem)
    if match is None:
        return None
    return parse_image_name(f"{match.group('base')}{suffix}")


def parse_path_identity(path: Path) -> ParsedImageName | None:
    """Parse canonical names plus app-created legacy names under `_conflicts`."""
    return _parse_identity(
        path.name,
        allow_legacy_conflict=CONFLICT_SUBDIR in path.parts[:-1],
    )


def parse_key_identity(key: str) -> ParsedImageName | None:
    """Parse a normalized relative cache/label key without broad fuzzy matching."""
    path = Path(key.replace("\\", "/"))
    return _parse_identity(
        path.name,
        allow_legacy_conflict=CONFLICT_SUBDIR in path.parts[:-1],
    )

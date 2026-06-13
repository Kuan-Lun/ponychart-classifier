from __future__ import annotations

from pathlib import Path

from app.label_images.navigator import ImageNavigator


class _FakeStore:
    def path_to_key(self, path: Path) -> str:
        return str(path)


def _make_paths(base: Path) -> list[Path]:
    paths = [
        base / "1" / "rarity" / "pony_chart_20260102_000000.png",
        base / "1" / "twilight" / "pony_chart_20260101_000000.png",
        base / "2" / "twilight+rarity" / "pony_chart_20260101_000000_crop1.png",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"img")
    paths.sort()
    return paths


def test_default_sort_mode_is_path(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    assert nav.sort_mode == "path"
    assert nav.all_paths == paths


def test_toggle_sort_mode_groups_crops_by_filename(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()

    assert nav.sort_mode == "filename"
    assert [p.name for p in nav.all_paths] == [
        "pony_chart_20260101_000000.png",
        "pony_chart_20260101_000000_crop1.png",
        "pony_chart_20260102_000000.png",
    ]


def test_toggle_sort_mode_twice_restores_path_order(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()
    nav.toggle_sort_mode()

    assert nav.sort_mode == "path"
    assert nav.all_paths == paths


def test_toggle_sort_mode_preserves_current_image(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    crop_path = next(p for p in paths if "crop1" in p.name)
    nav.go_to_key(str(crop_path))
    assert nav.current_path == crop_path

    nav.toggle_sort_mode()

    assert nav.current_path == crop_path


def test_sync_with_disk_adds_new_paths(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    original_count = len(paths)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    new_path = tmp_path / "unlabeled" / "pony_chart_20260103_000000.png"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"img")

    added = nav.sync_with_disk([*paths, new_path])

    assert added == 1
    assert new_path in nav.all_paths
    assert nav.total == original_count + 1


def test_sync_with_disk_returns_zero_when_nothing_new(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    assert nav.sync_with_disk(paths) == 0
    assert nav.all_paths == paths


def test_sync_with_disk_preserves_current_image(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    crop_path = next(p for p in paths if "crop1" in p.name)
    nav.go_to_key(str(crop_path))

    new_path = tmp_path / "unlabeled" / "pony_chart_20260103_000000.png"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"img")
    nav.sync_with_disk([*paths, new_path])

    assert nav.current_path == crop_path


def test_sync_with_disk_respects_filter(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.apply_filter(lambda p: "crop1" not in p.name)
    assert nav.total == 2

    new_path = tmp_path / "1" / "twilight" / "pony_chart_20260103_000000_crop1.png"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"img")
    added = nav.sync_with_disk([*paths, new_path])

    assert added == 1
    assert new_path in nav.all_paths
    assert nav.total == 2


def test_toggle_sort_mode_respects_filter(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.apply_filter(lambda p: "crop1" not in p.name)
    assert nav.total == 2

    nav.toggle_sort_mode()

    assert nav.total == 2
    visible = []
    for n in range(1, nav.total + 1):
        nav.go_to(n)
        visible.append(nav.current_path)
    assert all("crop1" not in p.name for p in visible)

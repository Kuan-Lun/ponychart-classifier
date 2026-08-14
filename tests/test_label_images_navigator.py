from pathlib import Path

from PIL import Image

from app.label_images.navigator import ImageNavigator


class _FakeStore:
    def path_to_key(self, path: Path) -> str:
        return str(path)


def _make_paths(base: Path) -> list[Path]:
    paths = [
        base / "1" / "rarity" / "pony_chart_20260102_000000_000000_abcdefgh.png",
        base / "1" / "twilight" / "pony_chart_20260101_000000_000000_abcdefgh.png",
        base
        / "2"
        / "twilight+rarity"
        / "pony_chart_20260101_000000_000000_abcdefgh_crop1.png",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"img")
    paths.sort()
    return paths


def _make_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


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
        "pony_chart_20260101_000000_000000_abcdefgh.png",
        "pony_chart_20260101_000000_000000_abcdefgh_crop1.png",
        "pony_chart_20260102_000000_000000_abcdefgh.png",
    ]


def test_toggle_sort_mode_cycles_through_four_modes(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()
    mode = nav.sort_mode
    assert mode == "filename"
    nav.toggle_sort_mode()
    mode = nav.sort_mode
    assert mode == "size"
    nav.toggle_sort_mode()
    mode = nav.sort_mode
    assert mode == "confidence"
    nav.toggle_sort_mode()
    mode = nav.sort_mode
    assert mode == "path"


def test_toggle_sort_mode_four_times_restores_path_order(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()
    nav.toggle_sort_mode()
    nav.toggle_sort_mode()
    nav.toggle_sort_mode()

    assert nav.sort_mode == "path"
    assert nav.all_paths == paths


def test_size_sort_mode_orders_by_min_dimension_ascending(tmp_path: Path) -> None:
    small = tmp_path / "small.png"
    medium = tmp_path / "medium.png"
    large = tmp_path / "large.png"
    _make_image(small, (100, 80))
    _make_image(medium, (300, 200))
    _make_image(large, (600, 500))

    nav = ImageNavigator([large, small, medium], _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()  # filename
    nav.toggle_sort_mode()  # size

    assert nav.all_paths == [small, medium, large]


def test_size_sort_mode_treats_unreadable_files_as_smallest(tmp_path: Path) -> None:
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not an image")
    real = tmp_path / "real.png"
    _make_image(real, (100, 80))

    nav = ImageNavigator([real, fake], _FakeStore())  # type: ignore[arg-type]

    nav.toggle_sort_mode()  # filename
    nav.toggle_sort_mode()  # size

    assert nav.all_paths == [fake, real]


def test_confidence_sort_mode_orders_by_min_distance_from_half_ascending(
    tmp_path: Path,
) -> None:
    confident = tmp_path / "confident.png"
    uncertain = tmp_path / "uncertain.png"
    no_probs = tmp_path / "no_probs.png"
    for p in (confident, uncertain, no_probs):
        p.write_bytes(b"img")

    probs = {
        str(confident): [0.95, 0.9, 0.1, 0.05, 0.99, 0.02],
        str(uncertain): [0.95, 0.9, 0.51, 0.05, 0.99, 0.02],
    }
    nav = ImageNavigator([confident, uncertain, no_probs], _FakeStore())  # type: ignore[arg-type]
    nav.set_probs_provider(lambda key: probs.get(key))

    nav.toggle_sort_mode()  # filename
    nav.toggle_sort_mode()  # size
    nav.toggle_sort_mode()  # confidence

    assert nav.all_paths == [uncertain, confident, no_probs]


def test_resort_reapplies_confidence_order_without_changing_mode(
    tmp_path: Path,
) -> None:
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    for p in (a_path, b_path):
        p.write_bytes(b"img")

    probs: dict[str, list[float]] = {}
    nav = ImageNavigator([a_path, b_path], _FakeStore())  # type: ignore[arg-type]
    nav.set_probs_provider(lambda key: probs.get(key))

    nav.toggle_sort_mode()  # filename
    nav.toggle_sort_mode()  # size
    nav.toggle_sort_mode()  # confidence
    assert nav.all_paths == [a_path, b_path]  # both missing, order unchanged

    # 模擬分析完成後，b 變成最不確定的圖片。
    probs[str(b_path)] = [0.51, 0.9, 0.1, 0.05, 0.99, 0.02]
    nav.resort()

    assert nav.sort_mode == "confidence"
    assert nav.all_paths == [b_path, a_path]


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

    new_path = tmp_path / "unlabeled" / "pony_chart_20260103_000000_000000_abcdefgh.png"
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

    new_path = tmp_path / "unlabeled" / "pony_chart_20260103_000000_000000_abcdefgh.png"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"img")
    nav.sync_with_disk([*paths, new_path])

    assert nav.current_path == crop_path


def test_sync_with_disk_respects_filter(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    nav = ImageNavigator(paths, _FakeStore())  # type: ignore[arg-type]

    nav.apply_filter(lambda p: "crop1" not in p.name)
    assert nav.total == 2

    new_path = (
        tmp_path
        / "1"
        / "twilight"
        / "pony_chart_20260103_000000_000000_abcdefgh_crop1.png"
    )
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

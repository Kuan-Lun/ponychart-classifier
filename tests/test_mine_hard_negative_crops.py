import random
from pathlib import Path

import pytest
from PIL import Image

import ponychart_classifier
from ponychart_classifier.inference import ClassThresholds, PredictionResult
from ponychart_classifier.training import Sample
from scripts import mine_hard_negative_crops as script

_THRESHOLDS = ClassThresholds(
    twilight_sparkle=0.5,
    rarity=0.5,
    fluttershy=0.5,
    rainbow_dash=0.5,
    pinkie_pie=0.5,
    applejack=0.5,
)


def _result(**scores: float) -> PredictionResult:
    base = dict.fromkeys(
        (
            "twilight_sparkle",
            "rarity",
            "fluttershy",
            "rainbow_dash",
            "pinkie_pie",
            "applejack",
        ),
        0.1,
    )
    base.update(scores)
    return PredictionResult(**base, labels=frozenset())


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size).save(path)


def _make_sample(tmp_path: Path, stem: str) -> Sample:
    src = tmp_path / f"{stem}.png"
    _make_image(src, (1200, 900))
    return Sample(str(src), [1])


def _touch_tmp(tmp_path: Path) -> Path:
    p = tmp_path / f"tmp_{random.randint(0, 10**9)}.png"
    _make_image(p, (10, 10))
    return p


# ---------------------------------------------------------------------------
# _misclassified_characters
# ---------------------------------------------------------------------------


def test_misclassified_characters_ignores_ground_truth_labels() -> None:
    # label 1 (twilight_sparkle) is ground truth; high score there shouldn't count.
    result = _result(twilight_sparkle=0.9, rarity=0.9)
    hits = script._misclassified_characters(result, _THRESHOLDS, [1])
    assert hits == [("rarity", 0.9)]


def test_misclassified_characters_no_hit_below_threshold() -> None:
    result = _result(rarity=0.4)
    assert script._misclassified_characters(result, _THRESHOLDS, [1]) == []


def test_misclassified_characters_multiple_hits() -> None:
    result = _result(rarity=0.9, fluttershy=0.7)
    hits = script._misclassified_characters(result, _THRESHOLDS, [1])
    assert hits == [("rarity", 0.9), ("fluttershy", 0.7)]


# ---------------------------------------------------------------------------
# find_candidate_pool
# ---------------------------------------------------------------------------


def test_find_candidate_pool_excludes_unlabeled_and_already_cropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        Sample("a/pony_chart_20260101_000000_000000_abcdefgh.png", [1]),
        Sample(
            "b/pony_chart_20260102_000000_000000_abcdefgh.png", []
        ),  # no labels -> excluded
        Sample(
            "c/pony_chart_20260103_000000_000000_abcdefgh.png", [2]
        ),  # already has a crop
    ]
    monkeypatch.setattr(script, "load_samples", lambda: samples)
    monkeypatch.setattr(script, "separate_orig_crop", lambda s: (s, []))
    monkeypatch.setattr(
        script,
        "has_existing_crop",
        lambda stem: stem == "pony_chart_20260103_000000_000000_abcdefgh",
    )

    pool = script.find_candidate_pool()
    assert [s.path for s in pool] == [
        "a/pony_chart_20260101_000000_000000_abcdefgh.png"
    ]


# ---------------------------------------------------------------------------
# _random_crop_box
# ---------------------------------------------------------------------------


def test_random_crop_box_within_bounds() -> None:
    rng = random.Random(0)
    img_w, img_h = 1200, 900
    box = script._random_crop_box(img_w, img_h, rng)
    assert box is not None
    corners, width, height = box
    assert width > 0
    assert height > 0
    assert corners.all_in_bounds(img_w, img_h)


def test_random_crop_box_none_when_image_too_small() -> None:
    rng = random.Random(0)
    # Far smaller than CROP_MIN_WIDTH_ORIG in every dimension/orientation.
    box = script._random_crop_box(10, 10, rng)
    assert box is None


# ---------------------------------------------------------------------------
# try_one_crop
# ---------------------------------------------------------------------------


def test_try_one_crop_returns_none_when_not_misclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh.png"
    _make_image(src, (1200, 900))
    sample = Sample(str(src), [1])

    monkeypatch.setattr(ponychart_classifier, "predict", lambda p: _result())

    rng = random.Random(0)
    assert script.try_one_crop(sample, _THRESHOLDS, rng) is None


def test_try_one_crop_returns_tmp_path_and_hits_on_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "pony_chart_20260101_000000_000000_abcdefgh.png"
    _make_image(src, (1200, 900))
    sample = Sample(str(src), [1])

    monkeypatch.setattr(ponychart_classifier, "predict", lambda p: _result(rarity=0.9))

    rng = random.Random(0)
    result = script.try_one_crop(sample, _THRESHOLDS, rng)
    assert result is not None
    out_path, hits = result
    assert out_path.exists()
    assert hits == [("rarity", 0.9)]
    out_path.unlink()


# ---------------------------------------------------------------------------
# mine (main loop)
# ---------------------------------------------------------------------------


def test_mine_stops_when_count_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [
        _make_sample(
            tmp_path,
            f"pony_chart_202601{i + 1:02d}_000000_000000_abcdefgh",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(script, "find_candidate_pool", lambda: list(samples))
    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(ponychart_classifier, "get_thresholds", lambda: _THRESHOLDS)
    monkeypatch.setattr(
        script,
        "try_one_crop",
        lambda sample, thresholds, rng: (_touch_tmp(tmp_path), [("rarity", 0.9)]),
    )

    successes = script.mine(count=2, max_failures=10, rng=random.Random(0))
    assert successes == 2


def test_mine_stops_on_max_consecutive_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_make_sample(tmp_path, "pony_chart_20260101_000000_000000_abcdefgh")]
    monkeypatch.setattr(script, "find_candidate_pool", lambda: list(samples))
    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(ponychart_classifier, "get_thresholds", lambda: _THRESHOLDS)
    monkeypatch.setattr(script, "try_one_crop", lambda sample, thresholds, rng: None)

    successes = script.mine(count=5, max_failures=3, rng=random.Random(0))
    assert successes == 0


def test_mine_stops_when_pool_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_make_sample(tmp_path, "pony_chart_20260101_000000_000000_abcdefgh")]
    monkeypatch.setattr(script, "find_candidate_pool", lambda: list(samples))
    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(ponychart_classifier, "get_thresholds", lambda: _THRESHOLDS)
    monkeypatch.setattr(
        script,
        "try_one_crop",
        lambda sample, thresholds, rng: (_touch_tmp(tmp_path), [("rarity", 0.9)]),
    )

    # Only one candidate available; asking for 5 successes should stop after 1.
    successes = script.mine(count=5, max_failures=10, rng=random.Random(0))
    assert successes == 1


def test_mine_does_not_retry_succeeded_sample_within_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_make_sample(tmp_path, "pony_chart_20260101_000000_000000_abcdefgh")]
    calls: list[Sample] = []

    def fake_try_one_crop(
        sample: Sample, thresholds: ClassThresholds, rng: random.Random
    ) -> tuple[Path, list[tuple[str, float]]]:
        calls.append(sample)
        return _touch_tmp(tmp_path), [("rarity", 0.9)]

    monkeypatch.setattr(script, "find_candidate_pool", lambda: list(samples))
    monkeypatch.setattr(script, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(ponychart_classifier, "get_thresholds", lambda: _THRESHOLDS)
    monkeypatch.setattr(script, "try_one_crop", fake_try_one_crop)

    # Asking for more successes than the pool can supply: should stop after
    # exhausting the single candidate, not call try_one_crop on it twice.
    successes = script.mine(count=5, max_failures=10, rng=random.Random(0))
    assert successes == 1
    assert len(calls) == 1

import datetime as dt

import numpy as np
import pytest

from ponychart_classifier.stats import GoFTestResult
from ponychart_classifier.training.constants import GOF_RECENT_ORIG_LIMIT
from ponychart_classifier.training.sampling import (
    Sample,
    prepare_balanced_samples,
    select_recent_original_keys,
    select_recent_original_samples,
    separate_orig_crop,
)


def _stem_for_index(idx: int) -> str:
    timestamp = dt.datetime(2024, 1, 1) + dt.timedelta(seconds=idx)
    return timestamp.strftime("pony_chart_%Y%m%d_%H%M%S_000000_abcdefgh")


def _key_for_index(idx: int) -> str:
    return f"rawimage/{_stem_for_index(idx)}.png"


def _sample_for_index(idx: int) -> Sample:
    return Sample(path=f"/tmp/{_key_for_index(idx)}", labels=[1])


def test_separate_orig_crop_rejects_legacy_name() -> None:
    samples = [Sample(path="/tmp/pony_chart_20240101_000000.png", labels=[1])]

    with pytest.raises(ValueError, match="migrate_rawimage_filenames"):
        separate_orig_crop(samples)


def test_select_recent_original_samples_returns_all_when_fewer_than_limit() -> None:
    samples = [
        _sample_for_index(2),
        _sample_for_index(0),
        _sample_for_index(1),
    ]

    selected = select_recent_original_samples(samples)

    assert [sample.path for sample in selected] == [
        _sample_for_index(2).path,
        _sample_for_index(1).path,
        _sample_for_index(0).path,
    ]


def test_select_recent_original_keys_keeps_all_when_count_matches_limit() -> None:
    keys = [_key_for_index(idx) for idx in range(GOF_RECENT_ORIG_LIMIT)]

    selected = select_recent_original_keys(keys)

    assert len(selected) == GOF_RECENT_ORIG_LIMIT
    assert selected[0] == _key_for_index(GOF_RECENT_ORIG_LIMIT - 1)
    assert selected[-1] == _key_for_index(0)


def test_select_recent_original_keys_trims_to_limit_when_exceeding_limit() -> None:
    keys = [_key_for_index(idx) for idx in range(GOF_RECENT_ORIG_LIMIT + 1)]

    selected = select_recent_original_keys(keys)

    assert len(selected) == GOF_RECENT_ORIG_LIMIT
    assert selected[0] == _key_for_index(GOF_RECENT_ORIG_LIMIT)
    assert selected[-1] == _key_for_index(1)
    assert _key_for_index(0) not in selected


def test_select_recent_original_samples_break_ties_by_full_path_descending() -> None:
    samples = [
        Sample(
            path="/tmp/a/pony_chart_20240101_000000_000000_abcdefgh.png",
            labels=[1],
        ),
        Sample(
            path="/tmp/z/pony_chart_20240101_000000_000000_abcdefgh.png",
            labels=[1],
        ),
        Sample(
            path="/tmp/m/pony_chart_20231231_235959_000000_abcdefgh.png",
            labels=[1],
        ),
    ]

    selected = select_recent_original_samples(samples)

    assert [sample.path for sample in selected] == [
        "/tmp/z/pony_chart_20240101_000000_000000_abcdefgh.png",
        "/tmp/a/pony_chart_20240101_000000_000000_abcdefgh.png",
        "/tmp/m/pony_chart_20231231_235959_000000_abcdefgh.png",
    ]


def test_prepare_balanced_samples_runs_gof_only_on_recent_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        Sample(
            path=f"/tmp/{_stem_for_index(idx)}.png",
            labels=[1 if idx % 2 == 0 else 2],
        )
        for idx in range(GOF_RECENT_ORIG_LIMIT + 10)
    ]
    seen_counts: list[list[int]] = []

    def fake_gof(
        counts: list[int],
        probs: list[float] | None = None,
        method: str = "pearson_asymptotic",
    ) -> GoFTestResult:
        del method
        seen_counts.append(list(counts))
        n = sum(counts)
        if probs is None:
            probs = [1.0 / len(counts)] * len(counts)
        return GoFTestResult(
            method="pearson_exact",
            statistic=0.0,
            p_value=1.0,
            df=None,
            counts=np.array(counts, dtype=np.int64),
            probs=np.array(probs, dtype=np.float64),
            expected=np.array([n * prob for prob in probs], dtype=np.float64),
            n=n,
            exact=True,
        )

    monkeypatch.setattr(
        "ponychart_classifier.training.sampling.goodness_of_fit_test",
        fake_gof,
    )

    prepare_balanced_samples(samples, np.random.RandomState(0))

    assert len(seen_counts) == 2
    assert sum(seen_counts[0]) == GOF_RECENT_ORIG_LIMIT
    assert sum(seen_counts[1]) == GOF_RECENT_ORIG_LIMIT

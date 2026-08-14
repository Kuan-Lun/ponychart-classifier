import pytest

from ponychart_classifier.training.sampling import Sample
from ponychart_classifier.training.splitting import build_groups


def test_build_groups_uses_full_source_identity() -> None:
    source_a = "pony_chart_20260101_010203_100000_aaaaaaaa"
    source_b = "pony_chart_20260101_010203_200000_bbbbbbbb"
    samples = [
        Sample(path=f"/tmp/{source_a}.png", labels=[1]),
        Sample(path=f"/tmp/{source_a}_crop1.png", labels=[1]),
        Sample(path=f"/tmp/{source_b}.png", labels=[2]),
    ]

    assert build_groups(samples) == {
        source_a: [0, 1],
        source_b: [2],
    }


def test_build_groups_rejects_legacy_name() -> None:
    samples = [Sample(path="/tmp/pony_chart_20260101_010203.png", labels=[1])]

    with pytest.raises(ValueError, match="Unsupported image filename"):
        build_groups(samples)

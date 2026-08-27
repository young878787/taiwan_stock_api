import pytest

from conftest import make_daily_bars

from kstock.pipeline.validation import compare_sources


def test_validation_passes_on_match():
    primary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    secondary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    report = compare_sources(primary, secondary)
    assert report.passed is True
    assert report.checked == 2  # close + volume_shares


def test_validation_catches_mismatch():
    primary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    secondary = make_daily_bars("2330", ["2024-01-02"], [595.0])
    primary = primary.with_columns(primary["close"].alias("close"))
    secondary_adj = secondary.with_columns(secondary["close"] * 1.05)
    report = compare_sources(primary, secondary_adj, tolerance=0.001)
    assert report.passed is False
    issue = report.mismatches[0]
    assert issue["field"] == "close"
    assert issue["primary"] == 590.0
    assert issue["secondary"] == pytest.approx(595.0 * 1.05)


def test_validation_tolerates_small_diff():
    primary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    secondary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    secondary = secondary.with_columns(secondary["close"] * 1.0001)
    report = compare_sources(primary, secondary, tolerance=0.005)
    assert report.passed is True


def test_validation_skips_missing_dates():
    primary = make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 592.0])
    secondary = make_daily_bars("2330", ["2024-01-02"], [590.0])
    report = compare_sources(primary, secondary, fields=("close",))
    assert report.checked == 1  # only compare trades both sources share
from datetime import date

import polars as pl
import pytest

from kstock.quality.checks import QualityIssue, check_daily_quality, check_institutional, clean_daily


def _bars(dates: list[str], prices: list[float], volumes: list[int] | None = None) -> pl.DataFrame:
    n = len(dates)
    vols = volumes or [1_000_000] * n
    return pl.DataFrame(
        {
            "symbol": ["2330"] * n,
            "market": ["TSE"] * n,
            "date": [date(int(d[:4]), int(d[5:7]), int(d[8:10])) for d in dates],
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume_shares": vols,
            "turnover_twd": [1e9] * n,
            "trade_count": [10_000] * n,
            "source": ["test"] * n,
        }
    )


def test_clean_data_passes():
    dates = [f"2024-01-{d:02d}" for d in range(2, 22)]
    prices = [100 + i for i in range(20)]
    report = check_daily_quality(_bars(dates, prices))
    assert report.passed
    assert report.summary == "issues=0"


def test_missing_trading_day_flagged():
    dates = ["2024-01-02", "2024-01-03", "2024-01-06", "2024-01-07"]  # 1/4 缺 2 天
    report = check_daily_quality(_bars(dates, [100, 101, 102, 103]), max_gap_days=2)
    assert not report.passed
    assert any(i.check == "date_gap" for i in report.issues)


def test_nonpositive_price_flagged():
    df = _bars(["2024-01-02", "2024-01-03"], [100.0, 101.0])
    bad = df.with_columns(df["close"].alias("close") * 0)  # close 全部變 0
    report = check_daily_quality(bad)
    issues = [i for i in report.issues if i.check == "ohlc_nonpositive"]
    assert len(issues) == 2


def test_big_move_flagged():
    df = _bars(["2024-01-02", "2024-01-03"], [100.0, 130.0])  # +30%
    report = check_daily_quality(df, max_daily_move=0.15)
    assert any(i.check == "price_jump" for i in report.issues)


def test_volume_spike_flagged():
    vols = [1_000_000] * 20
    vols.append(100_000_000)
    df = _bars([f"2024-03-{d:02d}" for d in range(1, 22)], [100.0] * 21, vols)
    report = check_daily_quality(df, volume_score_factor=20.0, volume_window=20)
    assert any(i.check == "volume_spike" for i in report.issues)


def test_institutional_mismatch_detected():
    base = pl.DataFrame(
        {
            "symbol": ["2330"] * 2,
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "foreign_buy": [1000.0, 1000.0],
            "foreign_sell": [500.0, 600.0],
        }
    )
    other = base.with_columns((pl.col("foreign_buy") * 1.1).alias("foreign_buy"))
    report = check_institutional(base, other, tolerance=0.005)
    assert not report.passed
    assert any(i.check == "institutional_foreign_buy_mismatch" for i in report.issues)


def test_check_report_dicts():
    rep = check_daily_quality(_bars(["2024-01-02"], [100.0]))
    assert rep.passed
    assert rep.dicts() == []


def test_clean_daily_drops_invalid_price_rows():
    df = _bars(["2024-01-02", "2024-01-03", "2024-01-06"], [100.0, 0.0, 101.0])
    cleaned = clean_daily(df)
    assert cleaned.height == 2
    assert cleaned["date"].to_list()[0] == date(2024, 1, 2)
    assert (cleaned["close"] > 0).all()


def test_rounding_noise_not_flagged():
    # high 比 open 低 0.01 元屬四捨五入雜訊，不應列為 inconsistent
    df = _bars(["2024-01-02"], [100.0]).with_columns(
        pl.lit(99.99).alias("high"),
        pl.lit(98.5).alias("low"),
    )
    report = check_daily_quality(df)
    assert not any(i.check == "ohlc_inconsistent" for i in report.issues)


def test_dataclass_misc():
    i = QualityIssue("2330", "ohlc_nonpositive", "2024-01-02", "close=0")
    assert i.as_dict()["symbol"] == "2330"
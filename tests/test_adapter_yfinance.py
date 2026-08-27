from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from kstock.adapters.yfinance import YFinanceAdapter, default_hourly_start_date, yf_symbol


def _hourly_frame() -> pd.DataFrame:
    """模擬 yfinance.Ticker.history(interval='60m') 的回傳格式。"""
    idx = pd.DatetimeIndex(
        [
            "2026-08-25 09:00:00+08:00",
            "2026-08-25 10:00:00+08:00",
            "2026-08-25 11:00:00+08:00",
            "2026-08-25 12:00:00+08:00",
            "2026-08-25 13:00:00+08:00",
            "2026-08-26 09:00:00+08:00",
            "2026-08-26 10:00:00+08:00",
        ],
        name="Datetime",
    )
    return pd.DataFrame(
        {
            "Open": [1080.0, 1082.0, 1084.0, 1081.0, 1080.0, 1090.0, 1091.5],
            "High": [1083.0, 1085.0, 1085.0, 1081.5, 1082.0, 1095.0, 1093.0],
            "Low": [1079.0, 1081.0, 1080.5, 1079.5, 1078.0, 1088.0, 1090.0],
            "Close": [1082.0, 1084.0, 1081.5, 1080.0, 1081.0, 1091.0, 1092.0],
            "Volume": [1234567, 2345678, 1111111, 999999, 888888, 1500000, 1600000],
        },
        index=idx,
    )


@pytest.fixture
def adapter(monkeypatch) -> YFinanceAdapter:
    a = YFinanceAdapter()

    def fake_history(self, ticker, interval, start, end):
        assert ticker.endswith(".TW")
        assert interval in ("60m", "1d")
        return _hourly_frame()

    monkeypatch.setattr(YFinanceAdapter, "_history", fake_history)
    return a


def test_yf_symbol_suffix():
    assert yf_symbol("2330") == "2330.TW"
    assert yf_symbol("2618", market="OTC") == "2618.TWO"
    assert yf_symbol("0050.TW") == "0050.TW"


def test_get_hourly_bars_schema(adapter):
    df = adapter.get_hourly_bars("2330", "2026-06-01", "2026-08-27")
    assert df.height == 7
    # tz 轉成 naive、dtype 為 Datetime
    assert df["date"].dtype == pl.Datetime("us")
    row = df.filter(pl.col("date") == pl.datetime(2026, 8, 25, 9)).row(0, named=True)
    assert row["close"] == 1082.0
    assert row["volume_shares"] == 1_234_567
    assert row["source"] == "yfinance"
    assert set(df.columns) == {
        "symbol", "market", "date", "open", "high", "low", "close",
        "volume_shares", "turnover_twd", "trade_count", "source",
    }


def test_get_hourly_bars_empty(monkeypatch):
    a = YFinanceAdapter()

    def fake_history(self, ticker, interval, start, end):
        return pd.DataFrame()

    monkeypatch.setattr(YFinanceAdapter, "_history", fake_history)
    df = a.get_hourly_bars("9999")
    assert df.height == 0
    assert "volume_shares" in df.columns
    assert "trade_count" in df.columns


def test_default_hourly_start_uses_ceiling():
    s = default_hourly_start_date("2026-08-27")
    assert s < "2026-08-27" and int(s[:4]) in (2024,)


def test_instruments_not_supported(adapter):
    with pytest.raises(NotImplementedError):
        adapter.get_instruments()

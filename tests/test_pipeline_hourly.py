from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from kstock.adapters.yfinance import YFinanceAdapter
from kstock.pipeline.hourly import HourlyUpdatePipeline
from kstock.storage.duckdb import DuckStore
from kstock.storage.parquet import ParquetStore


def _hourly_bars(symbol: str = "2330") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol, symbol],
            "market": ["TSE", "TSE"],
            "date": [
                datetime(2025, 12, 30, 9),
                datetime(2026, 1, 2, 13),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume_shares": [500_000, 600_000],
            "turnover_twd": [None, None],
            "trade_count": [None, None],
            "source": ["yfinance", "yfinance"],
        },
        schema_overrides={"date": pl.Datetime("us")},
    )


@pytest.fixture
def pipeline(monkeypatch, store: ParquetStore) -> HourlyUpdatePipeline:
    adapter = YFinanceAdapter()
    monkeypatch.setattr(adapter, "get_hourly_bars", lambda s, a, b, m="TSE": _hourly_bars(s))
    return HourlyUpdatePipeline(adapter=adapter, store=store)


def test_hourly_pipeline_writes_raw_and_normalized(pipeline: HourlyUpdatePipeline, store: ParquetStore):
    results = pipeline.run(["2330"], start_date="2024-01-01")
    assert len(results) == 1
    assert results[0].row_count == 2

    df = store.read_normalized("hourly", symbols=["2330"])
    assert df.height == 2
    assert sorted(df["symbol"].to_list()) == ["2330", "2330"]

    raw = store.read_raw("yfinance", "hourly")
    assert raw is not None and raw.height == 2


def test_hourly_pipeline_idempotent(pipeline: HourlyUpdatePipeline, store: ParquetStore):
    """重跑同一天資料不應重複（unique(symbol,date)）。"""
    pipeline.run(["2330"], start_date="2024-01-01")
    pipeline.run(["2330"], start_date="2024-01-01")
    assert store.read_normalized("hourly").height == 2


def test_hourly_partitioned_by_year(pipeline: HourlyUpdatePipeline, store: ParquetStore):
    pipeline.run(["2330"])
    years = {p.parent.name for p in store.normalized_files("hourly")}
    assert years == {"year=2025", "year=2026"}


def test_hourly_empty_symbol(pipeline: HourlyUpdatePipeline):
    empty = pl.DataFrame(schema={"symbol": pl.Utf8})
    pipeline.adapter.get_hourly_bars = lambda *a, **k: empty  # type: ignore[assignment]
    r = pipeline.update_symbol("9999")
    assert r.row_count == 0 and r.raw_path == ""


def test_duckdb_register_hourly(pipeline: HourlyUpdatePipeline, duck: DuckStore):
    pipeline.run(["2330"])
    duck.register_views(("daily", "hourly"))
    out = duck.query("SELECT COUNT(*) AS n FROM hourly WHERE symbol='2330'")
    assert out["n"].to_list() == [2]

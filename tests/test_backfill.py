"""backfill（歷史回填）測試：FakeAdapter 驗證分段、冪等與統計。"""

from __future__ import annotations

import datetime
from datetime import date

import polars as pl
import pytest

from conftest import make_daily_bars
from kstock.adapters.base import DataSourceAdapter
from kstock.backfill import (
    DEFAULT_START,
    BackfillStats,
    _clean_error,
    _fetch_with_retry,
    load_delisted_symbols,
    load_universe_symbols,
    run_backfill,
    run_delisted_backfill,
    year_segments,
)
from kstock.storage.parquet import ParquetStore


class FakeBackfillAdapter(DataSourceAdapter):
    name = "fake"

    def __init__(self, bars: pl.DataFrame):
        self._bars = bars
        self.calls: list[tuple[str, str, str]] = []

    def _select(self, symbol, start_date, end_date) -> pl.DataFrame:
        lo = pl.lit(start_date).str.to_date("%Y-%m-%d") if start_date else None
        hi = pl.lit(end_date).str.to_date("%Y-%m-%d") if end_date else None
        df = self._bars.filter(pl.col("symbol") == symbol)
        if lo is not None:
            df = df.filter(pl.col("date") >= lo)
        if hi is not None:
            df = df.filter(pl.col("date") <= hi)
        return df

    def get_daily_bars(self, symbol, start_date=None, end_date=None, market="TSE"):
        self.calls.append(("daily", symbol, str(start_date)))
        return self._select(symbol, start_date, end_date)

    def get_instruments(self):
        raise NotImplementedError

    def get_institutional(self, symbol, start_date=None, end_date=None):
        self.calls.append(("institutional", symbol, str(start_date)))
        return self._select(symbol, start_date, end_date)

    def get_margin(self, symbol, start_date=None, end_date=None):
        self.calls.append(("margin", symbol, str(start_date)))
        return self._select(symbol, start_date, end_date)


def test_year_segments_splits_per_year_and_clamps():
    segs = year_segments("2012-11-01", "2014-02-01")
    assert segs == [
        ("2012-11-01", "2012-12-31"),
        ("2013-01-01", "2013-12-31"),
        ("2014-01-01", "2014-02-01"),
    ]
    # 起點晚於終點 → 空段
    assert year_segments("2015-01-01", "2014-01-01") == []


def test_load_universe_symbols(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("2330\n# 註解\n2454\n\n", encoding="utf-8")
    assert load_universe_symbols(p) == ["2330", "2454"]


def test_run_backfill_writes_and_merges(store: ParquetStore):
    # 既有 2024 資料（模擬現況），回填 2012-2013
    store.write_normalized(
        "daily", make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0])
    )
    bars = make_daily_bars("2330", ["2012-03-05", "2012-03-06", "2013-07-01"])
    adapter = FakeBackfillAdapter(bars)

    stats = run_backfill(
        symbols=["2330"],
        start="2012-01-01",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=adapter,
        store=store,
    )

    # end 預設今日 → 段數 2012~2026；2024 已有資料被跳過，
    # fake 只有 2012/2013 資料，其餘 12 個年段回空列屬正常
    assert len(adapter.calls) == 14
    assert adapter.calls[0] == ("daily", "2330", "2012-01-01")
    assert stats.rows_written == 3
    assert stats.empty_responses == 12

    merged = store.read_normalized("daily", symbols=["2330"]).sort("date")
    assert merged["date"].to_list()[0] == date(2012, 3, 5)
    assert merged.height == 5  # 3（新） + 2（舊 2024）


def test_run_backfill_resume_skips_existing(store: ParquetStore):
    bars = make_daily_bars("2330", ["2012-03-05"], [570.0])
    adapter = FakeBackfillAdapter(bars)
    run_backfill(
        symbols=["2330"],
        start="2012-01-01",
        end="2012-12-31",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=adapter,
        store=store,
    )
    assert len(adapter.calls) == 1
    # 重跑：該 (symbol, 年) 已有資料 → 不再發請求
    stats = run_backfill(
        symbols=["2330"],
        start="2012-01-01",
        end="2012-12-31",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=adapter,
        store=store,
    )
    assert len(adapter.calls) == 1  # 跳過既有段
    assert stats.rows_written == 0
    assert store.read_normalized("daily", symbols=["2330"]).height == 1


def test_run_backfill_idempotent_rerun(store: ParquetStore):
    bars = make_daily_bars("2330", ["2012-03-05"], [570.0])
    adapter = FakeBackfillAdapter(bars)
    for _ in range(2):
        run_backfill(
            symbols=["2330"],
            start="2012-01-01",
            end="2012-12-31",
            tables=("daily",),
            sleep_seconds=0.0,
            adapter=adapter,
            store=store,
        )
    merged = store.read_normalized("daily", symbols=["2330"])
    assert merged.height == 1  # unique(symbol, date) 不重複


def test_run_backfill_failure_isolated(store: ParquetStore):
    bars = make_daily_bars("good", ["2012-03-05"], [570.0])

    class FlakyAdapter(FakeBackfillAdapter):
        def get_daily_bars(self, symbol, start_date=None, end_date=None, market="TSE"):
            self.calls.append(("daily", symbol, str(start_date)))
            if symbol == "bad":
                raise RuntimeError("模擬 API 失敗")
            return bars

    adapter = FlakyAdapter(bars)
    stats = run_backfill(
        symbols=["good", "bad"],
        start="2012-01-01",
        end="2012-12-31",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=adapter,
        store=store,
    )
    assert stats.failures == [("daily", "bad", "2012")]
    assert stats.rows_written == 1
    assert store.read_normalized("daily", symbols=["good"]).height == 1


def test_run_backfill_dry_run(store: ParquetStore):
    adapter = FakeBackfillAdapter(make_daily_bars("2330"))
    stats = run_backfill(
        symbols=["2330"],
        start=DEFAULT_START,
        tables=("daily",),
        sleep_seconds=0.0,
        dry_run=True,
        adapter=adapter,
        store=store,
    )
    assert isinstance(stats, BackfillStats)
    assert stats.requests == 0
    assert adapter.calls == []
    assert store.normalized_files("daily") == []


def test_run_backfill_aborts_on_quota(store: ParquetStore, monkeypatch):
    import httpx

    bars = make_daily_bars("2330", ["2012-03-05"], [570.0])

    def fake_fetch(adapter, table, symbol, start_date, end_date, **kwargs):
        raise httpx.HTTPStatusError(
            f"Client error '402' for url 'https://api.finmindtrade.com/api/v4/data?token=SECRET'",
            request=None,  # type: ignore[arg-type]
            response=httpx.Response(402, request=None),
        )

    monkeypatch.setattr("kstock.backfill._fetch_with_retry", fake_fetch)
    stats = run_backfill(
        symbols=["2330"],
        start="2012-01-01",
        end="2012-12-31",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=FakeBackfillAdapter(bars),
        store=store,
    )
    assert stats.failures == [("daily", "2330", "2012")]
    # 中止：其餘段（2021~2026）不會再被嘗試
    assert stats.requests == 1


def test_clean_error_masks_token():
    import httpx

    e = httpx.HTTPStatusError(
        "Client error '402' for url 'https://api.finmindtrade.com/api/v4/data?dataset=x&token=SECRET'",
        request=None,  # type: ignore[arg-type]
        response=httpx.Response(402, request=None),
    )
    masked = _clean_error(e)
    assert "SECRET" not in masked
    assert "token=***" in masked


# ---- 下市回填模式（WP4） ----

SAMPLE_DELISTED_CSV = (
    "symbol,name,delist_date,source\n"
    "2841,台開,2022-08-04,twse_terminated\n"
    "1001,早退,2005-01-01,twse_terminated\n"
    "9999,無日期,,mi_index_sampling\n"
)


def test_load_delisted_symbols_filters_and_parses(tmp_path):
    p = tmp_path / "delisted.csv"
    p.write_text(SAMPLE_DELISTED_CSV, encoding="utf-8")
    rows = load_delisted_symbols(p, "2008-01-01")
    # 早退（2005 < 2008 起點）被剔除；無日期保留（None）
    assert rows == [("2841", "2022-08-04"), ("9999", None)]


def test_run_delisted_backfill_clips_at_delist_date(store: ParquetStore, tmp_path):
    # 2841 台開：2008-2022 歷史 + 下市後（2022-09，重用代碼/新公司）價格 → 後者不得入庫
    bars = make_daily_bars(
        "2841",
        ["2012-03-05", "2012-03-06", "2022-08-04", "2022-09-05"],
        [10.0, 10.1, 9.9, 88.8],
    )
    adapter = FakeBackfillAdapter(bars)
    csv_path = tmp_path / "delisted.csv"
    csv_path.write_text(SAMPLE_DELISTED_CSV, encoding="utf-8")
    stats = run_delisted_backfill(
        delisted_csv=csv_path,
        start="2008-01-01",
        tables=("daily",),
        sleep_seconds=0.0,
        adapter=adapter,
        store=store,
    )
    written = store.read_normalized("daily", symbols=["2841"]).sort("date")
    # 9999（無日期）也會被回補到今天——fixture bars 只含 2841，其段回空
    assert written["date"].max() <= datetime.date(2022, 8, 4)  # 裁切生效：2022-09-05 不入庫
    assert written["date"].min() == datetime.date(2012, 3, 5)
    assert written["close"].to_list() == [10.0, 10.1, 9.9]
    assert stats.failures == []
    # 早退（2005 下市）不發請求
    called = {s for _, s, _ in adapter.calls}
    assert "1001" not in called


def test_run_delisted_backfill_dry_run(tmp_path, capsys):
    p = tmp_path / "delisted.csv"
    p.write_text(SAMPLE_DELISTED_CSV, encoding="utf-8")
    stats = run_delisted_backfill(delisted_csv=p, start="2008-01-01", dry_run=True)
    out = capsys.readouterr().out
    assert "[dry-run] 下市回填 2 檔" in out
    assert stats.rows_written == 0


def test_main_delisted_csv_mode(tmp_path, monkeypatch, capsys):
    from kstock import backfill as bf

    p = tmp_path / "delisted.csv"
    p.write_text(SAMPLE_DELISTED_CSV, encoding="utf-8")

    class FakeAdapter(FakeBackfillAdapter):
        def __init__(self):
            super().__init__(
                pl.DataFrame(
                    schema={
                        "symbol": pl.Utf8, "market": pl.Utf8, "date": pl.Date,
                        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                        "close": pl.Float64, "volume_shares": pl.Int64,
                        "turnover_twd": pl.Float64, "trade_count": pl.Int64, "source": pl.Utf8,
                    }
                )
            )

    fake = FakeAdapter()
    monkeypatch.setattr(bf, "FinMindAdapter", lambda: fake)
    monkeypatch.setattr(bf, "ParquetStore", lambda: ParquetStore(settings_=None) if False else ParquetStore(root=tmp_path / "store"))
    rc = bf.main(
        ["--delisted-csv", str(p), "--start", "2008-01-01", "--sleep", "0"],
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "下市回填 2 檔" in out

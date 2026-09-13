import datetime

import polars as pl

from conftest import make_daily_bars

from kstock.storage.parquet import ParquetStore


def test_write_and_read_partitioned(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0]))
    store.write_normalized("daily", make_daily_bars("2330", ["2025-02-04"], [600.0]))

    parts = store.normalized_files("daily")
    assert len(parts) == 2
    for p in parts:
        assert "year=2024" in str(p) or "year=2025" in str(p)

    read = store.read_normalized("daily")
    assert read.height == 3


def test_partition_dirs_use_hive_layout(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    folders = [p.parent.name for p in store.normalized_files("daily")]
    assert folders == ["year=2024"]


def test_read_normalized_symbol_and_date_filters(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-06-01", "2024-07-02"], [100.0, 101.0]))
    store.write_normalized("daily", make_daily_bars("0050", ["2024-06-15"], [30.0]))

    only_2330 = store.read_normalized("daily", symbols=["2330"])
    assert only_2330.height == 2

    window = store.read_normalized("daily", start_date="2024-06-15", end_date="2024-06-30")
    assert window.height == 1
    assert window["symbol"].to_list() == ["0050"]


def test_write_raw_appends_and_dedups(store: ParquetStore):
    store.write_raw("finmind", "daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    store.write_raw(
        "finmind",
        "daily",
        make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [591.0, 592.0]),
    )
    raw = store.read_raw("finmind", "daily")
    assert raw.height == 2


def test_read_empty_table_returns_typed(store: ParquetStore):
    df = store.read_normalized("daily")
    assert df.height == 0
    assert "volume_shares" in df.columns


def _instrument_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["2330", "0050"],
            "name": ["台積電", "元大台灣50"],
            "market": ["TSE", "TSE"],
            "industry": ["半導體", "ETF"],
            "list_date": [datetime.date(1994, 9, 5), datetime.date(2003, 6, 30)],
            "delist_date": [None, None],
            "status": ["active", "active"],
        }
    )


def test_write_reference_table_upsert_idempotent(store: ParquetStore):
    df = _instrument_df()
    store.write_reference_table("instrument", df, subset=["symbol"])

    target = store.normalized_dir / "instrument" / "data.parquet"
    assert target.exists()
    # 單檔版式：無 year= 目錄
    assert not list((store.normalized_dir / "instrument").glob("year=*"))
    assert pl.read_parquet(target).height == 2

    # 同 symbol 新值 → keep="last" 冪等：height 不變、值為新值
    updated = df.with_columns(
        pl.when(pl.col("symbol") == "2330")
        .then(pl.lit("台積電（更名）"))
        .otherwise(pl.col("name"))
        .alias("name")
    )
    store.write_reference_table("instrument", updated, subset=["symbol"])
    merged = pl.read_parquet(target)
    assert merged.height == 2
    assert merged.filter(pl.col("symbol") == "2330")["name"][0] == "台積電（更名）"


def test_write_reference_table_empty_noop(store: ParquetStore):
    store.write_reference_table(
        "instrument", pl.DataFrame(schema={"symbol": pl.Utf8}), subset=["symbol"]
    )
    assert not (store.normalized_dir / "instrument").exists()
from conftest import make_daily_bars

import polars as pl

from kstock.storage.parquet import ParquetStore


def test_query_daily_via_duckdb(store: ParquetStore, duck):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0]))
    store.write_normalized("daily", make_daily_bars("0050", ["2024-01-04"], [30.0]))
    duck.register_views(("daily",))

    ordered = duck.query("SELECT symbol, close FROM daily ORDER BY date DESC")
    assert ordered["close"].to_list() == [30.0, 598.0, 590.0]

    agg = duck.query(
        "SELECT symbol, count(*) AS n, avg(close) AS m FROM daily GROUP BY symbol ORDER BY symbol"
    )
    assert agg.height == 2
    assert agg.row(0, named=True)["symbol"] == "0050"
    assert agg.row(0, named=True)["n"] == 1


def test_query_parquet_directly(store: ParquetStore, duck):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    glob_path = str(store.normalized_dir / "daily" / "year=*" / "*.parquet")
    df = duck.query_parquet(glob_path, "SELECT * FROM <TABLE> WHERE symbol = '2330'")
    assert df.height == 1
    assert df["close"].to_list() == [590.0]


def test_view_missing_table_creates_placeholder(duck):
    duck.register_view("does_not_exist")
    duck.register_views(("does_not_exist_too",))
    assert duck.query("SELECT * FROM does_not_exist LIMIT 1").height == 0


def test_register_view_reference_table_layout(store: ParquetStore, duck):
    """單檔參考表版式（無 year= 目錄）register_view 也能查到。"""
    df = pl.DataFrame(
        {
            "symbol": ["2330", "0050"],
            "name": ["台積電", "元大台灣50"],
            "market": ["TSE", "TSE"],
            "industry": ["半導體", "ETF"],
            "list_date": [None, None],
            "delist_date": [None, None],
            "status": ["active", "active"],
        },
        schema_overrides={"list_date": pl.Date, "delist_date": pl.Date},
    )
    store.write_reference_table("instrument", df, subset=["symbol"])
    duck.register_view("instrument")

    out = duck.query("SELECT count(*) AS n FROM instrument")
    assert out["n"][0] == 2
    markets = duck.query("SELECT market FROM instrument ORDER BY symbol")
    assert markets["market"].to_list() == ["TSE", "TSE"]


def test_table_exists(store: ParquetStore, duck):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    duck.register_view("daily")
    assert duck.table_exists("daily") is True
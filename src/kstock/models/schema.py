"""標準 Normalized Schema（資料進 Parquet 前的統一格式）。"""

from __future__ import annotations

import polars as pl

DAILY_BAR_COLUMNS = [
    "symbol",
    "market",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume_shares",
    "turnover_twd",
    "trade_count",
    "source",
]

MINUTE_BAR_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume_shares",
    "turnover_twd",
    "source",
]

HOURLY_BAR_COLUMNS = [
    "symbol",
    "market",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume_shares",
    "turnover_twd",
    "trade_count",
    "source",
]

TICK_COLUMNS = [
    "symbol",
    "timestamp",
    "price",
    "size_shares",
    "tick_type",
    "bid",
    "ask",
    "source",
]

INSTRUMENT_COLUMNS = [
    "symbol",
    "name",
    "market",
    "industry",
    "list_date",
    "delist_date",
    "status",
    "snapshot_date",
]

INSTITUTIONAL_COLUMNS = [
    "symbol",
    "date",
    "foreign_buy",
    "foreign_sell",
    "investment_trust_buy",
    "investment_trust_sell",
    "dealer_buy",
    "dealer_sell",
    "source",
]

MARGIN_COLUMNS = [
    "symbol",
    "date",
    "margin_balance",
    "margin_buy",
    "margin_sell",
    "short_balance",
    "short_sell",
    "short_cover",
    "source",
]

TABLES = {
    "daily": DAILY_BAR_COLUMNS,
    "hourly": HOURLY_BAR_COLUMNS,
    "minute": MINUTE_BAR_COLUMNS,
    "tick": TICK_COLUMNS,
    "instrument": INSTRUMENT_COLUMNS,
    "institutional": INSTITUTIONAL_COLUMNS,
    "margin": MARGIN_COLUMNS,
}


def table_columns(table: str) -> list[str]:
    if table not in TABLES:
        raise KeyError(f"unknown table: {table}")
    return list(TABLES[table])


TABLE_DTYPES: dict[str, dict[str, pl.DataType]] = {
    "daily": {
        "symbol": pl.Utf8,
        "market": pl.Utf8,
        "date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume_shares": pl.Int64,
        "turnover_twd": pl.Float64,
        "trade_count": pl.Int64,
        "source": pl.Utf8,
    },
    "minute": {
        "symbol": pl.Utf8,
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume_shares": pl.Int64,
        "turnover_twd": pl.Float64,
        "source": pl.Utf8,
    },
    "hourly": {
        "symbol": pl.Utf8,
        "market": pl.Utf8,
        # 小時K 以「台北時間、無時區」的小時開始時間標記（例：2026-08-26 09:00）
        "date": pl.Datetime("us"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume_shares": pl.Int64,
        "turnover_twd": pl.Float64,
        "trade_count": pl.Int64,
        "source": pl.Utf8,
    },
    "tick": {
        "symbol": pl.Utf8,
        "timestamp": pl.Datetime,
        "price": pl.Float64,
        "size_shares": pl.Int64,
        "tick_type": pl.Utf8,
        "bid": pl.Float64,
        "ask": pl.Float64,
        "source": pl.Utf8,
    },
    "instrument": {
        "symbol": pl.Utf8,
        "name": pl.Utf8,
        "market": pl.Utf8,
        "industry": pl.Utf8,
        "list_date": pl.Date,
        "delist_date": pl.Date,
        "status": pl.Utf8,
        # TaiwanStockInfo 的「資料異動快照日」：活躍股日日更新（≈今天），
        # stale 舊列混有已下市/異動歷史 → 判定現活躍需以 max(snapshot_date) 近期為準
        "snapshot_date": pl.Date,
    },
    "institutional": {
        "symbol": pl.Utf8,
        "date": pl.Date,
        "foreign_buy": pl.Float64,
        "foreign_sell": pl.Float64,
        "investment_trust_buy": pl.Float64,
        "investment_trust_sell": pl.Float64,
        "dealer_buy": pl.Float64,
        "dealer_sell": pl.Float64,
        "source": pl.Utf8,
    },
    "margin": {
        "symbol": pl.Utf8,
        "date": pl.Date,
        "margin_balance": pl.Float64,
        "margin_buy": pl.Float64,
        "margin_sell": pl.Float64,
        "short_balance": pl.Float64,
        "short_sell": pl.Float64,
        "short_cover": pl.Float64,
        "source": pl.Utf8,
    },
}


def empty_dataframe(table: str) -> pl.DataFrame:
    """回傳符合該表 typed schema 的空 DataFrame。"""
    return pl.DataFrame(schema=TABLE_DTYPES[table])


def validate_columns(df: pl.DataFrame, table: str) -> list[str]:
    expected = TABLES[table]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{table} expected missing columns: {missing}")
    return expected
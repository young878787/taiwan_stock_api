"""ML 特徵工程管線：daily × 籌碼 → 單一特徵表。

所有欄位僅使用「當日（含）以前」資訊；label_ret_1f 為隔日報酬（訓練目標）。
"""

from __future__ import annotations

import polars as pl

from kstock.features.technical import add_returns, add_rsi, add_sma, add_volume_ratio

BASE_DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume_shares", "turnover_twd"]


def _daily_price_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.select(BASE_DAILY_COLS).sort(["symbol", "date"])
    out = add_returns(out, n=1)          # return_1d
    out = add_returns(out, n=5)          # return_5d
    out = add_returns(out, n=20)         # return_20d
    out = add_sma(out, window=5)         # sma_5
    out = add_sma(out, window=20)        # sma_20
    out = add_volume_ratio(out, window=20)   # volume_ratio_20
    out = add_rsi(out, n=14)             # rsi_14

    out = out.with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).over("symbol").alias("label_ret_1f"),
        pl.col("return_1d").rolling_std(window_size=20).over("symbol").alias("volatility_20d"),
        pl.col("turnover_twd").rolling_mean(window_size=20).over("symbol").alias("turnover_mean_20"),
    ).with_columns(
        (pl.col("sma_5") / pl.col("sma_20")).alias("ma5_ma20_ratio"),
        (pl.col("close") / pl.col("sma_20") - 1).alias("close_vs_ma20"),
    )
    return out.drop(["sma_5", "sma_20"])


def _attach_institutional(df: pl.DataFrame, inst: pl.DataFrame | None) -> pl.DataFrame:
    if inst is None or inst.is_empty():
        for name in ("foreign_net_lots", "trust_net_lots", "tii_net_lots", "foreign_net_ratio"):
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        return df
    chips = (
        inst.sort(["symbol", "date"])
        .with_columns(
            ((pl.col("foreign_buy") - pl.col("foreign_sell")) / 1000.0).alias("foreign_net_lots"),
            ((pl.col("investment_trust_buy") - pl.col("investment_trust_sell")) / 1000.0).alias("trust_net_lots"),
            (
                (
                    pl.col("foreign_buy")
                    + pl.col("investment_trust_buy")
                    + pl.col("dealer_buy")
                    - pl.col("foreign_sell")
                    - pl.col("investment_trust_sell")
                    - pl.col("dealer_sell")
                )
                / 1000.0
            ).alias("tii_net_lots"),
        )
        .select(["symbol", "date", "foreign_net_lots", "trust_net_lots", "tii_net_lots"])
    )
    joined = df.join(chips, on=["symbol", "date"], how="left").with_columns(
        (pl.col("foreign_net_lots") * 1000 / pl.col("volume_shares")).alias("foreign_net_ratio")
    )
    return joined


def _attach_margin(df: pl.DataFrame, margin: pl.DataFrame | None) -> pl.DataFrame:
    if margin is None or margin.is_empty():
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("margin_change_pct"),
                pl.lit(None, dtype=pl.Float64).alias("short_change_pct"),
                pl.lit(None, dtype=pl.Float64).alias("margin_short_ratio"),
            ]
        )
    mar = (
        margin.sort(["symbol", "date"])
        .select(["symbol", "date", "margin_balance", "short_balance"])
        .with_columns(
            pl.col("margin_balance").pct_change(1).over("symbol").alias("margin_change_pct"),
            pl.col("short_balance").pct_change(1).over("symbol").alias("short_change_pct"),
            (pl.col("short_balance") / pl.col("margin_balance")).alias("margin_short_ratio"),
        )
        .drop(["margin_balance", "short_balance"])
    )
    return df.join(mar, on=["symbol", "date"], how="left")


FEATURE_ORDER = [
    "symbol",
    "date",
    "close",
    "return_1d",
    "return_5d",
    "return_20d",
    "rsi_14",
    "ma5_ma20_ratio",
    "close_vs_ma20",
    "volatility_20d",
    "volume_ratio_20",
    "turnover_mean_20",
    "foreign_net_lots",
    "trust_net_lots",
    "tii_net_lots",
    "foreign_net_ratio",
    "margin_change_pct",
    "short_change_pct",
    "margin_short_ratio",
    "label_ret_1f",
]


def build_ml_features(
    daily: pl.DataFrame,
    institutional: pl.DataFrame | None = None,
    margin: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """組合全部特徵；回傳依 (symbol, date) 排序的特徵表。"""
    df = _daily_price_features(daily)
    df = _attach_institutional(df, institutional)
    df = _attach_margin(df, margin)
    return df.select(FEATURE_ORDER).sort(["symbol", "date"])
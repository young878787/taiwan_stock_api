"""技術指標與策略計算（FEATURE 層）。"""

from __future__ import annotations

import polars as pl


def add_returns(df: pl.DataFrame, price_col: str = "close", n: int = 1, group_col: str = "symbol") -> pl.DataFrame:
    """依標的計算 N 日報酬率，新增欄位 return_<n>d（% / 100）。"""
    return df.sort(group_col, "date").with_columns(
        pl.col(price_col)
        .pct_change(n=n)
        .over(group_col)
        .alias(f"return_{n}d")
    )


def add_sma(df: pl.DataFrame, price_col: str = "close", window: int = 5, group_col: str = "symbol") -> pl.DataFrame:
    """新增簡單移動平均欄位 sma_<window>。"""
    return df.sort(group_col, "date").with_columns(
        pl.col(price_col)
        .rolling_mean(window_size=window)
        .over(group_col)
        .alias(f"sma_{window}")
    )


def _rsi_from_close(close: pl.Series, n: int = 14) -> pl.Series:
    """Wilder 平滑 RSI 序列（ewm 近似）。"""
    diff = close.diff()
    gains = diff.clip(lower_bound=0.0)
    losses = (-diff).clip(lower_bound=0.0)
    avg_gain = gains.ewm_mean(alpha=1.0 / n, adjust=False)
    avg_loss = losses.ewm_mean(alpha=1.0 / n, adjust=False)
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fill_nan(50.0).fill_null(50.0).alias("rsi")


def add_rsi(df: pl.DataFrame, price_col: str = "close", n: int = 14, group_col: str = "symbol") -> pl.DataFrame:
    """新增 RSI 欄位 rsi_<n>（每標的分開計算）。"""
    frames = []
    name = f"rsi_{n}"
    for sub in df.sort(group_col, "date").partition_by(group_col, maintain_order=True):
        frames.append(sub.with_columns(_rsi_from_close(sub[price_col], n).rename(name)))
    return pl.concat(frames)


def add_volume_ratio(df: pl.DataFrame, volume_col: str = "volume_shares", window: int = 5, group_col: str = "symbol") -> pl.DataFrame:
    """量比 = 當日成交量 / 過去 window 日平均成交量。"""
    return df.sort(group_col, "date").with_columns(
        (pl.col(volume_col) / pl.col(volume_col).rolling_mean(window_size=window).over(group_col)).alias(
            f"volume_ratio_{window}"
        )
    )
"""資料品質檢查（缺日、異常價格/量、法人對照）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from kstock.storage.parquet import ParquetStore

INSTITUTIONAL_NET_FIELDS = ("foreign_buy", "foreign_sell", "investment_trust_buy", "investment_trust_sell", "dealer_buy", "dealer_sell")


@dataclass
class QualityIssue:
    symbol: str
    check: str
    date: str | None
    message: str

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "check": self.check, "date": self.date, "message": self.message}


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def summary(self) -> str:
        return f"issues={len(self.issues)}"

    def dicts(self) -> list[dict]:
        return [i.as_dict() for i in self.issues]


def _as_iso(value) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def check_daily_quality(
    df: pl.DataFrame,
    max_daily_move: float = 0.15,
    max_gap_days: int = 15,
    volume_score_factor: float = 20.0,
    volume_window: int = 20,
) -> QualityReport:
    """對每日報表做品質檢查（逐標的）。

    - price：O/H/L/C 為正且有限、OHLC 一致性、單日漲跌幅度上限
    - gap：交易日間隔超過 max_gap_days 天（預設 15，農曆年休市約 11~13 天屬正常）
    - volume：大於 20 日 median 的 volume_score_factor 倍
    """
    report = QualityReport()
    if df.is_empty():
        return report
    for sub in df.partition_by(["symbol"], maintain_order=True):
        symbol = str(sub["symbol"][0])
        ordered = sub.sort("date")
        report.issues.extend(_check_ohlc(ordered, symbol))
        report.issues.extend(_check_dates(ordered, symbol, max_gap_days))
        report.issues.extend(_check_returns(ordered, symbol, max_daily_move))
        report.issues.extend(_check_volume(ordered, symbol, volume_score_factor, volume_window))
    return report


def clean_daily(df: pl.DataFrame) -> pl.DataFrame:
    """移除無效價格列（價格 ≤ 0 或空值），供 ML 訓練前清洗；保留合法資料不動。"""
    if df.is_empty():
        return df
    valid = (
        pl.col("open").is_not_null() & (pl.col("open") > 0)
        & pl.col("high").is_not_null() & (pl.col("high") > 0)
        & pl.col("low").is_not_null() & (pl.col("low") > 0)
        & pl.col("close").is_not_null() & (pl.col("close") > 0)
    )
    return df.filter(valid)


def _check_ohlc(sub: pl.DataFrame, symbol: str, tolerance_abs: float = 0.06, tolerance_rel: float = 0.002) -> list[QualityIssue]:
    """OHLC 一致性檢查帶容差：小幅四捨五入雜訊（≤ max(0.06元, 0.2%×價格)）不算違規。"""
    issues: list[QualityIssue] = []
    for col in ("open", "high", "low", "close"):
        bad = sub.filter(pl.col(col).is_null() | (pl.col(col) <= 0))
        for r in bad.iter_rows(named=True):
            issues.append(QualityIssue(symbol=symbol, check="ohlc_nonpositive", date=_as_iso(r["date"]), message=f"{col}={r[col]}"))
    tol = pl.max_horizontal(
        pl.lit(tolerance_abs),
        (pl.max_horizontal("open", "close") * tolerance_rel),
    )
    inconsistent = sub.filter(
        ((pl.max_horizontal("open", "close") - pl.col("high")) > tol)
        | ((pl.col("low") - pl.min_horizontal("open", "close")) > tol)
    )
    for r in inconsistent.iter_rows(named=True):
        issues.append(QualityIssue(symbol=symbol, check="ohlc_inconsistent", date=_as_iso(r["date"]), message=f"high={r['high']} low={r['low']} o={r['open']} c={r['close']}"))
    return issues


def _check_dates(sub: pl.DataFrame, symbol: str, max_gap_days: int) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    dates = sub["date"].to_list()
    for prev, nxt in zip(dates, dates[1:]):
        gap = (nxt - prev).days
        if gap > max_gap_days:
            issues.append(QualityIssue(symbol=symbol, check="date_gap", date=_as_iso(nxt), message=f"gap={gap}d from {prev}"))
    return issues


def _check_returns(sub: pl.DataFrame, symbol: str, max_move: float) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    prev = sub["close"].shift(1)
    moved = sub.filter((sub["close"] / prev - 1).abs() > max_move)
    for r in moved.iter_rows(named=True):
        issues.append(QualityIssue(symbol=symbol, check="price_jump", date=_as_iso(r["date"]), message=f"close={r['close']}"))
    return issues


def _check_volume(
    sub: pl.DataFrame, symbol: str, factor: float, window: int
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n = max(1, min(window, sub.height))
    checked = sub.with_columns(pl.col("volume_shares").rolling_median(window_size=n).alias("med"))
    spike = checked.filter(
        (pl.col("med").is_not_null()) & (pl.col("med") > 0) & (pl.col("volume_shares") > pl.col("med") * factor)
    )
    for r in spike.iter_rows(named=True):
        issues.append(
            QualityIssue(symbol=symbol, check="volume_spike", date=_as_iso(r["date"]), message=f"vol={r['volume_shares']} median={r['med']}")
        )
    return issues


def check_institutional(
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    tolerance: float = 0.005,
) -> QualityReport:
    """比較兩法人資料來源（primary 為基準），同交易日同欄位以相對誤差比較。"""
    report = QualityReport()
    if primary.is_empty() or secondary.is_empty():
        return report
    for field in INSTITUTIONAL_NET_FIELDS:
        if field not in primary.columns or field not in secondary.columns:
            continue
        joined = primary.select(["symbol", "date", field]).rename({field: "a"})
        comp = secondary.select(["symbol", "date", field]).rename({field: "b"})
        rows = joined.join(comp, on=["symbol", "date"], how="inner")
        if rows.height == 0:
            continue
        for r in rows.iter_rows(named=True):
            a, b = r["a"], r["b"]
            if a is None or b is None or a == b:
                continue
            base = abs(a)
            err = abs(a - b)
            if (err / base if base > 0 else err) > tolerance:
                report.issues.append(
                    QualityIssue(symbol=r["symbol"], check=f"institutional_{field}_mismatch", date=_as_iso(r["date"]), message=f"{a} != {b}")
                )
    return report


def audit_daily_report(store: ParquetStore) -> QualityReport:
    """對 normalized/daily 全部資料跑完整品質檢查。"""
    df = store.read_normalized("daily")
    return check_daily_quality(df)
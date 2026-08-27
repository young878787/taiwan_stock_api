"""Yahoo Finance Adapter（盤中小時K線）。

用途：FinMind 免費等級無法取得台股盤中資料，改用 Yahoo Finance
補小時K線（interval=60m）。Yahoo 對 60m 資料的歷史回溯上限約 **730 天**。

注意：
- 台股代號需掛市場後綴：上市 `{symbol}.TW`、上櫃 `{symbol}.TWO`
- 成交量單位為「股」（與 daily volume_shares 同單位）
- 時間戳為台北時間的小時開始（例：09:00 / 10:00 / ... / 13:00），統一轉成
  naive Asia/Taipei Datetime 儲存
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from kstock.adapters.base import DataSourceAdapter
from kstock.models.schema import empty_dataframe

_BAR_TZ = "Asia/Taipei"
_HOURLY_MAX_DAYS = 700  # Yahoo 上限 730 天，留緩衝


class YFinanceError(Exception):
    pass


def yf_symbol(symbol: str, market: str = "TSE") -> str:
    """標準 symbol → Yahoo 代號（TSE→.TW / OTC→.TWO）。"""
    if "." in symbol:
        return symbol
    suffix = "TWO" if market.upper() == "OTC" else "TW"
    return f"{symbol}.{suffix}"


def default_hourly_start_date(end_date: str | None = None) -> str:
    """60m 歷史上限對應的預設起始日（今天往前約 700 天）。"""
    end = (
        datetime.strptime(end_date, "%Y-%m-%d")
        if end_date
        else datetime.now(timezone.utc) + timedelta(days=1)
    )
    return (end - timedelta(days=_HOURLY_MAX_DAYS)).strftime("%Y-%m-%d")


def _naive_local(ts: Any) -> datetime | None:
    """pandas index → naive Asia/Taipei datetime。"""
    ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def _history_df_to_hourly(frame: Any, *, symbol: str, market: str, source: str) -> pl.DataFrame:
    """Yahoo history DataFrame → hourly 標準 schema。"""
    records = []
    for idx, row in frame.iterrows():
        ts = _naive_local(idx)
        if ts is None:
            continue
        open_ = row.get("Open")
        high = row.get("High")
        low = row.get("Low")
        close_ = row.get("Close")
        volume = row.get("Volume")
        if any(v is None or v != v or float(v) == 0 for v in (open_, close_, high, low)):
            continue
        vol = 0 if volume is None or volume != volume else int(round(float(volume)))
        if vol == 0 and float(close_) == 0:
            continue
        records.append(
            {
                "symbol": symbol,
                "market": market,
                "date": ts,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close_),
                "volume_shares": vol,
                "turnover_twd": None,
                "trade_count": None,
                "source": source,
            }
        )
    if not records:
        return empty_dataframe("hourly")
    return pl.from_dicts(records).sort("date").select(empty_dataframe("hourly").columns)


class YFinanceAdapter(DataSourceAdapter):
    name = "yfinance"

    def __init__(self, session: Any = None) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - 環境未安裝 yfinance 才會發生
            raise YFinanceError("請先安裝 yfinance：uv add yfinance") from exc
        self._yf = yf
        self._session = session

    # ------------------------------------------------------------------
    # 內部工具（測試可 monkeypatch）
    # ------------------------------------------------------------------
    def _history(self, ticker: str, interval: str, start: str | None, end: str | None):
        t = self._yf.Ticker(ticker)
        return t.history(interval=interval, start=start, end=end, auto_adjust=False)

    # ------------------------------------------------------------------
    # DataSourceAdapter 介面
    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
    ) -> pl.DataFrame:
        """輔助功能：Yahoo 日K（僅供 FinMind 日K 的交叉驗證用）。"""
        frame = self._history(yf_symbol(symbol, market), "1d", start_date, end_date)
        if frame is None or len(frame) == 0:
            return empty_dataframe("daily")
        hourly_schema_cols = ["symbol", "market", "date", "open", "high", "low", "close",
                              "volume_shares", "turnover_twd", "trade_count", "source"]
        records = []
        for idx, row in frame.iterrows():
            d = _naive_local(idx)
            if d is None:
                continue
            close_ = row.get("Close")
            if close_ is None or close_ != close_:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "date": d.date(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(close_),
                    "volume_shares": int(round(float(row["Volume"]))) if row.get("Volume") is not None else None,
                    "turnover_twd": None,
                    "trade_count": None,
                    "source": self.name,
                }
            )
        if not records:
            return empty_dataframe("daily")
        return pl.from_dicts(records).sort("date").select(hourly_schema_cols)

    def get_hourly_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
    ) -> pl.DataFrame:
        """取得小時K（interval=60m），歷史上限約 730 天。"""
        if start_date is None:
            start_date = default_hourly_start_date(end_date)
        frame = self._history(yf_symbol(symbol, market), "60m", start_date, end_date)
        if frame is None or len(frame) == 0:
            return empty_dataframe("hourly")
        return _history_df_to_hourly(frame, symbol=symbol, market=market, source=self.name)

    def get_instruments(self) -> pl.DataFrame:
        raise NotImplementedError(f"{self.name} 不提供 instrument 清單，請改用 FinMindAdapter")

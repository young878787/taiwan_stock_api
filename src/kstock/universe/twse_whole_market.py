"""選股宇宙（Universe）：台股全市場報表解析與流動性排名。

用途：建立「台灣前 N 大流動性股票」清單給 ML 選股使用。
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta

import httpx
import polars as pl

from kstock.config.settings import settings

SYMBOL_RE = re.compile(r"^\d{4}$")


def _num(value) -> float | None:
    text = str(value).replace(",", "").strip()
    if text in ("---", "", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_table(payload: dict) -> dict | None:
    """相容新舊格式：頂層 fields/data 或 tables[] 內的多表。"""
    if payload.get("fields") and payload.get("data"):
        return {"fields": payload["fields"], "data": payload["data"]}
    for table in payload.get("tables") or []:
        if "證券代號" in (table.get("fields") or []):
            return table
    return None


class WholeMarketQuotes:
    """用 TWSE MI_INDEX 抓「一天全市場」行情，供選股排名。"""

    def __init__(self, base_url: str | None = None, session: httpx.Client | None = None) -> None:
        self.base_url = (base_url or settings.twse_base_url).rstrip("/")
        self._session = session or httpx.Client(timeout=45.0)

    def _get_json(self, url: str, params: dict) -> dict:
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_day(self, iso_date: str) -> pl.DataFrame:
        """取得某日全市場（除權證指數外）個股摘要：symbol/name/close/volume_shares/turnover_twd。"""
        ymd = iso_date.replace("-", "")
        payloads = (
            self._get_json(
                f"{self.base_url}/rwd/zh/afterTrading/MI_INDEX",
                {"date": ymd, "type": "ALLBUT0999", "response": "json"},
            ),
            self._get_json(
                f"{self.base_url}/exchangeReport/MI_INDEX",
                {"date": ymd, "type": "ALLBUT0999", "response": "json"},
            ),
        )
        table = None
        for payload in payloads:
            if isinstance(payload, dict) and payload.get("stat") == "OK":
                table = _extract_table(payload)
                if table:
                    break
        if table is None:
            return pl.DataFrame(schema={"symbol": pl.Utf8, "name": pl.Utf8, "close": pl.Float64, "volume_shares": pl.Int64, "turnover_twd": pl.Float64})

        fields = table["fields"]

        def idx(col: str) -> int | None:
            return fields.index(col) if col in fields else None

        i_sym, i_name, i_close = idx("證券代號"), idx("證券名稱"), idx("收盤價")
        i_vol, i_amt = idx("成交股數"), idx("成交金額")
        records = []
        for row in table["data"]:
            sym = row[i_sym].strip()
            if not SYMBOL_RE.match(sym):
                continue
            close = _num(row[i_close]) if i_close is not None else None
            vol = int(_num(row[i_vol]) or 0) if i_vol is not None else 0
            amt = _num(row[i_amt]) if i_amt is not None else None
            records.append({"symbol": sym, "name": row[i_name] if i_name is not None else "", "close": close, "volume_shares": vol, "turnover_twd": amt})
        return pl.DataFrame(records)

    def fetch_recent_days(self, sample_days: int = 5, max_lookback: int = 15) -> list[pl.DataFrame]:
        """往前找最近 sample_days 個交易日（自動跳過週末/休市）。"""
        frames: list[pl.DataFrame] = []
        cursor = date.today()
        tried = 0
        while len(frames) < sample_days and tried < max_lookback:
            if cursor.weekday() >= 5:
                cursor -= timedelta(days=1)
                continue
            df = self.fetch_day(cursor.isoformat())
            if df.height > 100:  # 有效交易日才有大量列
                frames.append(df.with_columns(pl.lit(cursor.isoformat()).alias("source_date")))
            cursor -= timedelta(days=1)
            tried += 1
            time.sleep(0.2)
        return frames


def select_top_liquid(
    quotes: list[pl.DataFrame],
    n: int = 300,
    exclude_etf_prefix: bool = True,
    min_price: float = 5.0,
) -> pl.DataFrame:
    """依平均成交金額排名，回傳前 N 名（附 rank/avg_turnover/交易日數）。"""
    if not quotes:
        return pl.DataFrame()
    merged = pl.concat(quotes)
    eligible = (
        pl.col("close").is_not_null()
        & (pl.col("close") >= min_price)
        & (pl.col("volume_shares") > 0)
    )
    if exclude_etf_prefix:
        eligible &= ~pl.col("symbol").str.starts_with("00")
    ranked = (
        merged.filter(eligible)
        .group_by(["symbol", "name"])
        .agg(
            pl.col("turnover_twd").mean().alias("avg_turnover_twd"),
            pl.col("source_date").n_unique().alias("sampled_days"),
        )
        .sort("avg_turnover_twd", descending=True)
        .with_row_index(name="rank")
        .head(n)
    )
    return ranked.select(["rank", "symbol", "name", "avg_turnover_twd", "sampled_days"])

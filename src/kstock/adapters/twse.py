"""TWSE OpenAPI Adapter（官方資料：校驗 / 補資料，公開端點無需金鑰）。

不作為主要歷史資料來源。
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import polars as pl

from kstock.adapters.base import DataSourceAdapter
from kstock.config.settings import settings
from kstock.models.schema import empty_dataframe
from kstock.normalizers.volume import volume_to_shares

_FIELDS_EXPECTED = ("日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數")


def roc_date_to_iso(value: str) -> str:
    """民國年日期 '112/08/28' → ISO '2023-08-28'。"""
    parts = value.replace(" ", "").split("/")
    if len(parts) != 3:
        raise ValueError(f"無法解析民國日期: {value!r}")
    year = int(parts[0]) + 1911
    return f"{year:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _iter_months(start_date: str, end_date: str) -> list[tuple[int, int]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()
    months: list[tuple[int, int]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


class TWSEAdapter(DataSourceAdapter):
    name = "twse"

    def __init__(
        self,
        base_url: str | None = None,
        volume_unit: str = "shares",
        session: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.twse_base_url).rstrip("/")
        self.volume_unit = volume_unit
        self._session = session or httpx.Client(timeout=30.0)

    def get_instruments(self) -> pl.DataFrame:
        return empty_dataframe("instrument")

    def _get_json(self, url: str, params: dict) -> dict:
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_daily_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
    ) -> pl.DataFrame:
        if not start_date:
            start_date = "2000-01-01"
        end_date = end_date or date.today().isoformat()
        records = []
        for year, month in _iter_months(start_date, end_date):
            payload = self._get_json(
                f"{self.base_url}/exchangeReport/STOCK_DAY",
                {"response": "json", "date": f"{year}{month:02d}01", "stockNo": symbol},
            )
            if payload.get("stat") != "OK":
                continue
            header = payload.get("fields") or list(_FIELDS_EXPECTED)
            records.extend(_parse_month_rows(payload.get("data") or [], header, symbol, market, self.volume_unit))
        if not records:
            return empty_dataframe("daily")
        df = pl.DataFrame(records)
        return df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))


def _parse_month_rows(rows: list[list], fields: list[str], symbol: str, market: str, volume_unit: str) -> list[dict]:
    def idx(name: str) -> int | None:
        try:
            return fields.index(name)
        except ValueError:
            return None

    _d = idx("日期")
    _price = (idx("開盤價"), idx("最高價"), idx("最低價"), idx("收盤價"))
    _volume = idx("成交股數")
    _turnover = idx("成交金額")
    _tickets = idx("成交筆數")

    out = []
    for r in rows:
        iso = roc_date_to_iso(r[_d])
        o, h, l, c = (_to_float(r[i]) if i is not None else None for i in _price)
        out.append(
            {
                "symbol": symbol,
                "market": market,
                "date": iso,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume_shares": volume_to_shares(_to_float(r[_volume]) if _volume is not None else 0, unit=volume_unit),
                "turnover_twd": _to_float(r[_turnover]) if _turnover is not None else None,
                "trade_count": _to_int(r[_tickets]) if _tickets is not None else None,
                "source": "twse",
            }
        )
    return out


def _to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None
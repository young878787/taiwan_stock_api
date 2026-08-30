"""FinMind Adapter（主要研究資料來源）。

設計：
- fetch_rows(dataset, ...) 取得原始 JSON 列（不成形，除錯用）
- get_instruments / get_daily_bars 輸出標準 NORMALIZED Schema
- 原始成交量單位由 settings.finmind_volume_unit 控制（預設 lots=張）

注意：FinMind 各 dataset 欄位可能隨官方調整，不確定時用
`FinMindAdapter.field_names(dataset)` 印出實際欄位來對齊。
"""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl

from kstock.adapters.base import DataSourceAdapter
from kstock.config.settings import settings
from kstock.models.schema import empty_dataframe
from kstock.normalizers.volume import volume_to_shares

_STOCK_PRICE_DATASETS = (
    "TaiwanStockPrice",
    "TaiwanStockPriceAdj",
)
_STOCK_INFO_DATASET = "TaiwanStockInfo"
_INSTITUTIONAL_DATASET = "TaiwanStockInstitutionalInvestorsBuySellWide"
_MARGIN_DATASET = "TaiwanStockMarginPurchaseShortSale"
MARGIN_LOTS = 1000

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("stock_id", "symbol"),
    "date": ("date", "trade_date", "date_time"),
    "open": ("open", "open_price"),
    "high": ("max", "high", "max_price"),
    "low": ("min", "low", "min_price"),
    "close": ("close", "close_price"),
    "volume": ("Trading_Volume", "trading_volume", "volume"),
    "turnover": ("Trading_money", "trading_money", "turnover", "amount"),
    "trade_count": ("Trading_turnover", "Trading_tickets", "trading_turnover", "trade_count"),
}


def _pick(row: dict[str, Any], field: str, default: Any = None) -> Any:
    for alias in _FIELD_ALIASES.get(field, (field,)):
        if alias in row:
            return row[alias]
    return default


class FinMindAdapter(DataSourceAdapter):
    name = "finmind"

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        volume_unit: str | None = None,
        session: httpx.Client | None = None,
    ) -> None:
        self.token = token if token is not None else settings.finmind_token
        self.base_url = (base_url or settings.finmind_base_url).rstrip("/")
        self.volume_unit = volume_unit or settings.finmind_volume_unit
        self._session = session or httpx.Client(timeout=30.0)

    @property
    def data_url(self) -> str:
        return f"{self.base_url}/data"

    def _get_json(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        req_params = dict(params)
        if self.token:
            req_params["token"] = self.token
        resp = self._session.get(self.data_url, params=req_params)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        raise ValueError(f"FinMind 回應格式異常: {payload!r}")

    def fetch_rows(self, dataset: str, **params: Any) -> list[dict[str, Any]]:
        query = {"dataset": dataset, **params}
        return self._get_json(query)

    def field_names(self, dataset: str, **params: Any) -> list[str]:
        rows = self.fetch_rows(dataset, limit=1, **params)
        if not rows:
            return []
        return list(rows[0].keys())

    def get_daily_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
    ) -> pl.DataFrame:
        rows = self.fetch_rows(
            _STOCK_PRICE_DATASETS[0],
            data_id=symbol,
            start_date=start_date or "",
            end_date=end_date or "",
            limit=5000,
        )
        records = []
        for row in rows:
            close = _to_float(_pick(row, "close"))
            if close is None or close <= 0:
                # FinMind 對停牌日會回傳 OHLC/量全 0 的「列」而非缺列。
                # 0 元價格不可能是真實成交，直接略過，避免 0 價污染下游
                # （inf 報酬、復權因子除零、量能指標失真等）。
                continue
            records.append(
                {
                    "symbol": _pick(row, "symbol") or symbol,
                    "market": market,
                    "date": _pick(row, "date"),
                    "open": _to_float(_pick(row, "open")),
                    "high": _to_float(_pick(row, "high")),
                    "low": _to_float(_pick(row, "low")),
                    "close": close,
                    "volume_shares": volume_to_shares(_pick(row, "volume"), unit=self.volume_unit),
                    "turnover_twd": _to_float(_pick(row, "turnover")),
                    "trade_count": _to_int(_pick(row, "trade_count")),
                    "source": self.name,
                }
            )
        if not records:
            return empty_dataframe("daily")
        df = pl.DataFrame(records)
        return df.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d").cast(pl.Date).alias("date")
        )

    def get_instruments(self) -> pl.DataFrame:
        rows = self.fetch_rows(_STOCK_INFO_DATASET)
        records = []
        for row in rows:
            records.append(
                {
                    "symbol": _pick(row, "symbol"),
                    "name": row.get("stock_name") or row.get("name") or "",
                    "market": (
                        "TSE"
                        if row.get("type") in ("twse", "上市")
                        else "OTC"
                        if row.get("type") in ("tpex", "上櫃", "興櫃")
                        else row.get("type") or ""
                    ),
                    "industry": row.get("industry") or "",
                    "list_date": row.get("list_date") or row.get("date") or "",
                    "delist_date": row.get("delist_date") or "",
                    "status": (
                        "active"
                        if row.get("status") in (None, "正常")
                        else str(row.get("status", ""))
                    ),
                }
            )
        if not records:
            return empty_dataframe("instrument")
        df = pl.DataFrame(records)
        return df.with_columns(pl.col("list_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))

    def get_institutional(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pl.DataFrame:
        rows = self.fetch_rows(
            _INSTITUTIONAL_DATASET,
            data_id=symbol,
            start_date=start_date or "",
            end_date=end_date or "",
            limit=10000,
        )
        records = []
        for row in rows:
            records.append(
                {
                    "symbol": row.get("stock_id") or symbol,
                    "date": row.get("date"),
                    "foreign_buy": _i(row, "Foreign_Investor_buy"),
                    "foreign_sell": _i(row, "Foreign_Investor_sell"),
                    "investment_trust_buy": _i(row, "Investment_Trust_buy"),
                    "investment_trust_sell": _i(row, "Investment_Trust_sell"),
                    "dealer_buy": _i(row, "Dealer_buy") + _i(row, "Dealer_self_buy") + _i(row, "Dealer_Hedging_buy"),
                    "dealer_sell": _i(row, "Dealer_sell") + _i(row, "Dealer_self_sell") + _i(row, "Dealer_Hedging_sell"),
                    "source": self.name,
                }
            )
        if not records:
            return empty_dataframe("institutional")
        return _from_iso_date(pl.DataFrame(records))

    def get_margin(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pl.DataFrame:
        rows = self.fetch_rows(
            _MARGIN_DATASET,
            data_id=symbol,
            start_date=start_date or "",
            end_date=end_date or "",
        )
        records = []
        for row in rows:
            records.append(
                {
                    "symbol": row.get("stock_id") or symbol,
                    "date": row.get("date"),
                    "margin_balance": _i(row, "MarginPurchaseTodayBalance") * MARGIN_LOTS,
                    "margin_buy": _i(row, "MarginPurchaseBuy") * MARGIN_LOTS,
                    "margin_sell": _i(row, "MarginPurchaseSell") * MARGIN_LOTS,
                    "short_balance": _i(row, "ShortSaleTodayBalance") * MARGIN_LOTS,
                    "short_sell": _i(row, "ShortSaleSell") * MARGIN_LOTS,
                    "short_cover": _i(row, "ShortSaleBuy") * MARGIN_LOTS,
                    "source": self.name,
                }
            )
        if not records:
            return empty_dataframe("margin")
        return _from_iso_date(pl.DataFrame(records))


def _i(row: dict[str, Any], name: str) -> float:
    return _to_float(row.get(name)) or 0.0


def _from_iso_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d").cast(pl.Date).alias("date"))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    v = _to_float(value)
    return int(v) if v is not None else None
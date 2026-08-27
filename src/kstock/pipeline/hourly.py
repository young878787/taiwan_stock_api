"""小時K資料更新管道：Yahoo Finance → raw/normalized Parquet。

與 DailyUpdatePipeline 同構，但：
- table = "hourly"（date 欄位為 naive Datetime，標記小時開始）
- 單檔 raw（provider=yfinance），normalized 依 year 分區
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from kstock.adapters.base import DataSourceAdapter
from kstock.models.schema import validate_columns
from kstock.storage.parquet import ParquetStore


@dataclass
class HourlyUpdateResult:
    symbol: str
    row_count: int
    raw_path: str


class HourlyUpdatePipeline:
    def __init__(self, adapter: DataSourceAdapter, store: ParquetStore) -> None:
        self.adapter = adapter
        self.store = store

    def update_symbol(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
        write_raw: bool = True,
    ) -> HourlyUpdateResult:
        bars = self.adapter.get_hourly_bars(symbol, start_date, end_date, market)
        if bars.is_empty():
            return HourlyUpdateResult(symbol=symbol, row_count=0, raw_path="")
        validate_columns(bars, "hourly")
        if write_raw:
            raw_path = self.store.write_raw(self.adapter.name, "hourly", bars)
        else:
            raw_path = ""
        self.store.write_normalized("hourly", bars)
        return HourlyUpdateResult(symbol=symbol, row_count=bars.height, raw_path=str(raw_path))

    def run(
        self,
        symbols: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
        write_raw: bool = True,
    ) -> list[HourlyUpdateResult]:
        results: list[HourlyUpdateResult] = []
        for symbol in symbols:
            results.append(
                self.update_symbol(symbol, start_date, end_date, market, write_raw)
            )
        return results

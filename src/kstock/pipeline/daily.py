"""每日資料更新管道：下載 → 標準化 → 寫入 raw/normalized → 校驗。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import polars as pl

from kstock.adapters.base import DataSourceAdapter
from kstock.models.schema import validate_columns
from kstock.pipeline.validation import ValidationReport, compare_sources
from kstock.quality.checks import QualityReport, check_daily_quality
from kstock.storage.duckdb import DuckStore
from kstock.storage.parquet import ParquetStore


@dataclass
class DailyUpdateResult:
    symbol: str
    row_count: int
    raw_path: str
    validation: ValidationReport = field(default_factory=ValidationReport)
    quality: QualityReport = field(default_factory=QualityReport)


class DailyPipelineError(Exception):
    pass


class DailyUpdatePipeline:
    def __init__(
        self,
        adapter: DataSourceAdapter,
        store: ParquetStore,
        duck: DuckStore | None = None,
        validator_adapter: DataSourceAdapter | None = None,
    ) -> None:
        self.adapter = adapter
        self.store = store
        self.duck = duck
        self.validator_adapter = validator_adapter

    def update_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: str = "TSE",
        write_raw: bool = True,
        validate: bool = True,
        check_quality: bool = True,
    ) -> DailyUpdateResult:
        bars = self.adapter.get_daily_bars(symbol, start_date, end_date, market)
        if bars.is_empty():
            return DailyUpdateResult(symbol=symbol, row_count=0, raw_path="")
        validate_columns(bars, "daily")
        if write_raw:
            raw_path = self.store.write_raw(self.adapter.name, "daily", bars)
        else:
            raw_path = ""
        self.store.write_normalized("daily", bars)
        report = ValidationReport()
        if validate and self.validator_adapter is not None:
            official = self.validator_adapter.get_daily_bars(symbol, start_date, end_date, market)
            report = compare_sources(bars, official)
        quality = check_daily_quality(bars) if check_quality else QualityReport()
        return DailyUpdateResult(
            symbol=symbol,
            row_count=bars.height,
            raw_path=str(raw_path),
            validation=report,
            quality=quality,
        )

    def run(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        market: str = "TSE",
        write_raw: bool = True,
        validate: bool = True,
        check_quality: bool = True,
    ) -> list[DailyUpdateResult]:
        results: list[DailyUpdateResult] = []
        for symbol in symbols:
            result = self.update_symbol(symbol, start_date, end_date, market, write_raw, validate, check_quality)
            results.append(result)
        if self.duck is not None:
            self.duck.register_views(("daily",))
        return results
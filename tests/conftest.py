from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl
import pytest

from kstock.config.settings import Settings
from kstock.storage.duckdb import DuckStore
from kstock.storage.parquet import ParquetStore


def make_daily_bars(
    symbol: str = "2330",
    dates: list[str] | None = None,
    prices: list[float] | None = None,
    market: str = "TSE",
) -> pl.DataFrame:
    dates = dates or ["2024-01-02", "2024-01-03", "2024-01-04"]
    prices = prices or [590.0, 598.0, 601.0]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "market": [market] * len(dates),
            "date": [date(int(d[:4]), int(d[5:7]), int(d[8:10])) for d in dates],
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume_shares": [1_000_000] * len(dates),
            "turnover_twd": [1.2e9] * len(dates),
            "trade_count": [10_000] * len(dates),
            "source": ["test"] * len(dates),
        }
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        finmind_token="test-token",
        finmind_volume_unit="lots",
    )


@pytest.fixture
def store(test_settings: Settings) -> ParquetStore:
    return ParquetStore(settings_=test_settings)


@pytest.fixture
def duck(store: ParquetStore) -> Iterable[DuckStore]:
    d = DuckStore(settings_=Settings(data_dir=store.root))
    yield d
    d.close()
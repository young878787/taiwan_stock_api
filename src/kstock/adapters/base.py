from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class DataSourceAdapter(ABC):
    """資料來源統一介面：所有 adapter 都輸出 NORMALIZED Schema。"""

    name: str = "base"

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        market: str = "TSE",
    ) -> pl.DataFrame:
        """回傳符合 daily_bar schema 的 DataFrame（含 volume_shares/turnover_twd/trade_count/source）。"""

    @abstractmethod
    def get_instruments(self) -> pl.DataFrame:
        """回傳符合 instrument schema 的 DataFrame。"""

    def get_institutional(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
        raise NotImplementedError(f"{self.name} 尚未提供 institutional 資料")

    def get_margin(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
        raise NotImplementedError(f"{self.name} 尚未提供 margin 資料")
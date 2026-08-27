"""Parquet 分層儲存（raw / normalized）。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kstock.config.settings import Settings, settings
from kstock.models.schema import TABLE_DTYPES


class ParquetStore:
    def __init__(self, root: Path | None = None, settings_: Settings | None = None) -> None:
        s = settings_ or settings
        self.root = Path(root) if root is not None else s.data_dir
        self.raw_dir = self.root / "raw"
        self.normalized_dir = self.root / "normalized"
        self.features_dir = self.root / "features"

    # ---- raw（保留原始資料） ----
    def write_raw(self, provider: str, dataset: str, df: pl.DataFrame) -> Path:
        target = self.raw_dir / provider / dataset / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        merged = df
        if target.exists():
            existing = pl.read_parquet(target)
            merged = pl.concat([existing, df])
            if "date" in merged.columns:
                merged = merged.unique(subset=["symbol", "date"], keep="last")
        merged.write_parquet(target, compression="zstd")
        return target

    # ---- normalized（標準化後） ----
    def write_normalized(self, table: str, df: pl.DataFrame) -> None:
        if df.height == 0:
            return
        if "date" not in df.columns:
            raise ValueError(f"{table} 缺少 date 欄位")
        table_dir = self.normalized_dir / table
        table_dir.mkdir(parents=True, exist_ok=True)
        with_year = df.with_columns(pl.col("date").dt.strftime("%Y").alias("year"))
        for year in with_year["year"].unique().sort().to_list():
            part_dir = table_dir / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part = with_year.filter(pl.col("year") == year).drop("year")
            target = part_dir / "data.parquet"
            merged = pl.read_parquet(target) if target.exists() else None
            if merged is not None:
                part = pl.concat([merged, part]).unique(subset=["symbol", "date"], keep="last")
            part.write_parquet(target, compression="zstd")

    def normalized_files(self, table: str) -> list[Path]:
        return sorted((self.normalized_dir / table).glob("year=*/*.parquet"))

    def read_raw(self, provider: str, dataset: str) -> pl.DataFrame | None:
        target = self.raw_dir / provider / dataset / "data.parquet"
        return pl.read_parquet(target) if target.exists() else None

    def read_normalized(
        self,
        table: str,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        files = self.normalized_files(table)
        if not files:
            return pl.DataFrame(schema=TABLE_DTYPES[table])
        df = pl.read_parquet(files)
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        if start_date:
            df = df.filter(pl.col("date") >= pl.datetime(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10])).cast(pl.Date))
        if end_date:
            df = df.filter(pl.col("date") <= pl.datetime(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10])).cast(pl.Date))
        return df

    # ---- features ----
    def write_feature(self, name: str, df: pl.DataFrame) -> None:
        target = self.features_dir / f"{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(target, compression="zstd")

    def read_feature(self, name: str) -> pl.DataFrame | None:
        target = self.features_dir / f"{name}.parquet"
        return pl.read_parquet(target) if target.exists() else None
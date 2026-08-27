"""DuckDB 查詢層（直接查 normalized Parquet，不 Import 進 DB）。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from kstock.config.settings import Settings, settings


class DuckStore:
    def __init__(self, settings_: Settings | None = None, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        s = settings_ or settings
        self.normalized_dir = s.normalized_dir
        self._con = connection or duckdb.connect()

    def close(self) -> None:
        self._con.close()

    def register_view(self, table: str) -> None:
        files = sorted(self.normalized_dir.joinpath(table).glob("year=*/*.parquet"))
        if not files:
            self._con.execute(
                f"CREATE OR REPLACE VIEW {table} AS SELECT NULL::INT AS _empty WHERE 1=0"
            )
            return
        glob = str(self.normalized_dir / table / "year=*" / "*.parquet").replace("\\", "/")
        self._con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{glob}')"
        )

    def register_views(self, tables: tuple[str, ...] = ("daily", "minute", "tick", "instrument", "institutional", "margin")) -> None:
        for t in tables:
            self.register_view(t)

    def query(self, sql: str, params: tuple | None = None) -> pl.DataFrame:
        args = params or ()
        result = self._con.execute(sql, args)
        return pl.from_arrow(result.to_arrow_table())

    def table_exists(self, table: str) -> bool:
        try:
            self._con.execute(f"SELECT * FROM {table} LIMIT 0")
            return True
        except Exception:
            return False

    def query_parquet(self, parquet_glob: str | Path, sql: str) -> pl.DataFrame:
        """直接對 Parquet glob 執行 SQL（不建 View）。"""
        safe = str(parquet_glob).replace("\\", "/")
        resolved = f"read_parquet('{safe}')"
        sql = sql.replace("<TABLE>", resolved)
        return self.query(sql)
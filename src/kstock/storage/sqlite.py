"""作業 API 的 SQLite 儲存層，每次操作使用獨立連線與交易。"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from kstock.models.schema import DAILY_BAR_COLUMNS

_COLUMNS = ", ".join(DAILY_BAR_COLUMNS)
_PLACEHOLDERS = ", ".join("?" for _ in DAILY_BAR_COLUMNS)


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS daily (
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL CHECK (market = 'TSE'),
                    date TEXT NOT NULL,
                    open REAL CHECK (open >= 0),
                    high REAL CHECK (high >= 0),
                    low REAL CHECK (low >= 0),
                    close REAL CHECK (close >= 0),
                    volume_shares INTEGER CHECK (volume_shares >= 0),
                    turnover_twd REAL CHECK (turnover_twd >= 0),
                    trade_count INTEGER CHECK (trade_count >= 0),
                    source TEXT NOT NULL CHECK (source IN ('twse', 'manual')),
                    PRIMARY KEY (symbol, date)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS daily_date_symbol ON daily(date, symbol)")

    def import_daily(self, records: list[dict]) -> int:
        """官方資料依自然主鍵更新；整批成功才提交，其他主鍵不受影響。"""
        updates = ", ".join(f"{column}=excluded.{column}" for column in DAILY_BAR_COLUMNS if column not in ("symbol", "date"))
        with self.connect() as connection:
            connection.executemany(
                f"INSERT INTO daily ({_COLUMNS}) VALUES ({_PLACEHOLDERS}) "
                f"ON CONFLICT(symbol, date) DO UPDATE SET {updates}",
                [tuple(record[column] for column in DAILY_BAR_COLUMNS) for record in records],
            )
        return len(records)

    def create(self, record: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO daily ({_COLUMNS}) VALUES ({_PLACEHOLDERS})",
                tuple(record[column] for column in DAILY_BAR_COLUMNS),
            )

    def get(self, symbol: str, date: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM daily WHERE symbol=? AND date=?", (symbol, date)).fetchone()
        return dict(row) if row else None

    def list(self, symbol: str | None, start_date: str | None, end_date: str | None, limit: int, offset: int) -> dict:
        conditions, parameters = [], []
        for column, operator, value in (("symbol", "=", symbol), ("date", ">=", start_date), ("date", "<=", end_date)):
            if value is not None:
                conditions.append(f"{column} {operator} ?")
                parameters.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connect() as connection:
            # 明確開啟讀取交易，讓 total 與 items 來自相同快照。
            connection.execute("BEGIN")
            total = connection.execute("SELECT COUNT(*) FROM daily" + where, parameters).fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM daily" + where + " ORDER BY date DESC, symbol ASC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    def replace(self, symbol: str, date: str, values: dict) -> bool:
        columns = [column for column in DAILY_BAR_COLUMNS if column not in ("symbol", "date")]
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE daily SET " + ", ".join(f"{column}=?" for column in columns) + " WHERE symbol=? AND date=?",
                [*(values[column] for column in columns), symbol, date],
            )
        return cursor.rowcount > 0

    def delete(self, symbol: str, date: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM daily WHERE symbol=? AND date=?", (symbol, date))
        return cursor.rowcount > 0

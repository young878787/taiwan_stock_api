"""歷史資料回填（backfill）：把 daily / institutional / margin 補到更早年份。

既有資料從 2021-01-04 起；本腳本把指定宇宙的資料按年分段回填到
``--start``（預設 2012-01-01），沿用 DailyUpdatePipeline 的標準化與
ParquetStore 寫入（(symbol, date) unique keep=last，重跑冪等）。

設計重點：
- **按年分段抓**：FinMind `TaiwanStockPrice` 單次 limit=5000，2008~2026
  全程一次抓貼近上限；按年分段每年約 250 列，安全且可斷點續傳。
- **節流**：FinMind 免費 token 約 600 請求/小時，預設每請求間隔 0.5 秒
  （--sleep 可調），超量時自動等待後重試。
- **宇宙**：預設 data/universe/top_liquidity_300.txt（現有 300 檔）；
  --symbols 可覆寫（逗號分隔）。
- **斷點續傳**：以 normalized 既有 (symbol, 年) 資料集為跳過依據，重跑只補
  失敗/缺漏段，成功段不重打（省 FinMind 額度）。副作用：上市前空年每次
  重跑會再試一次（回空列，無害）。
- 停牌/未上市期間 FinMind 回空列 → 該年 0 列屬正常，不視為錯誤。

用法::

    uv run python -m kstock.backfill                     # 2012 起，300 檔
    uv run python -m kstock.backfill --start 2008-01-01  # 回填到 2008
    uv run python -m kstock.backfill --symbols 2330,2454 --tables daily
    uv run python -m kstock.backfill --dry-run           # 只列計畫不抓
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from kstock.adapters.finmind import FinMindAdapter
from kstock.config.settings import settings
from kstock.storage.parquet import ParquetStore

DEFAULT_START = "2012-01-01"
DEFAULT_UNIVERSE = "top_liquidity_300.txt"
DEFAULT_TABLES = ("daily", "institutional", "margin")


@dataclass
class BackfillStats:
    requests: int = 0
    rows_written: int = 0
    empty_responses: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (table, symbol, year)

    def __str__(self) -> str:
        lines = [
            f"請求數：{self.requests}，寫入列數：{self.rows_written:,}，空回應：{self.empty_responses}"
        ]
        if self.failures:
            lines.append(f"失敗 {len(self.failures)} 筆（重跑本腳本可續補）：")
            lines.extend(f"  {t} {s} {y}" for t, s, y in self.failures[:20])
            if len(self.failures) > 20:
                lines.append(f"  ...（其餘 {len(self.failures) - 20} 筆略）")
        return "\n".join(lines)


def load_universe_symbols(path: Path) -> list[str]:
    """宇宙檔（每行一個代碼）→ symbol 清單。"""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def year_segments(start: str, end: str) -> list[tuple[str, str]]:
    """[start, end] 切成每年一段的 [(YYYY-01-01, YYYY-12-31), ...]。"""
    y0 = int(start[:4])
    y1 = int(end[:4])
    today = date.today().isoformat()
    segments: list[tuple[str, str]] = []
    for y in range(y0, y1 + 1):
        seg_start = max(f"{y}-01-01", start)
        seg_end = min(f"{y}-12-31", end, today)
        if seg_start > seg_end:
            continue
        segments.append((seg_start, seg_end))
    return segments


class QuotaExceededError(RuntimeError):
    """FinMind 配額耗盡（HTTP 402 重試多次仍失敗）→ 中止整輪，重跑續補。"""


def _clean_error(e: Exception) -> str:
    """錯誤訊息遮罩 token（FinMind URL 帶 token，避免外洩到 log）。"""
    text = str(e)
    if "token=" in text:
        text = text.split("token=")[0].rstrip("&?") + " token=***"
    return text


def _fetch_with_retry(
    adapter: FinMindAdapter,
    table: str,
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 4,
    quota_retries: int = 6,
) -> pl.DataFrame:
    """帶限流等待的抓取重試。

    FinMind 以 HTTP 402 表示「too many requests」（免費 token 約 600 請求/小時）。
    402 用獨立指數退避（90s→3min→6min→12min…，共 quota_retries 次，約半小時）
    等配額回滾；連續失敗仍 402 → QuotaExceededError（中止整輪，重跑續補）。
    429/403 與其他 HTTP 錯誤用 max_retries 次短退避。
    """
    quota_failures = 0
    other_failures = 0
    while True:
        try:
            if table == "daily":
                return adapter.get_daily_bars(symbol, start_date, end_date)
            if table == "institutional":
                return adapter.get_institutional(symbol, start_date, end_date)
            return adapter.get_margin(symbol, start_date, end_date)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 402:
                quota_failures += 1
                if quota_failures > quota_retries:
                    raise QuotaExceededError(
                        f"FinMind 配額耗盡（HTTP 402 連續 {quota_failures} 次失敗）"
                    ) from e
                wait = 90 * 2 ** (quota_failures - 1)  # 90s, 3min, 6min, 12min, 24min, 48min
                print(f"    限流（HTTP 402），等待 {wait}s 後重試…", flush=True)
                time.sleep(wait)
                continue
            if status in (429, 403):
                other_failures += 1
                if other_failures > max_retries:
                    raise
                wait = 60 * other_failures
                print(f"    限流（HTTP {status}），等待 {wait}s 後重試…", flush=True)
                time.sleep(wait)
                continue
            raise
        except httpx.HTTPError:
            other_failures += 1
            if other_failures > max_retries:
                raise
            time.sleep(5 * other_failures)


def backfill_symbol_year(
    adapter: FinMindAdapter,
    store: ParquetStore,
    table: str,
    symbol: str,
    start_date: str,
    end_date: str,
    write_raw: bool,
    stats: BackfillStats,
) -> int:
    """抓單一 (table, symbol, 年段) 並寫入 normalized；回傳寫入列數。"""
    stats.requests += 1
    df = _fetch_with_retry(adapter, table, symbol, start_date, end_date)
    if df.is_empty():
        stats.empty_responses += 1
        return 0
    if write_raw:
        store.write_raw(adapter.name, table, df)
    store.write_normalized(table, df)
    stats.rows_written += df.height
    return df.height


def run_backfill(
    symbols: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    tables: tuple[str, ...] = DEFAULT_TABLES,
    sleep_seconds: float = 0.5,
    write_raw: bool = False,
    dry_run: bool = False,
    adapter: FinMindAdapter | None = None,
    store: ParquetStore | None = None,
) -> BackfillStats:
    """執行回填主流程（adapter/store 可注入，測試用）。"""
    store = store or ParquetStore()
    adapter = adapter or FinMindAdapter()
    end = end or date.today().isoformat()
    segments = year_segments(start, end)
    stats = BackfillStats()

    if dry_run:
        segments = year_segments(start, end)
        print(
            f"[dry-run] 宇宙 {len(symbols)} 檔 × {len(tables)} 表 × {len(segments)} 年段 "
            f"= {len(symbols) * len(tables) * len(segments)} 請求"
        )
        for s in segments:
            print(f"  {s[0]} ~ {s[1]}")
        return stats

    # 每個 (symbol, 年) 段只要 normalized 已有該檔該年的任何資料就跳過：
    # 重跑 = 只補失敗/缺漏段，成功段不重打（省 FinMind 額度）
    existing_years = _existing_symbol_years(store, tables)

    total = len(symbols) * len(tables) * len(segments)
    done = 0
    for table in tables:
        have = existing_years.get(table, set())
        for symbol in symbols:
            for seg_start, seg_end in segments:
                year = seg_start[:4]
                if (symbol, year) in have:
                    continue
                done += 1
                try:
                    n = backfill_symbol_year(
                        adapter, store, table, symbol, seg_start, seg_end, write_raw, stats
                    )
                    if done % 50 == 0 or n > 0:
                        print(f"  [{done}/{total}] {table} {symbol} {year}: {n} 列", flush=True)
                except QuotaExceededError as e:
                    stats.failures.append((table, symbol, year))
                    print(
                        f"\n[中止] {e}\n已完成 {done}/{total}，剩餘段重跑本腳本即可續補（等額度回滾後）。",
                        flush=True,
                    )
                    return stats
                except Exception as e:  # 單點失敗不中斷整體回填
                    stats.failures.append((table, symbol, year))
                    print(
                        f"  [{done}/{total}] {table} {symbol} {year}: 失敗（{_clean_error(e)}）",
                        flush=True,
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    return stats


def _existing_symbol_years(store: ParquetStore, tables: tuple[str, ...]) -> dict[str, set]:
    """各表既有 (symbol, year) 集合，供斷點續傳跳過。"""
    out: dict[str, set] = {}
    for table in tables:
        files = store.normalized_files(table)
        if not files:
            out[table] = set()
            continue
        df = pl.read_parquet(files, columns=["symbol", "date"]).with_columns(
            pl.col("date").dt.year().cast(str).alias("year")
        )
        out[table] = set(zip(df["symbol"], df["year"]))
    return out


def load_delisted_symbols(
    path: Path, start: str
) -> list[tuple[str, str | None]]:
    """下市清單 CSV（universe_history 產出：symbol,name,delist_date,source）→ [(symbol, delist_date)]。

    - 剔除下市日早於 ``start`` 的標的（回填範圍之外）。
    - delist_date 空值（source=mi_index_sampling 無官方日期）→ 以 None 帶過，
      抓取終點退回今天（由呼叫端裁切防代碼重用污染）。
    """
    df = pl.read_csv(
        path,
        schema_overrides={"symbol": pl.Utf8, "name": pl.Utf8, "source": pl.Utf8},
    ).with_columns(pl.col("delist_date").str.to_date("%Y-%m-%d", strict=False))
    df = df.filter(pl.col("delist_date").is_null() | (pl.col("delist_date") >= date.fromisoformat(start)))
    return [
        (r["symbol"], r["delist_date"].isoformat() if r["delist_date"] else None)
        for r in df.rows(named=True)
    ]


def run_delisted_backfill(
    delisted_csv: Path,
    start: str = DEFAULT_START,
    tables: tuple[str, ...] = ("daily",),
    sleep_seconds: float = 0.5,
    write_raw: bool = False,
    dry_run: bool = False,
    adapter: FinMindAdapter | None = None,
    store: ParquetStore | None = None,
) -> BackfillStats:
    """下市股回填：依官方 delist_date 逐檔裁切（**防代碼重用污染**）。

    與 run_backfill 的差異：
    - 每檔抓取終點 = min(delist_date, 今天)，下市後的列（可能是重用代碼的新公司）一律不抓。
    - 不套用「既有資料起點前一天」的全域 end 調整（下市股不在既有資料內，
      全段都要補）；斷點續傳同樣以 _existing_symbol_years 跳過已補段。
    - 剔除下市日早於 start 的標的。
    """
    store = store or ParquetStore()
    adapter = adapter or FinMindAdapter()
    symbols = load_delisted_symbols(delisted_csv, start)
    stats = BackfillStats()

    if dry_run:
        n_req = sum(
            len(year_segments(start, end_ or date.today().isoformat()))
            for _sym, end_ in symbols
        )
        print(
            f"[dry-run] 下市回填 {len(symbols)} 檔 × {len(tables)} 表 × ~{n_req} 年段 = ~{n_req * len(tables)} 請求"
        )
        return stats

    existing_years = _existing_symbol_years(store, tables)
    today = date.today().isoformat()
    total = sum(len(year_segments(start, end_ or today)) * len(tables) for _sym, end_ in symbols)
    done = 0
    for table in tables:
        have = existing_years.get(table, set())
        for symbol, delist_end in symbols:
            segs = year_segments(start, delist_end or today)
            for seg_start, seg_end in segs:
                year = seg_start[:4]
                if (symbol, year) in have:
                    continue
                done += 1
                try:
                    n = backfill_symbol_year(
                        adapter, store, table, symbol, seg_start, seg_end, write_raw, stats
                    )
                    if done % 50 == 0 or n > 0:
                        print(
                            f"  [{done}/{total}] {table} {symbol} {year}: {n} 列（end={delist_end}）",
                            flush=True,
                        )
                except QuotaExceededError as e:
                    stats.failures.append((table, symbol, year))
                    print(
                        f"\n[中止] {e}\n已完成 {done}/{total}，等額度回滾後重跑本腳本即可續補。",
                        flush=True,
                    )
                    return stats
                except Exception as e:  # 單點失敗不中斷整體回填
                    stats.failures.append((table, symbol, year))
                    print(
                        f"  [{done}/{total}] {table} {symbol} {year}: 失敗（{_clean_error(e)}）",
                        flush=True,
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kstock.backfill",
        description="把 daily/institutional/margin 回填到更早年份（按年分段 + 節流 + 冪等寫入）",
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"回填起點 YYYY-MM-DD（預設 {DEFAULT_START}）")
    parser.add_argument("--end", default=None, help="回填終點 YYYY-MM-DD（預設為既有資料起點前一天）")
    parser.add_argument(
        "--symbols", default=None, help="逗號分隔代碼（預設讀 data/universe/top_liquidity_300.txt）"
    )
    parser.add_argument(
        "--tables", default=",".join(DEFAULT_TABLES), help=f"逗號分隔資料表（預設 {','.join(DEFAULT_TABLES)}）"
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="每請求間隔秒數（預設 0.5，FinMind 限流保護）")
    parser.add_argument("--write-raw", action="store_true", help="同時寫 raw 層（預設只寫 normalized）")
    parser.add_argument("--dry-run", action="store_true", help="只列計畫不發請求")
    parser.add_argument(
        "--delisted-csv",
        default=None,
        help=(
            "下市回填模式：讀 universe_history 的 delisted_tw CSV（symbol,name,delist_date,source），"
            "逐檔回填 daily 並以 delist_date 裁切（防代碼重用污染）；與 --symbols 互斥"
        ),
    )
    ns = parser.parse_args(argv)

    tables = tuple(t.strip() for t in ns.tables.split(",") if t.strip())
    unknown = [t for t in tables if t not in ("daily", "institutional", "margin")]
    if unknown:
        print(f"不支援的資料表：{unknown}", file=sys.stderr)
        return 1

    if ns.delisted_csv:
        path = Path(ns.delisted_csv)
        if not path.exists():
            print(f"找不到下市清單：{path}", file=sys.stderr)
            return 1
        symbols_rows = load_delisted_symbols(path, ns.start)
        print(
            f"下市回填 {len(symbols_rows)} 檔 × {tables}，{ns.start} 起（write_raw={ns.write_raw}）"
        )
        stats = run_delisted_backfill(
            delisted_csv=path,
            start=ns.start,
            tables=tables,
            sleep_seconds=ns.sleep,
            write_raw=ns.write_raw,
            dry_run=ns.dry_run,
        )
        print(stats)
        return 1 if stats.failures else 0

    if ns.symbols:
        symbols = [s.strip() for s in ns.symbols.split(",") if s.strip()]
    else:
        universe = settings.data_dir / "universe" / DEFAULT_UNIVERSE
        if not universe.exists():
            print(f"找不到宇宙檔：{universe}（用 --symbols 指定代碼）", file=sys.stderr)
            return 1
        symbols = load_universe_symbols(universe)
    tables = tuple(t.strip() for t in ns.tables.split(",") if t.strip())
    print(f"回填 {len(symbols)} 檔 × {tables}，{ns.start} 起（write_raw={ns.write_raw}）")
    stats = run_backfill(
        symbols=symbols,
        start=ns.start,
        end=ns.end,
        tables=tables,
        sleep_seconds=ns.sleep,
        write_raw=ns.write_raw,
        dry_run=ns.dry_run,
    )
    print(stats)
    return 1 if stats.failures else 0


if __name__ == "__main__":
    sys.exit(main())

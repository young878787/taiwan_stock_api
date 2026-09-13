"""歷史宇宙回推（WP3 / 階段 C1 修訂版）：MI_INDEX 逐月抽樣 + 官方終止上市清單 → 下市清單。

來源分工：
- **官方「終止上市公司」清單**（TWSE suspendListingCsvAndHtml，Big5/民國年，2001 起）：
  **權威下市來源**——含官方下市日期，且涵蓋「先停牌後下市」的標的
  （停止買賣者不會出現在 MI_INDEX 名單＝抽樣法盲點）；「終止上市」含上市→上櫃轉板，
  轉板股仍在交易，以 instrument 表的活躍清單剔除。
- **MI_INDEX 逐月抽樣**：建立 ``(code, name, 存在區間)`` 對照表（代碼重用切割、
  回補裁切用），並作為官方清單的補充候選（下櫃轉興櫃後再終止興櫃等邊角案例）。

用途：
- 每月抽一個交易日，呼叫 ``WholeMarketQuotes.fetch_day``（TWSE MI_INDEX 全市場報表）
  取得當日全市場 (symbol, name)，union 整段期間 → 歷史曾上市宇宙。
- 對每個代碼建立 ``(symbol, name, 存在區間)`` 對照表：相鄰抽樣日差 ≤ ``gap_days``
  視為同一家公司，差 > ``gap_days`` 視為斷點（代碼重用）。
- 合併官方終止上市清單 + 抽樣下市候選 → 下市清單（供 C2 回補下市股價格）。

限制：
- **僅涵蓋 TSE（上市）**：TPEX（上櫃）的下市（如下櫃的力晶 5346）不在範圍，
  TPEX 歷史列舉端點未解（OpenAPI 需申請 apikey），屬開放項目，v1 不做。
- 抽樣間隔 1 個月：存在區間 < 1 個月（兩次抽樣日之間整段進出）的標的可能漏掉。
- MI_INDEX 歷史深度實測自 2008 年可用（2008-01-04 → 717 檔、2012 → 854、2026 → 1,379）。
- **TWSE 有突發式限流（HTTP 428）**：整月失敗會嚴格中止（缺月會造成假斷點），
  進度以 checkpoint 續存，`--resume` 續跑（見 SampleFailure）。
- 網路邊界集中在 ``fetch_day``／``fetch_terminated_list``：核心函式全部可離線注入 mock 測試。

用法::

    uv run python -m kstock.universe_history --start 2008-01-01 --sleep 5
    uv run python -m kstock.universe_history --resume          # 從 checkpoint 續抽
    uv run python -m kstock.universe_history --dry-run         # 只印抽樣計畫不發請求

產出（``--output`` 目錄，預設 ``<data>/universe/manifests``）：

- ``tw_seen_<今天YYYYMMDD>.csv``：全部存在區間 (symbol, name, first_seen, last_seen, n_days)
- ``delisted_tw_<今天YYYYMMDD>.csv``：下市清單 (symbol, name, delist_date, source)
  （source=twse_terminated 為官方權威；mi_index_sampling 為抽樣補充、無官方日期）
- ``tw_sample_partial.csv``：進行中的 checkpoint（完成後自動刪除）
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from kstock.config.settings import settings
from kstock.storage.parquet import ParquetStore

if TYPE_CHECKING:
    import httpx
from kstock.universe.twse_whole_market import WholeMarketQuotes

DEFAULT_START = "2008-01-01"
DEFAULT_MAX_TRIES = 12
DEFAULT_GAP_DAYS = 90
DEFAULT_INACTIVE_DAYS = 90
DEFAULT_SLEEP = 0.3

SEEN_SCHEMA = {"symbol": pl.Utf8, "name": pl.Utf8, "seen_date": pl.Date}
EPOCH = date(1900, 1, 1)
INTERVALS_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "first_seen": pl.Date,
    "last_seen": pl.Date,
    "n_days": pl.Int64,
}


def _empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _add_month(y: int, m: int, k: int = 1) -> tuple[int, int]:
    total = y * 12 + (m - 1) + k
    return total // 12, total % 12 + 1


def _month_plan(start: str, end: str) -> list[tuple[date, date]]:
    """逐月的（起試日, 試探上限日）：首月起試日 = start、其後每月 1 號；上限 = 月底與 end 取小。"""
    start_d = date(int(start[:4]), int(start[5:7]), int(start[8:10]))
    end_d = date(int(end[:4]), int(end[5:7]), int(end[8:10]))
    plan: list[tuple[date, date]] = []
    y, m = start_d.year, start_d.month
    while True:
        month_first = date(y, m, 1)
        if month_first > end_d:
            break
        probe_start = max(month_first, start_d)
        ny, nm = _add_month(y, m)
        month_last = date(ny, nm, 1) - timedelta(days=1)
        plan.append((probe_start, min(month_last, end_d)))
        y, m = ny, nm
    return plan


class SampleFailure(RuntimeError):
    """整月抽樣失敗（多半為 TWSE 限流 HTTP 428/429/403）——殘缺 manifest 不可用，應中止重跑。"""


def _fetch_with_backoff(
    fetch_day: Callable[[str], pl.DataFrame], iso: str, max_retries: int = 3
) -> pl.DataFrame:
    """帶限流退避的單日抓取：428/429/403 時指數等待後重試，其他例外視為無資料。

    TWSE 的 428 窗口實測可長達數分鐘（突發式限流），退避梯度取 60/180/420s。
    """
    import httpx

    waits = (60, 180, 420)
    for attempt in range(max_retries + 1):
        try:
            return fetch_day(iso)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (428, 429, 403) and attempt < max_retries:
                wait = waits[attempt]
                print(f"  [限流] {iso} HTTP {status}，等待 {wait}s 後重試…", flush=True)
                time.sleep(wait)
                continue
            raise
    return _empty(SEEN_SCHEMA)  # pragma: no cover - 迴圈內已 return/raise


def _monthly_probes(
    start: str,
    end: str,
    fetch_day: Callable[[str], pl.DataFrame],
    max_tries: int = DEFAULT_MAX_TRIES,
    skip_through: tuple[int, int] | None = None,
) -> Iterator[tuple[str, pl.DataFrame, bool]]:
    """逐月探測產生器：每次 fetch 產生 (iso_date, df, hit)。

    hit=True 代表該月找到第一個有效交易日（df 非空）；該月到此為止換下個月。
    單月自起試日起逐日試、最多 max_tries 天皆空才視為該月無資料。
    限流類例外先經退避重試；整月仍失敗（12 天皆無資料）一律 raise
    :class:`SampleFailure`——缺月會讓 gap 判定失真（假斷點＝假代碼重用），
    殘缺 manifest 不可用。

    ``skip_through``（續跑）：(年, 月) —— 該月（含）之前的月份直接跳過不發請求。
    """
    for probe_start, probe_end in _month_plan(start, end):
        if skip_through is not None and (probe_start.year, probe_start.month) <= skip_through:
            continue
        cursor = probe_start
        tries = 0
        hit = False
        while cursor <= probe_end and tries < max_tries:
            tries += 1
            iso = cursor.isoformat()
            try:
                df = _fetch_with_backoff(fetch_day, iso)
            except Exception as e:
                raise SampleFailure(
                    f"{iso} fetch 失敗（{type(e).__name__}）——中止抽樣避免殘缺 manifest（稍後重跑）"
                ) from e
            if not df.is_empty():
                hit = True
                yield iso, df, True
                break
            cursor += timedelta(days=1)
        if not hit:
            raise SampleFailure(
                f"{probe_start.strftime('%Y-%m')} 整月抽不到資料（12 日皆空）——"
                "多半為 TWSE 限流或端點變動，中止避免殘缺 manifest（稍後重跑）"
            )


def monthly_sample_dates(
    start: str,
    end: str,
    fetch_day: Callable[[str], pl.DataFrame],
    max_tries: int = DEFAULT_MAX_TRIES,
) -> list[str]:
    """每個月的第一個有效交易日 ISO 清單。

    每月從 1 號起逐日試 fetch_day（回空 df 視為非交易日），最多 max_tries 天；
    整月失敗（限流未解或無資料）會 raise :class:`SampleFailure`——嚴格模式，
    避免缺月造成存在區間的假斷點。
    """
    return [iso for iso, _df, hit in _monthly_probes(start, end, fetch_day, max_tries) if hit]


def sample_market_history(
    start: str,
    end: str,
    fetch_day: Callable[[str], pl.DataFrame] | None = None,
    sleep_seconds: float = 0.0,
    resume_df: pl.DataFrame | None = None,
    on_progress: Callable[[pl.DataFrame], None] | None = None,
) -> pl.DataFrame:
    """逐月抽樣 → (symbol, name, seen_date) 長表。

    fetch_day 為 None 時用 ``WholeMarketQuotes().fetch_day``（真網路）；
    每次 fetch 後 sleep（禮貌節流）。seen_date = 該次抽樣的 ISO 日期（pl.Date）。
    自行逐月探測（不重打 monthly_sample_dates 的請求），抽到即收；
    限流/缺月會 raise :class:`SampleFailure`（嚴格模式，殘缺資料不可用）。

    ``resume_df``：先前已抽到的長表（checkpoint 續跑），早於其最後一個月的月份直接跳過；
    ``on_progress``：每抽到一個月以**累積**長表回呼一次（checkpoint 落盤用）。
    """
    if fetch_day is None:
        fetch_day = WholeMarketQuotes().fetch_day
    last_month: tuple[int, int] | None = None
    if resume_df is not None and not resume_df.is_empty():
        last_seen = resume_df["seen_date"].max()
        last_month = (last_seen.year, last_seen.month)
        print(f"  [續跑] 跳過已抽月份（最後抽到 {last_seen.isoformat()}）", flush=True)
    frames: list[pl.DataFrame] = [resume_df] if resume_df is not None and not resume_df.is_empty() else []
    for iso, df, hit in _monthly_probes(start, end, fetch_day, skip_through=last_month):
        if hit:
            y, m = int(iso[:4]), int(iso[5:7])
            if last_month is not None and (y, m) <= last_month:
                continue  # 防禦：skip_through 已排除，此為冗餘保險
            frames.append(
                df.select(["symbol", "name"]).with_columns(
                    pl.lit(iso).str.to_date("%Y-%m-%d").alias("seen_date")
                )
            )
            if on_progress is not None:
                on_progress(
                    pl.concat(frames).select(["symbol", "name", "seen_date"]).sort(["symbol", "seen_date"])
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    if not frames:
        return _empty(SEEN_SCHEMA)
    return pl.concat(frames).select(["symbol", "name", "seen_date"]).sort(["symbol", "seen_date"])


def build_seen_intervals(
    seen: pl.DataFrame,
    gap_days: int = DEFAULT_GAP_DAYS,
    terminated: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """(symbol, name, seen_date) → (symbol, name, first_seen, last_seen, n_days)。

    規則：每個 symbol 依 seen_date 排序，切分為「同一公司」的存在區間：

    - ``terminated=None``（預設）：相鄰兩次抽樣日期差 > ``gap_days`` 即切新段
      （純 gap 啟發式）。
    - ``terminated``（官方終止上市清單，**建議**）：切新段 ⇔ 該代碼的下市事件
      落在兩次抽樣日之間（join_asof 比對「至本次抽樣日為止最近一次下市」是否變動）；
      純停牌 gap（如 3018 隆銘綠能 2022-11 停牌、2023-07 恢復交易）會被合併
      而非誤判為代碼重用。區間名稱取區間內**最後一次**出現的名稱（區間中改名屬改名）。

    實測依據（2026-09-13 manifest）：純 gap 啟發式把 14 檔「停牌後恢復交易」的
    上市股切成假公司段；下市事件才是代碼重用的必要條件。已知邊角：下市事件早於
    官方清單起點（2001 前）的重用案例無法以事件切割（退回 gap 語意須傳 None）。
    """
    if seen.is_empty():
        return _empty(INTERVALS_SCHEMA)
    df = seen.with_columns(pl.col("seen_date").cast(pl.Date)).sort(["symbol", "seen_date"])
    df = df.with_columns(
        pl.col("seen_date")
        .diff()
        .dt.total_days()
        .gt(gap_days)
        .fill_null(True)
        .alias("_gap_break")
    )
    if terminated is not None and not terminated.is_empty():
        delists = terminated.select(
            pl.col("symbol").cast(pl.Utf8), pl.col("delist_date").cast(pl.Date)
        ).unique()
        joined = df.join_asof(
            delists.sort("delist_date"),
            left_on="seen_date",
            right_on="delist_date",
            by="symbol",
            strategy="backward",
        )
        joined = joined.with_columns(
            (
                pl.col("delist_date").fill_null(EPOCH)
                != pl.col("delist_date").fill_null(EPOCH).shift(1).over("symbol")
            )
            .fill_null(False)  # 首列 shift 產生的 null 視為無斷點
            .alias("_ev_break")
        )
        joined = joined.with_columns(
            pl.col("_ev_break").cast(pl.Int64).cum_sum().over("symbol").alias("_seg")
        )
    else:
        joined = df.with_columns(
            pl.col("_gap_break").cast(pl.Int64).cum_sum().over("symbol").alias("_seg")
        )
    intervals = (
        joined.group_by(["symbol", "_seg"])
        .agg(
            pl.col("seen_date").min().alias("first_seen"),
            pl.col("seen_date").max().alias("last_seen"),
            pl.len().cast(pl.Int64).alias("n_days"),
            pl.col("name").sort_by("seen_date").last().alias("name"),
        )
        .sort(["symbol", "first_seen"])
        .select(list(INTERVALS_SCHEMA))
    )
    return intervals


def delisted_candidates(
    intervals: pl.DataFrame,
    active_symbols: list[str] | None = None,
    inactive_days: int = DEFAULT_INACTIVE_DAYS,
    as_of: date | None = None,
) -> pl.DataFrame:
    """下市候選：last_seen 早於 as_of - inactive_days 且 symbol 不在 active_symbols。

    as_of 預設今天；active_symbols 為 None 時不排除任何人（呼叫端自行過濾）。
    依 last_seen 降序。
    """
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=inactive_days)
    if intervals.is_empty():
        return _empty(INTERVALS_SCHEMA)
    out = intervals.filter(pl.col("last_seen") < cutoff)
    if active_symbols is not None:
        out = out.filter(~pl.col("symbol").is_in(active_symbols))
    return out.sort("last_seen", descending=True)


TERMINATED_URL = "https://www.twse.com.tw/company/suspendListingCsvAndHtml"
TERMINATED_SCHEMA = {"symbol": pl.Utf8, "name": pl.Utf8, "delist_date": pl.Date}


def parse_terminated_csv(text: str) -> pl.DataFrame:
    """解析 TWSE「終止上市公司」CSV（Big5、民國年日期）→ (symbol, name, delist_date)。

    樣本行：``"民國115年09月01日","三商壽","2867"``。
    這是**權威下市來源**：涵蓋「先停牌後下市」的標的（MI_INDEX 抽樣的盲點——
    停止買賣者不會出現在 MI_INDEX 名單），且含官方下市日期。
    """
    import re

    pat = re.compile(r'"(?:民國)?(\d{2,3})年(\d{1,2})月(\d{1,2})日","(.*?)","([0-9A-Z]+)"')
    rows = []
    for y, mo, d, name, code in pat.findall(text):
        iso = date(int(y) + 1911, int(mo), int(d)).isoformat()
        rows.append({"symbol": code, "name": name, "delist_date": iso})
    if not rows:
        return _empty(TERMINATED_SCHEMA)
    return pl.DataFrame(rows, schema_overrides={"delist_date": pl.Utf8}).with_columns(
        pl.col("delist_date").str.to_date("%Y-%m-%d")
    )


def fetch_terminated_list(
    client: "httpx.Client | None" = None, timeout: float = 60.0
) -> pl.DataFrame:
    """抓 TWSE「終止上市公司」CSV（2001 年起）並解析；網路邊界集中於此。"""
    import httpx as _httpx

    with (_httpx.Client(timeout=timeout) if client is None else client) as c:
        resp = c.get(TERMINATED_URL, params={"lang": "zh", "startYear": "", "type": "csv"})
        resp.raise_for_status()
        text = resp.content.decode("big5", errors="replace")
    return parse_terminated_csv(text)


def merge_delisted_sources(
    official: pl.DataFrame,
    sampled: pl.DataFrame,
    active_symbols: list[str] | None = None,
) -> pl.DataFrame:
    """合併官方終止上市清單與 MI_INDEX 抽樣候選 → 統一下市清單（含 source 欄）。

    - 官方清單（source=twse_terminated）：權威，含停牌後下市者；
      **剔除仍活躍的代碼**（「終止上市」含上市→上櫃轉板者，轉板股仍在交易不算下市）。
    - MI_INDEX 抽樣候選（source=mi_index_sampling）：作為官方清單的補充
      （官方清單不含下櫃轉興櫃後再終止興櫃等邊角案例）；同樣剔除活躍代碼。
    輸出欄序：symbol, name, delist_date, source；官方清單優先去重。
    """
    off = official.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("name"),
        pl.col("delist_date"),
        pl.lit("twse_terminated").alias("source"),
    )
    samp = sampled.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("name"),
        pl.lit(None, dtype=pl.Date).alias("delist_date"),
        pl.lit("mi_index_sampling").alias("source"),
    )
    merged = pl.concat([off, samp])
    if active_symbols is not None:
        merged = merged.filter(~pl.col("symbol").is_in(active_symbols))
    merged = merged.with_columns(
        pl.col("source").eq(pl.lit("twse_terminated")).alias("_priority")
    )
    merged = (
        merged.sort("_priority", descending=True)  # 官方在前 → unique keep first
        .unique(subset=["symbol"], keep="first")
        .drop("_priority")
    )
    return merged.sort(["delist_date", "symbol"], nulls_last=True, descending=False)


def _active_symbols(inst: pl.DataFrame, inactive_days: int = 30) -> list[str]:
    """instrument 表 → 現活躍代碼清單。

    TaiwanStockInfo 的 ``snapshot_date`` 是異動快照日：活躍股的資訊日日更新（≈今天），
    stale 舊列混有已下市/異動歷史 → 只認 max(snapshot_date) 近 ``inactive_days`` 日內的代碼。
    """
    if "snapshot_date" not in inst.columns:
        return inst["symbol"].unique().sort().to_list()
    cutoff = date.today() - timedelta(days=inactive_days)
    recent = (
        inst.group_by("symbol")
        .agg(pl.col("snapshot_date").max().alias("latest"))
        .filter(pl.col("latest") >= pl.lit(cutoff))
    )
    return recent["symbol"].sort().to_list()


def main(
    argv: list[str] | None = None,
    fetch_day: Callable[[str], pl.DataFrame] | None = None,
    store: ParquetStore | None = None,
    terminated_csv: str | None = None,
    terminated_client: "httpx.Client | None" = None,
) -> int:
    """CLI 入口（fetch_day/store 為測試注入點，一般使用者不需要傳）。"""
    parser = argparse.ArgumentParser(
        prog="kstock.universe_history",
        description="TWSE MI_INDEX 逐月抽樣回推歷史上市宇宙 → 下市候選清單 + (code, name, 存在區間) 對照（僅 TSE）",
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"抽樣起點 YYYY-MM-DD（預設 {DEFAULT_START}）")
    parser.add_argument("--end", default=None, help="抽樣終點 YYYY-MM-DD（預設今天）")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="每請求間隔秒數（預設 0.3，TWSE 禮貌節流）")
    parser.add_argument(
        "--output",
        default=None,
        help="輸出目錄（預設 <data>/universe/manifests）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只印每月抽樣計畫的月數與預估請求數，不發請求")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="續跑：若存在 checkpoint（tw_sample_partial.csv）則從其最後已抽月份之後繼續",
    )
    parser.add_argument(
        "--skip-terminated",
        action="store_true",
        help="跳過官方「終止上市」清單抓取（下市清單僅由 MI_INDEX 抽樣產生；一般不用）",
    )
    ns = parser.parse_args(argv)

    end = ns.end or date.today().isoformat()
    plan = _month_plan(ns.start, end)
    n_months = len(plan)
    # 預估請求數：每月起試日起逐日試到第一個交易日，無法離線預知 → 以上限（月數 × max_tries）估
    est_requests = n_months * DEFAULT_MAX_TRIES

    if ns.dry_run:
        print(
            f"[dry-run] 抽樣月數：{n_months}；"
            f"預估請求數上限：{est_requests}（月數 × max_tries={DEFAULT_MAX_TRIES}）"
        )
        return 0

    out_dir = Path(ns.output) if ns.output else settings.data_dir / "universe" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"抽樣 {ns.start} ~ {end}，共 {n_months} 個月（輸出：{out_dir}）", flush=True)

    store_obj = store if store is not None else ParquetStore()
    active: list[str] | None = None
    try:
        inst = store_obj.read_normalized("instrument")
    except Exception as e:
        inst = None
        print(f"  [警告] 讀取 instrument 表失敗（{e}）", flush=True)
    if inst is not None and inst.height > 0:
        active = _active_symbols(inst, inactive_days=30)
        print(f"  instrument 表：{len(active)} 檔現活躍股（snapshot 30 日內），將據此剔除", flush=True)
    else:
        print("警告：無 instrument 表，未剔除活躍股", flush=True)

    stamp = date.today().strftime("%Y%m%d")
    partial_path = out_dir / "tw_sample_partial.csv"
    resume_df: pl.DataFrame | None = None
    if ns.resume and partial_path.exists():
        resume_df = pl.read_csv(partial_path, schema_overrides={"symbol": pl.Utf8, "name": pl.Utf8}).with_columns(
            pl.col("seen_date").str.to_date("%Y-%m-%d")
        )
    else:
        resume_df = None

    def _checkpoint(acc: pl.DataFrame) -> None:
        acc.write_csv(partial_path)

    try:
        seen = sample_market_history(
            ns.start,
            end,
            fetch_day=fetch_day,
            sleep_seconds=ns.sleep,
            resume_df=resume_df,
            on_progress=_checkpoint,
        )
    except SampleFailure as e:
        print(f"[錯誤] {e}", file=sys.stderr)
        if partial_path.exists():
            print(
                f"進度已存 checkpoint：{partial_path}（重跑時加 --resume 從中斷處續抽）",
                file=sys.stderr,
            )
        return 1
    months_sampled = seen["seen_date"].n_unique() if not seen.is_empty() else 0

    # 官方「終止上市」清單：權威下市來源（含先停牌後下市者；MI_INDEX 的盲點）
    official = None
    if not ns.skip_terminated:
        try:
            if terminated_csv is not None:
                official = parse_terminated_csv(terminated_csv)
            else:
                official = fetch_terminated_list(client=terminated_client)
        except Exception as e:
            print(f"  [警告] 官方終止上市清單抓取失敗（{type(e).__name__}），下市清單僅含 MI_INDEX 抽樣", flush=True)
            official = None
    # 存在區間以「下市事件」為斷點（純 gap 啟發式會把停牌恢復交易誤判為代碼重用）
    intervals = build_seen_intervals(seen, terminated=official)
    sampled = delisted_candidates(intervals, active_symbols=active)
    delisted = merge_delisted_sources(official, sampled, active_symbols=active)

    seen_path = out_dir / f"tw_seen_{stamp}.csv"
    raw_path = out_dir / f"tw_seen_raw_{stamp}.csv"
    delisted_path = out_dir / f"delisted_tw_{stamp}.csv"
    seen.write_csv(raw_path)  # raw 抽樣長表：離線重建區間規則用（不需重抓）
    intervals.write_csv(seen_path)
    delisted.write_csv(delisted_path)
    if partial_path.exists():
        partial_path.unlink()  # 完成 → checkpoint 退役
    print(
        f"已寫出：{raw_path}（{seen.height} 列抽樣）、{seen_path}（{intervals.height} 列）、"
        f"{delisted_path}（{delisted.height} 列）"
    )

    print(f"統計：抽樣月數 {months_sampled}/{n_months}；區間數 {intervals.height}；下市候選 {delisted.height} 檔")
    if seen.is_empty():
        print("[錯誤] 未取得任何抽樣資料（檢查網路或日期範圍）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

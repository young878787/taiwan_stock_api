"""台股日K → RD-Agent fin_factor 資料格式（daily_pv.h5）轉換。

RD-Agent 的 ``fin_factor`` 情境預設從 qlib 下載**中國 A 股**日K（``daily_pv.h5``），
與本專案 ``data/normalized/daily`` 的台股資料無關。本模組把台股日K轉成
fin_factor 期望的格式，並透過環境變數把 ``FACTOR_CoSTEER_DATA_FOLDER``
指到台股版資料目錄（``rdagent_runner`` 已自動注入），讓因子演化直接跑台股。

目標格式（與 RD-Agent ``factor_data_template`` 一致）：

- HDF5，key=``data``
- MultiIndex ``(datetime, instrument)``，instrument 為 ``{market}{symbol}``（如 ``TSE2330``）
- 欄位 ``$open $close $high $low $volume $factor``（$volume 為股；$factor 為 yfinance
  Adj Close / Close 累積復權因子，抓不到的標的 fallback 1.0）

用法::

    uv run python -m qlab export-h5            # 正式版 + debug 子集（含復權因子）
    uv run python -m qlab export-h5 --debug-symbols 10
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from kstock.config.settings import settings as kstock_settings

PRICE_COLUMNS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]


@dataclass(frozen=True)
class ExportReport:
    path: Path
    n_rows: int
    n_symbols: int
    start: str
    end: str
    n_adjusted: int = 0  # 成功取得 yfinance 復權因子的標的數


def _load_daily() -> pl.DataFrame:
    """讀取 data/normalized/daily 全部年分 parquet。"""
    root = kstock_settings.normalized_dir / "daily"
    files = sorted(root.glob("year=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到台股日K parquet：{root}")
    return pl.read_parquet(files)


def _fetch_adjust_factors(
    symbols: list[tuple[str, str]], start: str, end: str
) -> dict[str, pd.Series]:
    """用 yfinance 逐標的計算復權因子（Adj Close / Close）。

    回傳 {instrument: factor_series_by_date}；抓不到的標的不會出現在回傳中
    （呼叫端 fallback $factor=1.0）。instrument key = ``{market}{symbol}``。
    """
    from kstock.adapters.yfinance import yf_symbol

    try:
        import yfinance as yf
    except ImportError:
        return {}

    out: dict[str, pd.Series] = {}
    for market, sym in symbols:
        ysym = yf_symbol(sym, market)
        instrument = f"{market}{sym}"
        try:
            h = yf.Ticker(ysym).history(start=start, end=end, auto_adjust=False)
            if h is None or len(h) == 0 or "Adj Close" not in h.columns:
                continue
            factor = (h["Adj Close"] / h["Close"]).dropna()
            if len(factor) == 0:
                continue
            factor.index = pd.to_datetime(factor.index).date
            out[instrument] = factor
        except Exception:
            continue
    return out


def _num(value: Any) -> float:
    """FinMind 欄位 → float（空字串/None/非數字回 0）。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0  # NaN 防禦


def _fetch_dividend_events(
    symbols: list[tuple[str, str]], start: str
) -> dict[str, list[tuple[Any, float, float]]]:
    """用 FinMind ``TaiwanStockDividend`` 抓除權息事件。

    回傳 {instrument: [(ex_date, cash_per_share, stock_ratio), ...]}：
    - ex_date：除權息交易日（現金/股票股利擇一有值即為事件日）
    - cash_per_share：每股現金股利（元）
    - stock_ratio：每股配股率（股）；FinMind 股票股利欄位以面額 10 元記載 → /10
    抓不到的標的不會出現在回傳中（呼叫端視為無事件 → factor=1.0）。
    """
    from kstock.adapters.finmind import FinMindAdapter

    adapter = FinMindAdapter()
    out: dict[str, list[tuple[Any, float, float]]] = {}
    for market, sym in symbols:
        instrument = f"{market}{sym}"
        try:
            rows = adapter.fetch_rows("TaiwanStockDividend", data_id=sym, start_date=start)
        except Exception:
            continue
        events = []
        for r in rows:
            cash = _num(r.get("CashEarningsDistribution")) + _num(r.get("CashStatutorySurplus"))
            stock = _num(r.get("StockEarningsDistribution")) + _num(r.get("StockStatutorySurplus"))
            ratio = stock / 10.0  # 面額 10 元 → 配股率
            ex = r.get("CashExDividendTradingDate") or r.get("StockExDividendTradingDate")
            if not ex:
                continue
            try:
                ex_date = datetime.strptime(ex, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            if cash <= 0 and ratio <= 0:
                continue
            events.append((ex_date, cash, ratio))
        if events:
            out[instrument] = sorted(events, key=lambda e: e[0])
        time.sleep(0.1)  # 禮貌性節流（FinMind 有頻率限制）
    return out


def _dividend_factor_column(
    pdf: pd.DataFrame, events_map: dict[str, list[tuple[Any, float, float]]]
) -> pd.Series:
    """由除權息事件計算逐列復權因子（向後調整：最新日期 factor=1）。

    除權息參考價公式：ref = (prev_close - cash) / (1 + stock_ratio)，
    單次調整因子 f = ref / prev_close；所有早於除息日的日期累積乘上 f，
    使 adj = close × factor 在除息日平滑（與 yfinance 的回調語意一致）。
    """
    factor = pd.Series(1.0, index=pdf.index, name="$factor")
    for inst, events in events_map.items():
        mask = pdf["instrument"] == inst
        sub = pdf.loc[mask].sort_values("datetime")
        dates = sub["datetime"].dt.date.to_numpy()
        closes = sub["close"].to_numpy(dtype=float)
        fac = np.ones(len(sub))
        for ex_date, cash, ratio in events:
            idx = int(np.searchsorted(dates, ex_date))
            # 前一個有效收盤（停牌 0 收盤跳過）
            j = idx - 1
            while j >= 0 and not (closes[j] > 0):
                j -= 1
            if j < 0:
                continue
            prev_close = closes[j]
            if prev_close <= cash / 10:  # 極端防禦：參考價不應為負或近零
                continue
            f = (1.0 - cash / prev_close) / (1.0 + ratio)
            f = min(max(f, 0.05), 1.0)  # 除權息只會讓參考價低於前收盤
            fac[:idx] *= f
        factor.loc[sub.index] = fac
    return factor


def to_daily_pv(
    df: pl.DataFrame,
    max_symbols: int | None = None,
    adjust: bool = True,
    adjust_source: str = "finmind",
) -> pd.DataFrame:
    """台股日K Polars DataFrame → fin_factor 期望的 pandas 格式（含復權因子）。"""
    if max_symbols is not None:
        keep = df["symbol"].unique().sort()[:max_symbols]
        df = df.filter(pl.col("symbol").is_in(keep))
    pdf = df.to_pandas()
    instrument = pdf["market"].str.cat(pdf["symbol"], sep="")
    pdf = pdf.assign(
        **{
            "$open": pdf["open"],
            "$close": pdf["close"],
            "$high": pdf["high"],
            "$low": pdf["low"],
            "$volume": pdf["volume_shares"].astype("float64"),
            "datetime": pd.to_datetime(pdf["date"]),
            "instrument": instrument,
        }
    )

    factor_col = pd.Series(1.0, index=pdf.index, name="$factor")
    n_adjusted = 0
    if adjust:
        sym_list = list(pdf[["market", "symbol"]].drop_duplicates().itertuples(index=False, name=None))
        start = pdf["datetime"].min().strftime("%Y-%m-%d")
        if adjust_source == "finmind":
            events = _fetch_dividend_events(sym_list, start)
            if events:
                factor_col = _dividend_factor_column(pdf, events)
                n_adjusted = len(events)
                print(
                    f"  復權因子（FinMind 除權息）：{n_adjusted}/{len(sym_list)} 標的有事件（其餘 factor=1.0）"
                )
                pdf["$factor"] = factor_col
                out = pdf.set_index(["datetime", "instrument"])[PRICE_COLUMNS].sort_index()
                return out
            print("  FinMind 除權息事件取得失敗，fallback 至 yfinance")
        end = (pdf["datetime"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        adj_map = _fetch_adjust_factors(sym_list, start, end)
        n_adjusted = len(adj_map)
        for mkt, sym in sym_list:
            inst = f"{mkt}{sym}"
            fs = adj_map.get(inst)
            if fs is not None:
                mask = pdf["instrument"] == inst
                dates = pdf.loc[mask, "datetime"].dt.date
                # yfinance 缺日（兩地假期不同步等）用前後已知因子帶入；
                # 直接 fillna(1.0) 會在復權序列製造單日跳洞（factor≈0.34 的標的突然回 1.0）
                mapped = dates.map(fs).ffill().bfill()
                factor_col.loc[mask] = mapped.fillna(1.0).values
        print(f"  復權因子（yfinance）：{n_adjusted}/{len(sym_list)} 標的成功取得（其餘 fallback 1.0）")

    pdf["$factor"] = factor_col
    out = pdf.set_index(["datetime", "instrument"])[PRICE_COLUMNS].sort_index()
    return out


def slice_top_symbols(
    src: Path | None = None,
    n: int = 100,
    output_dir: Path | None = None,
) -> ExportReport:
    """從既有正式版 daily_pv.h5 切出前 N 檔子集（固定 code 序，正式口徑）。

    切片規則：``sorted(instruments)[:n]``（代碼序，OTC < TSE），與
    :class:`qlab.universe.UniverseSpec` 的 ``code_first_n`` 口徑一致，也是回測
    ``--top-symbols`` 的同一宇宙語意。直接切片不重抓、不做任何計算
    → 與正式版逐列零誤差（含 $factor）。

    註：turnover（成交金額排名）口徑已於 2026-08-31 對照實驗證實失效
    （volume_change_5d 訊號在流動性前 100 失效，結論保留在 AGENTS.md），
    已移除該分支，不做功能回退。
    """
    src = src or kstock_settings.data_dir / "qlab" / "factor_source_data_tw" / "daily_pv.h5"
    if not src.exists():
        raise FileNotFoundError(f"來源 h5 不存在：{src}（先跑 `uv run python -m qlab export-h5`）")
    df = pd.read_hdf(src, key="data")
    syms = sorted(df.index.get_level_values("instrument").unique())[:n]
    out = df[df.index.get_level_values("instrument").isin(syms)].sort_index()
    target = output_dir or kstock_settings.data_dir / "qlab" / f"factor_source_data_tw{n}"
    target.mkdir(parents=True, exist_ok=True)
    out.to_hdf(target / "daily_pv.h5", key="data")
    # _README_TW 含 {market} 等字面大括號，不能用 .format() → 以字串串接附註
    note = (
        f"\n- 宇宙：依 instrument 代碼排序的前 {len(syms)} 檔（正式口徑，"
        f"與 `qlab backtest --top-symbols {n}` 相同口徑）。\n"
    )
    (target / "README.md").write_text(_README_TW + note, encoding="utf-8")
    dts = out.index.get_level_values("datetime")
    return ExportReport(
        path=target / "daily_pv.h5",
        n_rows=len(out),
        n_symbols=out.index.get_level_values("instrument").nunique(),
        start=str(dts.min().date()),
        end=str(dts.max().date()),
    )


def export_daily_pv(
    output_dir: Path | None = None,
    debug_dir: Path | None = None,
    debug_symbols: int = 20,
    adjust: bool = True,
    adjust_source: str = "finmind",
) -> tuple[ExportReport, ExportReport]:
    """產出台股版 daily_pv.h5（正式版 + debug 子集）與資料說明 README.md。

    ``adjust_source``：finmind（預設，用 TaiwanStockDividend 自算，除息事件覆蓋完整）
    或 yfinance（Adj Close/Close，缺事件時 adj 會跳動）。
    """
    df = _load_daily()
    base = output_dir or kstock_settings.data_dir / "qlab" / "factor_source_data_tw"
    dbg = debug_dir or kstock_settings.data_dir / "qlab" / "factor_source_data_tw_debug"

    reports: list[ExportReport] = []
    for target, max_symbols in ((base, None), (dbg, debug_symbols)):
        pv = to_daily_pv(df, max_symbols=max_symbols, adjust=adjust, adjust_source=adjust_source)
        target.mkdir(parents=True, exist_ok=True)
        pv.to_hdf(target / "daily_pv.h5", key="data")
        (target / "README.md").write_text(_README_TW, encoding="utf-8")
        dts = pv.index.get_level_values("datetime")
        n_adj = int((pv["$factor"] != 1.0).groupby(pv.index.get_level_values("instrument")).any().sum())
        reports.append(
            ExportReport(
                path=target / "daily_pv.h5",
                n_rows=len(pv),
                n_symbols=pv.index.get_level_values("instrument").nunique(),
                start=str(dts.min().date()),
                end=str(dts.max().date()),
                n_adjusted=n_adj,
            )
        )
    return reports[0], reports[1]


_README_TW = """# 台股日K資料（daily_pv.h5）

由 kstock `data/normalized/daily`（FinMind + TWSE）轉換而來，取代 RD-Agent 預設的 qlib 中國 A 股資料。

- index：`(datetime, instrument)`；instrument 為 `{market}{symbol}`（TSE=上市、OTC=上櫃），例如 `TSE2330`。
- `$open/$close/$high/$low`：新台幣元，**未復權原始價**（FinMind 原始 OHLC）。
- `$factor`：累積復權因子（= yfinance Adj Close / Close），復權價 = 原始價 × $factor。
  跨除權息日計算報酬時請用 `$close × $factor`（後復權）。
- `$volume`：成交股數。
"""

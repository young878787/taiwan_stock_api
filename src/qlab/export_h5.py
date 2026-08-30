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

from dataclasses import dataclass
from pathlib import Path

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


def to_daily_pv(df: pl.DataFrame, max_symbols: int | None = None, adjust: bool = True) -> pd.DataFrame:
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
        print(f"  復權因子：{n_adjusted}/{len(sym_list)} 標的成功取得（其餘 fallback 1.0）")

    pdf["$factor"] = factor_col
    out = pdf.set_index(["datetime", "instrument"])[PRICE_COLUMNS].sort_index()
    return out


def export_daily_pv(
    output_dir: Path | None = None,
    debug_dir: Path | None = None,
    debug_symbols: int = 20,
    adjust: bool = True,
) -> tuple[ExportReport, ExportReport]:
    """產出台股版 daily_pv.h5（正式版 + debug 子集）與資料說明 README.md。"""
    df = _load_daily()
    base = output_dir or kstock_settings.data_dir / "qlab" / "factor_source_data_tw"
    dbg = debug_dir or kstock_settings.data_dir / "qlab" / "factor_source_data_tw_debug"

    reports: list[ExportReport] = []
    for target, max_symbols in ((base, None), (dbg, debug_symbols)):
        pv = to_daily_pv(df, max_symbols=max_symbols, adjust=adjust)
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

"""大盤基準：臺灣加權指數日報酬（FinMind TAIEX）。

偏好**發行量加權報酬指數**（``TaiwanStockTotalReturnIndex``，含息）——與策略的
復權含息口徑對齊；抓不到時 fallback **加權指數**（``TaiwanStockPrice``，未含息）。
抓取結果快取成 parquet（``data/qlab/benchmark_taiex.parquet``），已涵蓋請求區間時
離線重跑不會再打 API；抓取失敗但快取有資料時，回傳快取可涵蓋的部分。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kstock.config.settings import settings as kstock_settings

TRI_LABEL = "臺灣大盤 TAIEX 報酬指數（含息）"
PRICE_LABEL = "臺灣大盤 TAIEX 加權指數（未含息）"

_TRI_DATASET = "TaiwanStockTotalReturnIndex"
_PRICE_DATASET = "TaiwanStockPrice"


def _cache_path() -> Path:
    return kstock_settings.data_dir / "qlab" / "benchmark_taiex.parquet"


def _fetch_series(dataset: str, start: str) -> pd.Series | None:
    """抓 FinMind 指數日序列（TRI=price 欄、加權指數=close 欄）→ pd.Series（index=date）。"""
    from kstock.adapters.finmind import FinMindAdapter

    try:
        rows = FinMindAdapter().fetch_rows(dataset, data_id="TAIEX", start_date=start)
    except Exception:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    col = "price" if dataset == _TRI_DATASET else "close"
    if col not in df.columns:
        return None
    s = pd.Series(
        pd.to_numeric(df[col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["date"]),
        name="close",
    ).dropna()
    return s[s > 0]


def load_taiex_daily_returns(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str] | None:
    """回傳 (大盤日報酬 Series, 口徑標籤)；無資料時回 None。

    - 優先含息報酬指數；失敗改抓加權指數並在標籤註明「未含息」。
    - 快取 parquet 已涵蓋 [start, end] 時直接用快取（離線可重跑）。
    """
    cache_file = _cache_path()
    cache: pd.DataFrame | None = None
    if cache_file.exists():
        try:
            cache = pd.read_parquet(cache_file)
        except Exception:
            cache = None
    covered = (
        cache is not None
        and len(cache) > 0
        and cache.index.min() <= start
        and cache.index.max() >= end
    )
    if not covered:
        first = cache.index.min() if cache is not None and len(cache) else start
        for dataset, label in ((_TRI_DATASET, TRI_LABEL), (_PRICE_DATASET, PRICE_LABEL)):
            s = _fetch_series(dataset, first.strftime("%Y-%m-%d"))
            if s is None or s.empty:
                continue
            merged = s.to_frame() if cache is None or not len(cache) else pd.concat([cache, s.to_frame()])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged["source"] = label
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                merged.to_parquet(cache_file)
            except Exception:
                pass
            cache = merged
            break
    if cache is None or not len(cache):
        return None
    source = str(cache["source"].iloc[-1]) if "source" in cache.columns else TRI_LABEL
    sub = cache["close"].loc[(cache.index >= start) & (cache.index <= end)].sort_index()
    returns = sub.pct_change().dropna()
    if returns.empty:
        return None
    return returns, source

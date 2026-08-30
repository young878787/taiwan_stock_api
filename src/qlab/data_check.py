"""daily_pv.h5 ↔ kstock 台股日K parquet 資料庫交叉驗證。

回測用的 daily_pv.h5 是 `export-h5` 從 ``data/normalized/daily``（kstock 台股
資料庫，FinMind/TWSE/yfinance 三來源 normalized 後的 parquet）轉出的。本模組
逐項確認：

1. **來源一致**：h5 的標的集合 ⊆ parquet 資料庫；日期範圍一致。
2. **排列正確**：MultiIndex 單調遞增、無重複 (datetime, instrument)、
   每標的日期與 parquet 完全對齊（不缺不重）。
3. **數值正確**：全量比對 $close vs parquet `close`、$volume vs `volume_shares`；
   復權價（$close × $factor）日變動幅度合理性檢查。
4. **壞資料**：close ≤ 0、NaN/inf 統計。

用法::

    uv run python -m qlab verify-data                       # 驗證正式版 h5
    uv run python -m qlab verify-data --h5 <path>           # 驗證指定 h5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from kstock.config.settings import settings as kstock_settings

DEFAULT_H5 = kstock_settings.data_dir / "qlab" / "factor_source_data_tw" / "daily_pv.h5"


def _load_parquet_daily() -> pl.DataFrame:
    files = sorted((kstock_settings.normalized_dir / "daily").glob("year=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到台股日K parquet：{kstock_settings.normalized_dir / 'daily'}")
    return pl.read_parquet(files)


def verify_daily_pv(h5_path: Path | None = None) -> str:
    """執行交叉驗證，回傳 Markdown 報告字串（✅/❌ 逐項）。"""
    h5_path = Path(h5_path or DEFAULT_H5)
    pv = pd.read_hdf(h5_path, key="data")
    idx = pv.index
    lines = [
        "# daily_pv.h5 資料來源與品質驗證",
        "",
        f"- h5：`{h5_path}`（{len(pv):,} 列）",
        f"- parquet 資料庫：`{kstock_settings.normalized_dir / 'daily'}`（kstock 台股日K，入庫來源 FinMind/TWSE/yfinance）",
        "",
        "| # | 檢查項 | 結果 |",
        "|---|---|---|",
    ]
    results: list[tuple[str, str, str]] = []

    def _add(name: str, ok: bool, detail: str) -> None:
        results.append((name, "✅" if ok else "❌", detail))

    results_warn_only: set[str] = {"close 無 0/負值"}  # 上游停牌資料 → 已知限制而非管線錯誤

    # ── 1. 排列檢查 ──
    _add("MultiIndex 單調遞增（datetime→instrument 排列）", bool(idx.is_monotonic_increasing), "")
    dup = int(idx.duplicated().sum())
    _add("無重複 (datetime, instrument)", dup == 0, f"重複 {dup} 組")
    per_dup = int(pv.reset_index().duplicated(["instrument", "datetime"]).sum())
    _add("無重複 (instrument, datetime)", per_dup == 0, f"重複 {per_dup} 組")
    dt_min, dt_max = idx.get_level_values("datetime").min(), idx.get_level_values("datetime").max()
    _add("日期範圍", True, f"{dt_min:%Y-%m-%d} ~ {dt_max:%Y-%m-%d}")

    # ── 2. 與 parquet 資料庫對帳 ──
    daily = _load_parquet_daily()
    daily_pd = daily.with_columns(
        (pl.col("market") + pl.col("symbol")).alias("instrument")
    ).to_pandas()
    db = daily_pd[["instrument", "date", "close", "volume_shares"]].copy()
    db["date"] = pd.to_datetime(db["date"])

    h5_syms = set(idx.get_level_values("instrument"))
    db_syms = set(db["instrument"])
    missing_in_db = h5_syms - db_syms
    _add(
        "h5 標的 ⊆ parquet 資料庫（來源一致）",
        not missing_in_db,
        f"h5 {len(h5_syms)} 檔；資料庫 {len(db_syms)} 檔；缺失 {len(missing_in_db)}"
        + (f"：{sorted(missing_in_db)[:5]}" if missing_in_db else ""),
    )

    db_dates = set(db["date"].dt.normalize())
    h5_dates = set(idx.get_level_values("datetime").normalize())
    extra_dates = h5_dates - db_dates
    _add(
        "h5 交易日 ⊆ parquet 交易日",
        not extra_dates,
        f"h5 {len(h5_dates)} 個交易日；多出 {len(extra_dates)} 日"
        + (f"：{sorted(extra_dates)[:3]}" if extra_dates else ""),
    )

    # 全量 (instrument, date) merge 比對數值
    pv_flat = pv.reset_index()
    pv_flat["date"] = pd.to_datetime(pv_flat["datetime"]).dt.normalize()
    merged = pv_flat.merge(db, on=["instrument", "date"], how="left", validate="one_to_one")
    n_unmatched = int(merged["close"].isna().sum())
    _add("每列都能對應到 parquet 一列（不缺列）", n_unmatched == 0, f"未對應 {n_unmatched:,} 列")

    close_diff = (merged["$close"] - merged["close"]).abs()
    close_ok = int((close_diff > 1e-9).sum())
    _add(
        f"$close ≡ parquet close（全量 {len(merged):,} 列）",
        close_ok == 0,
        f"不一致 {close_ok} 列（最大差 {close_diff.max():.6g}）",
    )
    vol_diff = (merged["$volume"] - merged["volume_shares"]).abs()
    vol_ok = int((vol_diff > 0.5).sum())
    _add(
        f"$volume ≡ parquet volume_shares（股）",
        vol_ok == 0,
        f"不一致 {vol_ok} 列（最大差 {vol_diff.max():.4g}）",
    )

    # ── 3. 壞資料與復權合理性 ──
    n_zero = int((merged["$close"] <= 0).sum())
    if n_zero == 0:
        _add("close 無 0/負值", True, "")
    else:
        # 0 收盤若 parquet 來源即有（停牌日记 0）→ 上游已知限制（h5 忠實反映），非轉檔錯誤
        same_as_source = bool((merged.loc[merged["$close"] <= 0, "close"] <= 0).all())
        _add(
            "close 無 0/負值",
            False,
            f"≤0 共 {n_zero} 列（"
            + ("來源 parquet 同位置亦為 0 → 上游停牌資料；回測端視為缺價防禦" if same_as_source else "⚠️ 僅 h5 有、parquet 無 → 轉檔異常")
            + "）",
        )
    n_nan = int(merged["$close"].isna().sum())
    _add("close 無 NaN/inf", n_nan == 0, f"NaN/inf 共 {n_nan} 列")
    pv_sorted = pv.sort_index()
    # 孤立 1.0（真洞）：同標的前後日 factor 都明顯 <1，唯獨中間某日 =1.0。
    # 注意「除息後 factor 回 1.0」是正常現象（最新日期 Adj=Close），不可誤判。
    grp = pv_sorted.groupby(level="instrument")["$factor"]
    prev_f, next_f = grp.shift(1), grp.shift(-1)
    isolated = int(
        ((pv_sorted["$factor"] >= 0.99) & (prev_f < 0.9) & (next_f < 0.9)).sum()
    )
    _add(
        "無 factor 缺日洞（孤立 1.0）",
        isolated == 0,
        f"{isolated} 列（前後日因子都 <0.9 卻單日 =1.0）——缺日 fallback 所致，會讓復權價單日假跳",
    )

    # 復權價日跳動：台股漲跌停 10%，>60% 必為異常。
    # 先排除「停牌 0 收盤」相鄰的跳動（上游停牌資料，見 close≤0 檢查項），
    # 剩餘者為除權息未回調等真異常（FinMind 事件法下應為 0）。
    close_s = pv_sorted["$close"]
    adj_close = close_s * pv_sorted["$factor"]
    adj_ret = adj_close.groupby(level="instrument").pct_change(fill_method=None)
    prev_close = close_s.groupby(level="instrument").shift(1)
    next_close = close_s.groupby(level="instrument").shift(-1)
    zero_adjacent = (close_s <= 0) | (prev_close <= 0) | (next_close <= 0)
    big_mask = (adj_ret.abs() > 0.6) & ~zero_adjacent.reindex(adj_ret.index).fillna(False)
    big = int(big_mask.sum())
    n_syms = int(big_mask.groupby(level="instrument").any().sum())
    n_zero_jumps = int(((adj_ret.abs() > 0.6) & zero_adjacent.reindex(adj_ret.index).fillna(True)).sum())
    status = "✅" if big == 0 else ("⚠️" if big < len(pv_sorted) * 0.002 else "❌")
    results.append(
        (
            "復權價非停牌日跳動 >60%（除權息未回調等）",
            status,
            f"{big} 次 / {n_syms} 檔（另有 {n_zero_jumps} 次停牌 0 收盤造成的跳動，歸上游資料）；"
            "殘留者通常為減資/股票分割/面額變更（FinMind TaiwanStockCapitalReductionReferencePrice / ParValueChange 可再涵蓋）",
        )
    )
    f_min, f_max = float(pv["$factor"].min()), float(pv["$factor"].max())
    _add(
        "$factor 範圍合理",
        0 < f_min and f_max < 50,
        f"min={f_min:.4g}, max={f_max:.4g}（1.0=無除權息；跳變=復權調整）",
    )
    # 上游已知限制（h5 忠實反映來源）改標 ⚠️，不計入失敗
    for j, (name, status, detail) in enumerate(results):
        if name in results_warn_only and status == "❌" and "轉檔異常" not in detail:
            results[j] = (name, "⚠️", detail)

    for i, (name, status, detail) in enumerate(results, 1):
        lines.append(f"| {i} | {name} | {status} {detail} |")

    n_fail = sum(1 for _, status, _ in results if status == "❌")
    n_warn = sum(1 for _, status, _ in results if status == "⚠️")
    summary = (
        "全部通過"
        if n_fail == 0 and n_warn == 0
        else f"通過，{n_warn} 項已知限制" if n_fail == 0
        else f"{n_fail} 項未通過，需複查"
    )
    lines += ["", f"**結論：{summary}**（{len(results)} 項檢查；⚠️ = 已知限制、不影響回測防禦）", ""]
    return "\n".join(lines)

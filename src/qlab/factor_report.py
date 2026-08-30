"""fin_factor 因子結果報告匯出（含 IC 評估）。

掃描 RD-Agent 因子工作區（``<data>/qlab/rdagent_workspace/git_ignore_folder/RD-Agent_workspace``
）中每個因子目錄的 ``result.h5``（因子值）與 ``factor.py``（實作），
並對**前瞻報酬**計算日截面 Rank IC，產出 Markdown 報告到
``<data>/qlab/factor_report.md``。

用法::

    uv run python -m qlab factor-report
    uv run python -m qlab factor-report --fwd-days 5   # 改前瞻天數
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pytables 的損壞/空檔錯誤（HDF5ExtError 不是 OSError 子類）
    from tables.exceptions import HDF5ExtError as _HdfError
except ImportError:  # pragma: no cover
    _HdfError = OSError  # type: ignore[assignment]

from kstock.config.settings import settings as kstock_settings

WORKSPACE_DIRNAME = "RD-Agent_workspace"
REPORT_FILENAME = "factor_report.md"

# 因子值所依據的 daily_pv.h5 候選路徑（$close 用來算前瞻報酬；台股優先、A 股最後）
_DATA_DIRS = (
    kstock_settings.data_dir / "qlab" / "factor_source_data_tw_debug" / "daily_pv.h5",
    kstock_settings.data_dir / "qlab" / "factor_source_data_tw" / "daily_pv.h5",
    kstock_settings.data_dir / "qlab" / "rdagent_workspace" / "git_ignore_folder"
    / "factor_implementation_source_data_debug" / "daily_pv.h5",
    kstock_settings.data_dir / "qlab" / "rdagent_workspace" / "git_ignore_folder"
    / "factor_implementation_source_data" / "daily_pv.h5",
)


@dataclass(frozen=True)
class FactorResult:
    directory: str
    factor_name: str
    n_rows: int
    n_instruments: int
    start: str
    end: str
    stats: dict[str, float]
    sample: str
    success: bool
    wide: pd.DataFrame = field(default=None, repr=False, compare=False)  # type: ignore[assignment]


def _load_price_data() -> pd.DataFrame | None:
    """載入因子執行所依據的 daily_pv.h5（台股 debug → 台股正式 → 內建 A 股）。"""
    for p in _DATA_DIRS:
        if p.exists():
            try:
                return pd.read_hdf(p, key="data")
            except (KeyError, ValueError):
                continue
    return None


def _load_factor_dir(factor_dir: Path) -> FactorResult | None:
    """讀取單一因子目錄；沒有 result.h5（未執行成功）則回傳 None。"""
    result_file = factor_dir / "result.h5"
    if not result_file.exists():
        return None
    df = None
    for key in ("data", "factor"):
        try:
            df = pd.read_hdf(result_file, key=key)
            break
        except (KeyError, ValueError, OSError, _HdfError):
            continue
    if df is None:
        return None
    if isinstance(df, pd.Series):
        # ling 生成的程式碼可能以 Series（帶 name）儲存，統一轉 DataFrame
        df = df.to_frame(name=df.name or "factor")
    col = df.columns[0]
    values = df[col].dropna()
    # MultiIndex (datetime, instrument)
    instruments = df.index.get_level_values("instrument").nunique()
    dmin = df.index.get_level_values("datetime").min()
    dmax = df.index.get_level_values("datetime").max()
    wide = df[col].unstack("instrument")
    return FactorResult(
        directory=factor_dir.name,
        factor_name=str(col),
        n_rows=len(df),
        n_instruments=int(instruments),
        start=str(dmin.date()) if hasattr(dmin, "date") else str(dmin),
        end=str(dmax.date()) if hasattr(dmax, "date") else str(dmax),
        stats={
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "na_ratio": float(df[col].isna().mean()),
        },
        sample=values.tail(3).to_string(),
        success=True,
        wide=wide,
    )


def daily_rank_ic(
    factor_wide: pd.DataFrame,
    fwd_ret_wide: pd.DataFrame,
    min_symbols: int = 5,
) -> pd.Series:
    """逐日截面 Spearman Rank IC（因子值 vs 前瞻報酬），回傳 IC 時間序列。"""
    common = factor_wide.index.intersection(fwd_ret_wide.index)
    ics: list[float] = []
    dates: list = []
    for dt in common:
        f_row = factor_wide.loc[dt].dropna()
        r_row = fwd_ret_wide.loc[dt].dropna()
        syms = f_row.index.intersection(r_row.index)
        if len(syms) < min_symbols:
            continue
        ic = f_row[syms].rank().corr(r_row[syms].rank())
        if not np.isnan(ic):
            ics.append(float(ic))
            dates.append(dt)
    return pd.Series(ics, index=dates, name="rank_ic")


def forward_returns(price_df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """由 daily_pv（$close 欄、MultiIndex）計算 N 日前瞻報酬（wide 格式）。

    壞資料防禦：close <= 0（停牌誤植等）視為缺價，避免產生 inf/爆量前瞻報酬
    毒化 Rank IC（與回測端的 close_wide.mask 口徑一致）。
    """
    close = price_df["$close"].unstack("instrument")
    close = close.mask(close <= 0.0)
    return close.shift(-days) / close - 1


def evaluate_ics(
    results: list[FactorResult],
    price_df: pd.DataFrame | None,
    fwd_days: int = 5,
) -> dict[str, dict[str, float]]:
    """對每個因子計算 IC 統計（對 N 日前瞻報酬）。"""
    if price_df is None:
        return {}
    fwd = forward_returns(price_df, days=fwd_days)
    out: dict[str, dict[str, float]] = {}
    for r in results:
        if r.wide is None:
            continue
        s = daily_rank_ic(r.wide, fwd)
        if len(s) < 10 or s.std() == 0:
            continue
        ic_mean, ic_std = s.mean(), s.std()
        out[r.factor_name] = {
            "ic_mean": float(ic_mean),
            "icir": float(ic_mean / ic_std) if ic_std else 0.0,
            "ic_pos_ratio": float((s > 0).mean()),
            "t_stat": float(ic_mean / (ic_std / np.sqrt(len(s)))) if ic_std else 0.0,
            "n_days": len(s),
        }
    return out


def collect_results(workspace: Path | None = None) -> list[FactorResult]:
    """收集所有已成功產出 result.h5 的因子（依工作區目錄排序）。"""
    workspace = workspace or (
        kstock_settings.data_dir / "qlab" / "rdagent_workspace" / "git_ignore_folder" / WORKSPACE_DIRNAME
    )
    if not workspace.exists():
        return []
    results = []
    for factor_dir in sorted(p for p in workspace.iterdir() if p.is_dir()):
        r = _load_factor_dir(factor_dir)
        if r is not None:
            results.append(r)
    return results


def build_report(
    results: list[FactorResult],
    n_total: int | None = None,
    ics: dict[str, dict[str, float]] | None = None,
    fwd_days: int = 5,
) -> str:
    """組裝 Markdown 報告。``n_total`` 為演化迴圈中的因子總數（可選）。"""
    lines = [
        "# RD-Agent fin_factor 因子結果報告",
        "",
        f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 成功產出因子值：**{len(results)}** 個" + (f" / 演化進度 {n_total} 個" if n_total else ""),
        f"- 模型：`{kstock_settings.openrouter_model}`（instructor MD_JSON 結構化輸出）",
        "",
        "## 因子摘要",
        "",
        "| # | 因子名稱 | 列數 | 標的數 | 區間 | mean | std | NA 佔比 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | `{r.factor_name}` | {r.n_rows:,} | {r.n_instruments} "
            f"| {r.start} ~ {r.end} | {r.stats['mean']:.4g} | {r.stats['std']:.4g} "
            f"| {r.stats['na_ratio']:.1%} |"
        )

    if ics:
        lines += [
            "",
            f"## IC 評估（對 {fwd_days} 日前瞻報酬的日截面 Rank IC）",
            "",
            "判讀：|IC| > 0.03 且 |t| > 2 → 有初步預測力；IC < 0 表示反向（反轉）訊號。",
            "",
            "| 因子 | IC 均值 | ICIR | IC>0 比例 | t-stat | 樣本日數 |",
            "|---|---|---|---|---|---|",
        ]
        for name, s in ics.items():
            ic_str = f"**{s['ic_mean']:.4f}**" if abs(s["ic_mean"]) > 0.01 else f"{s['ic_mean']:.4f}"
            sig = " 🔥" if abs(s["t_stat"]) > 2 else ""
            lines.append(
                f"| `{name}`{sig} | {ic_str} | {s['icir']:.3f} "
                f"| {s['ic_pos_ratio']:.1%} | {s['t_stat']:.2f} | {s['n_days']} |"
            )
        skipped = [r.factor_name for r in results if r.factor_name not in ics]
        if skipped:
            lines += ["", f"未評估（樣本不足）：{', '.join(f'`{n}`' for n in skipped)}"]

    lines += ["", "## 各因子明細", ""]
    for i, r in enumerate(results, 1):
        ic_line = "- IC：未評估（價格資料不足）"
        if ics and r.factor_name in ics:
            s = ics[r.factor_name]
            ic_line = (
                f"- IC：mean={s['ic_mean']:.4f}、ICIR={s['icir']:.3f}、"
                f"t={s['t_stat']:.2f}（{fwd_days} 日前瞻）"
            )
        lines += [
            f"### {i}. `{r.factor_name}`（工作區 `{r.directory[:12]}…`）",
            "",
            f"- 資料範圍：{r.start} ~ {r.end}，{r.n_instruments} 個標的，{r.n_rows:,} 列",
            f"- 統計：mean={r.stats['mean']:.4g}、std={r.stats['std']:.4g}、"
            f"min={r.stats['min']:.4g}、max={r.stats['max']:.4g}、NA={r.stats['na_ratio']:.1%}",
            ic_line,
            "",
            "```",
            r.sample,
            "```",
            "",
        ]
    return "\n".join(lines)


def export_report(
    workspace: Path | None = None,
    output: Path | None = None,
    fwd_days: int = 5,
) -> Path:
    """產出報告檔，回傳報告路徑。"""
    results = collect_results(workspace)
    price_df = _load_price_data()
    ics = evaluate_ics(results, price_df, fwd_days=fwd_days)
    report = build_report(results, ics=ics, fwd_days=fwd_days)
    output = output or kstock_settings.data_dir / "qlab" / REPORT_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output

"""fin_factor 因子結果報告匯出。

掃描 RD-Agent 因子工作區（``<data>/qlab/rdagent_workspace/git_ignore_folder/RD-Agent_workspace``
）中每個因子目錄的 ``result.h5``（因子值）與 ``factor.py``（實作），
產出 Markdown 報告到 ``<data>/qlab/factor_report.md``。

用法::

    uv run python -m qlab factor-report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from kstock.config.settings import settings as kstock_settings

WORKSPACE_DIRNAME = "RD-Agent_workspace"
REPORT_FILENAME = "factor_report.md"


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
        except (KeyError, ValueError):
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
    )


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


def build_report(results: list[FactorResult], n_total: int | None = None) -> str:
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
    lines += ["", "## 各因子明細", ""]
    for i, r in enumerate(results, 1):
        lines += [
            f"### {i}. `{r.factor_name}`（工作區 `{r.directory[:12]}…`）",
            "",
            f"- 資料範圍：{r.start} ~ {r.end}，{r.n_instruments} 個標的，{r.n_rows:,} 列",
            f"- 統計：mean={r.stats['mean']:.4g}、std={r.stats['std']:.4g}、"
            f"min={r.stats['min']:.4g}、max={r.stats['max']:.4g}、NA={r.stats['na_ratio']:.1%}",
            "",
            "```",
            r.sample,
            "```",
            "",
        ]
    return "\n".join(lines)


def export_report(workspace: Path | None = None, output: Path | None = None) -> Path:
    """產出報告檔，回傳報告路徑。"""
    results = collect_results(workspace)
    report = build_report(results)
    output = output or kstock_settings.data_dir / "qlab" / REPORT_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output

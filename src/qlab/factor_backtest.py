"""fin_factor 因子 → 台股橫截面投組回測（result.h5 → 訊號 → kstock 回測引擎）。

流程：

1. 從 RD-Agent 工作區收集 ``result.h5``（沿用 ``qlab.factor_report`` 的載入邏輯）。
2. 每隔 ``rebalance_days`` 個交易日，依因子值橫截面排名選出前/後 ``top_n`` 檔
   （IC < 0 的反轉因子用「買因子最小者」），建立每日持倉訊號。
3. 逐標的呼叫 :func:`kstock.backtest.engine.run_backtest`（訊號 t 日產生、t+1 日
   開始計損益，引擎本身已處理無前視偏差），再以 1/N slot 等權聚合為投組日報酬。
4. 台股成本（純現金）：買入手續費 0.1425% + 賣出手續費 0.1425% + 證交稅 0.3%（借貸成本不計）；
   引擎單邊 fee_rate 取 ``commission + tax/2``，來回恰好 = 2×commission + tax。
5. 產出 Markdown 報告到 ``<data>/qlab/backtest_report.md``。

**評估準則：一律以含成本（淨）績效判斷策略可否交易**；免成本（所有成本設 0）
結果僅用於驗證訊號本身有無 alpha。

用法::

    uv run python -m qlab backtest                                   # 預設兩因子、20 日再平衡、含成本
    uv run python -m qlab backtest --factors momentum_5d --top-n 3
    uv run python -m qlab backtest --factors ma_deviation_20d,volume_change_5d --direction bottom
    uv run python -m qlab backtest --gross                           # 免成本（僅訊號驗證）

注意：目前因子跑在 20 檔 debug 子集上，回測結果僅供管線驗證；
正式評估需以全市場資料重跑因子。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kstock.backtest.engine import TRADING_DAYS, run_backtest
from kstock.config.settings import settings as kstock_settings

from qlab.factor_report import _load_price_data, collect_results, daily_rank_ic, forward_returns

BACKTEST_REPORT_FILENAME = "backtest_report.md"

DEFAULT_FACTORS = ("ma_deviation_20d", "volume_change_5d")

DEFAULT_DATA_H5 = kstock_settings.data_dir / "qlab" / "factor_source_data_tw" / "daily_pv.h5"


def _factor_python_bin() -> str:
    """執行 LLM 因子程式碼的 python（可用 env FACTOR_PYTHON_BIN 覆寫）。"""
    return os.environ.get("FACTOR_PYTHON_BIN") or sys.executable


@dataclass(frozen=True)
class PortfolioResult:
    """單一因子投組的回測結果。"""

    factor_name: str
    direction: str  # top=買因子最大者、bottom=買因子最小者
    total_return: float
    annual_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    avg_turnover: float  # 每次再平衡更換 slot 的比例（0~1）
    win_rate: float  # 持倉日中正報酬比例
    daily: pd.Series = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    equity: pd.Series = field(default=None, repr=False, compare=False)  # type: ignore[assignment]


def auto_direction(
    factor_wide: pd.DataFrame,
    price_df: pd.DataFrame,
    fwd_days: int = 5,
) -> tuple[str, float]:
    """以全樣本 Rank IC 決定買方方向（IC >= 0 買大、否則買小）。

    注意：這是 in-sample 判斷，僅方便快速驗證；嚴謹做法應以樣本外 IC 決定。
    """
    fwd = forward_returns(price_df, days=fwd_days)
    ic = daily_rank_ic(factor_wide, fwd)
    if len(ic) == 0 or np.isnan(ic.mean()):
        return "bottom", float("nan")
    return ("top", float(ic.mean())) if ic.mean() >= 0 else ("bottom", float(ic.mean()))


def build_signals(
    factor_wide: pd.DataFrame,
    top_n: int,
    rebalance_days: int,
    direction: str,
    buffer_n: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """由因子值建立每日持倉訊號（1=持有多方 slot、0=空手）。

    再平衡日 = 因子索引每隔 ``rebalance_days`` 列；訊號在再平衡日收盤後產生，
    由回測引擎延後一天生效，因此此處不需再 shift。

    緩衝帶（``buffer_n`` > ``top_n`` 時）：既有持倉只要排名仍在 buffer_n 內就續抱，
    跌出才賣；空出的 slot 由排名最佳的非持倉補上（同樣限 buffer_n 內）。
    目的：降低換倉比例、壓低台股雙邊交易成本。
    回傳 (訊號表, 平均每次再平衡換倉比例)。
    """
    if direction not in ("top", "bottom"):
        raise ValueError(f"direction 必須是 top/bottom，收到 {direction!r}")
    band = buffer_n or top_n
    if band < top_n:
        raise ValueError(f"buffer_n（{band}）不可小於 top_n（{top_n}）")
    ascending = direction == "bottom"  # bottom=買最小者 → rank 1 = 最小
    sig = pd.DataFrame(0.0, index=factor_wide.index, columns=factor_wide.columns)
    turnovers: list[float] = []
    prev_picks: set | None = None
    row_pos = list(range(0, len(factor_wide.index), rebalance_days))
    for k, start in enumerate(row_pos):
        end = row_pos[k + 1] if k + 1 < len(row_pos) else len(factor_wide.index)
        row = factor_wide.iloc[start].dropna()
        if row.empty:
            prev_picks = set()
            continue
        ranks = row.rank(ascending=ascending)
        if prev_picks is None:
            picks = set(ranks.nsmallest(top_n).index)
        else:
            keep = {s for s in prev_picks & set(ranks.index) if ranks[s] <= band}
            need = top_n - len(keep)
            fresh = ranks.drop(index=[s for s in keep if s in ranks.index]).nsmallest(need).index
            picks = keep | set(fresh)
            picks = {s for s in picks if ranks.get(s, np.inf) <= band}
        sig.iloc[start:end, sig.columns.get_indexer(sorted(picks))] = 1.0
        if prev_picks is not None:
            overlap = len(prev_picks & picks)
            turnovers.append(1.0 - overlap / top_n)
        prev_picks = picks
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    return sig, avg_turnover


def _metrics(daily: pd.Series) -> dict[str, float]:
    """投組日報酬 → 績效指標（沿用引擎的年化/夏普慣例：252 交易日）。"""
    daily = daily.fillna(0.0)
    equity = (1.0 + daily).cumprod()
    n = len(daily)
    total = float(equity.iloc[-1] - 1.0) if n else 0.0
    annual = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if n > 2 else 0.0
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 and daily.std(ddof=1) else 0.0
    peak = equity.cummax()
    mdd = float((1.0 - equity / peak).max()) if n else 0.0
    active = daily[daily != 0.0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    return {
        "total_return": total,
        "annual_return": annual,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
    }


def portfolio_daily_returns(
    signals: pd.DataFrame,
    close_wide: pd.DataFrame,
    top_n: int,
    fee_rate: float,
    slippage_rate: float = 0.0,
) -> tuple[pd.Series, int]:
    """逐標的跑 :func:`run_backtest`，等權聚合為投組日報酬。

    每個 slot 資金 1/``top_n``；未滿 ``top_n`` 檔時剩餘 slot 視為現金（報酬 0）。
    停牌等造成的盤中缺口以「壓縮序列」處理（該標的缺日不計損益）。
    回傳 (投組日報酬, 總進出場次數)。
    """
    common_cols = [c for c in signals.columns if c in close_wide.columns]
    rets = pd.DataFrame(0.0, index=close_wide.index, columns=common_cols)
    total_trades = 0
    for sym in common_cols:
        close_s = close_wide[sym].dropna()
        if len(close_s) < 3:
            continue
        sig_s = signals[sym].reindex(close_s.index).fillna(0.0).astype(int)
        res = run_backtest(
            sig_s.tolist(),
            close_s.tolist(),
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        rets.loc[close_s.index, sym] = np.asarray(res.daily_returns, dtype=float)
        total_trades += res.trade_count
    portfolio = rets.sum(axis=1) / top_n
    return portfolio, total_trades


def benchmark_daily_returns(close_wide: pd.DataFrame) -> pd.Series:
    """基準：所有標的等權、每日再平衡的買入持有（不含成本；買入持有成本趨近 0）。"""
    return close_wide.pct_change(fill_method=None).mean(axis=1)


def evaluate_portfolios(
    factor_wides: dict[str, pd.DataFrame],
    close_wide: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
    top_n: int = 5,
    rebalance_days: int = 20,
    direction: str | dict[str, str] = "auto",
    buffer_n: int | None = None,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
    fwd_days: int = 5,
) -> tuple[list[PortfolioResult], PortfolioResult, dict[str, float], dict[str, str]]:
    """對多個因子各跑一組投組回測，另附「因子組合」（等權平均各投組日報酬）。

    績效一律為**含成本淨值**（手續費 + 證交稅 + 滑價；純現金，借貸成本不計）；
    免成本情境請把 commission/tax/slippage 全設 0。
    回傳 (各因子結果, 因子組合結果, 基準指標, 各因子實際使用的方向與其 IC)。
    """
    fee_rate = commission + tax / 2.0  # 單邊費率：來回 = 2×commission + tax
    results: list[PortfolioResult] = []
    direction_used: dict[str, str] = {}
    daily_map: dict[str, pd.Series] = {}
    for name, fw in factor_wides.items():
        if isinstance(direction, dict):
            # 逐因子指定方向（樣本外驗證用：方向由 IS 期間決定）
            if name not in direction:
                raise ValueError(f"direction dict 缺少因子 {name!r}")
            used = direction[name]
            direction_used[name] = used
        elif direction == "auto":
            if price_df is None:
                raise ValueError("direction=auto 需要 price_df（daily_pv 長表）以計算 IC")
            used, ic_mean = auto_direction(fw, price_df, fwd_days=fwd_days)
            direction_used[name] = f"{used}（auto，全樣本 IC={ic_mean:.4f}）"
        else:
            used = direction
            direction_used[name] = used
        sig, avg_turnover = build_signals(
            fw, top_n=top_n, rebalance_days=rebalance_days, direction=used, buffer_n=buffer_n
        )
        daily, trades = portfolio_daily_returns(
            sig, close_wide, top_n=top_n, fee_rate=fee_rate, slippage_rate=slippage_rate
        )
        m = _metrics(daily)
        results.append(
            PortfolioResult(
                factor_name=name,
                direction=used,
                total_return=m["total_return"],
                annual_return=m["annual_return"],
                volatility=m["volatility"],
                sharpe=m["sharpe"],
                max_drawdown=m["max_drawdown"],
                trade_count=trades,
                avg_turnover=avg_turnover,
                win_rate=m["win_rate"],
                daily=daily,
                equity=(1.0 + daily).cumprod(),
            )
        )
        daily_map[name] = daily

    combined_daily = pd.concat(daily_map.values(), axis=1).fillna(0.0).mean(axis=1) if daily_map else pd.Series(dtype=float)
    combined = PortfolioResult(
        factor_name="組合（各因子投組等權平均）",
        direction="mixed" if len(set(r.direction for r in results)) > 1 else (results[0].direction if results else "-"),
        **_metrics(combined_daily),
        trade_count=sum(r.trade_count for r in results),
        avg_turnover=float(np.mean([r.avg_turnover for r in results])) if results else 0.0,
        daily=combined_daily,
        equity=(1.0 + combined_daily).cumprod(),
    )
    bench = _metrics(benchmark_daily_returns(close_wide))
    return results, combined, bench, direction_used


def load_factor_wides(
    factor_names: tuple[str, ...] | None = None,
    workspace: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """從 RD-Agent 工作區載入指定因子的 wide 因子值（依名稱匹配）。"""
    wanted = set(factor_names or DEFAULT_FACTORS)
    out: dict[str, pd.DataFrame] = {}
    for r in collect_results(workspace):
        if r.factor_name in wanted and r.wide is not None:
            out[r.factor_name] = r.wide
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"工作區找不到因子：{', '.join(sorted(missing))}（可先跑 factor-report 查看名稱）")
    return out


def rerun_factors_on_data(
    factor_names: tuple[str, ...] | None,
    data_path: Path,
    workdir: Path,
    python_bin: str | None = None,
    workspace: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """在指定資料集上重跑 LLM 因子程式碼，回傳 {因子名: wide 因子值}。

    每個因子建一個子工作區（factor.py + 指向 ``data_path`` 的 daily_pv.h5 symlink），
    以 ``python factor.py`` 執行（RD-Agent 的 factor.py 約定：讀 daily_pv.h5、寫 result.h5）。
    """
    names = set(factor_names or DEFAULT_FACTORS)
    workspace = workspace or (
        kstock_settings.data_dir / "qlab" / "rdagent_workspace" / "git_ignore_folder" / "RD-Agent_workspace"
    )
    matched: dict[str, Path] = {}
    for r in collect_results(workspace):
        if r.factor_name in names and r.wide is not None:
            matched[r.factor_name] = workspace / r.directory
    missing = names - set(matched)
    if missing:
        raise KeyError(f"工作區找不到因子：{', '.join(sorted(missing))}")

    workdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"資料集不存在：{data_path}")
    py = python_bin or _factor_python_bin()
    out: dict[str, pd.DataFrame] = {}
    for name, factor_dir in sorted(matched.items()):
        dstdir = workdir / name
        dstdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(factor_dir / "factor.py", dstdir / "factor.py")
        link = dstdir / "daily_pv.h5"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(data_path.resolve())
        proc = subprocess.run(
            [py, "factor.py"], cwd=dstdir, capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            raise RuntimeError(f"因子 {name} 重跑失敗：\n{proc.stderr[-2000:]}")
        df = pd.read_hdf(dstdir / "result.h5")
        if isinstance(df, pd.Series):
            df = df.to_frame(name=df.name or name)
        col = df.columns[0]
        out[name] = df[col].unstack("instrument")
    return out


def _yearly_stats(daily: pd.Series) -> dict[int, tuple[float, float, float]]:
    """日報酬 → {年份: (年報酬, 年內夏普, 年內 MDD)}。"""
    out: dict[int, tuple[float, float, float]] = {}
    for year, chunk in daily.groupby(daily.index.year):
        chunk = chunk.fillna(0.0)
        ret = float((1.0 + chunk).prod() - 1.0)
        std = chunk.std(ddof=1)
        sharpe = float(chunk.mean() / std * np.sqrt(TRADING_DAYS)) if std else 0.0
        equity = (1.0 + chunk).cumprod()
        mdd = float((1.0 - equity / equity.cummax()).max()) if len(chunk) else 0.0
        out[int(year)] = (ret, sharpe, mdd)
    return out


def _yearly_section(
    portfolios: list[tuple[str, pd.Series]],
    bench_daily: pd.Series,
) -> list[str]:
    """逐年表現區塊：檢視是否單一年度（暴漲/暴跌）貢獻了全部報酬。"""
    bench_stats = _yearly_stats(bench_daily)
    port_stats = {name: _yearly_stats(daily) for name, daily in portfolios}
    years = sorted(bench_stats)
    lines = [
        "",
        "## 逐年表現（年報酬，含成本淨值；括號內為年內夏普）",
        "",
        "用途：檢查報酬是否集中在單一年度（暴漲/暴跌年），而非逐年穩定",
        "",
        "| 年度 | " + " | ".join(name for name, _ in portfolios) + " | 基準 | 最佳因子超額 |",
        "|---|" + "---|" * (len(portfolios) + 2),
    ]
    for y in years:
        cells = []
        for name, _ in portfolios:
            ret, sharpe, _mdd = port_stats[name].get(y, (float("nan"), float("nan"), float("nan")))
            cells.append(f"{ret:+.1%}（{sharpe:.2f}）")
        b_ret = bench_stats.get(y, (float("nan"),) * 3)[0]
        best = max((port_stats[n].get(y, (float("nan"),))[0] for n, _ in portfolios), default=float("nan"))
        lines.append(f"| {y} | " + " | ".join(cells) + f" | {b_ret:+.1%} | {best - b_ret:+.1%} |")
    lines += [
        "",
        "各年度最大回撤：",
        "",
        "| 年度 | " + " | ".join(name for name, _ in portfolios) + " | 基準 |",
        "|---|" + "---|" * (len(portfolios) + 1),
    ]
    for y in years:
        cells = [f"{port_stats[n].get(y, (0, 0, float('nan')))[2]:.1%}" for n, _ in portfolios]
        b_mdd = bench_stats.get(y, (0, 0, float("nan")))[2]
        lines.append(f"| {y} | " + " | ".join(cells) + f" | {b_mdd:.1%} |")
    return lines


def build_report(
    results: list[PortfolioResult],
    combined: PortfolioResult,
    bench: dict[str, float],
    direction_used: dict[str, str],
    params: dict[str, str],
    bench_daily: pd.Series | None = None,
) -> str:
    """組裝 Markdown 回測報告。"""
    lines = [
        "# fin_factor 台股投組回測報告",
        "",
        f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **評估準則：一律以含成本（淨）績效為準**；免成本結果僅供訊號有效性驗證",
        "- 成本模型（純現金）：手續費單邊 commission、賣出證交稅 tax、滑價 slippage"
        "（台股預設來回 0.585% = 0.1425%×2 + 0.3%；借貸成本不計）",
        "- 無前視偏差：t 日收盤後訊號，t+1 日才開始計損益（kstock 回測引擎保證）",
    ]
    n_sym = int(str(params.get("標的數", "0")) or 0)
    if 0 < n_sym <= 20:
        lines.append("- ⚠️ 因子與回測跑在 debug 子集（20 檔），結果僅供管線驗證")
    else:
        lines.append(f"- 標的宇宙：{n_sym} 檔（正式評估口徑）")
    lines += ["## 參數", ""]
    lines += [f"- {k}：{v}" for k, v in params.items()]
    lines += [
        "",
        "## 績效總表",
        "",
        "| 投組 | 方向 | 總報酬 | 年化 | 夏普 | 最大回撤 | 波動 | 換倉比例/次 | 進出場次數 | 持倉日勝率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in [*results, combined]:
        d = direction_used.get(r.factor_name, r.direction)
        lines.append(
            f"| `{r.factor_name}` | {d} | {r.total_return:+.1%} | {r.annual_return:+.1%} "
            f"| {r.sharpe:.2f} | {r.max_drawdown:.1%} | {r.volatility:.1%} "
            f"| {r.avg_turnover:.0%} | {r.trade_count} | {r.win_rate:.0%} |"
        )
    lines.append(
        f"| 基準（全標的等權買入持有，免成本） | - | {bench['total_return']:+.1%} | {bench['annual_return']:+.1%} "
        f"| {bench['sharpe']:.2f} | {bench['max_drawdown']:.1%} | {bench['volatility']:.1%} | - | - | - |"
    )
    lines += _yearly_section(
        [(r.factor_name, r.daily) for r in results] + [(combined.factor_name, combined.daily)],
        bench_daily,
    )
    return "\n".join(lines) + "\n"


def _prepare_universe(
    factor_names: tuple[str, ...] | None,
    data: Path | None,
    top_symbols: int | None,
    workdir: Path | None,
    workspace: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, str]:
    """載入（或重跑）因子值與價格，回傳 (因子 wides, 價格長表, close wide, 資料描述)。"""
    if data is not None or top_symbols is not None:
        data_path = Path(data) if data is not None else DEFAULT_DATA_H5
        price_df = pd.read_hdf(data_path, key="data")
        if top_symbols is not None:
            syms = sorted(price_df.index.get_level_values("instrument").unique())[:top_symbols]
            price_df = price_df[price_df.index.get_level_values("instrument").isin(syms)]
            desc = f"{data_path}（前 {top_symbols} 檔，依代碼排序）"
        else:
            desc = str(data_path)
        wd = workdir or (kstock_settings.data_dir / "qlab" / "backtest_workdir")
        sub_h5 = wd / "daily_pv.h5"
        sub_h5.parent.mkdir(parents=True, exist_ok=True)
        price_df.to_hdf(sub_h5, key="data")
        factor_wides = rerun_factors_on_data(factor_names, sub_h5, wd, workspace=workspace)
    else:
        factor_wides = load_factor_wides(factor_names, workspace=workspace)
        price_df = _load_price_data()
        if price_df is None:
            raise FileNotFoundError("找不到 daily_pv.h5（factor_source_data_tw_debug / tw / 內建 A 股皆無）")
        desc = "factor_report 候選（debug 20 檔優先）"
    close_wide = price_df["$close"].unstack("instrument")
    # 壞資料防禦：0 或負值（停牌誤植等）視為缺價，避免產生 inf 日報酬毒化整條權益曲線
    close_wide = close_wide.mask(close_wide <= 0.0)
    return factor_wides, price_df, close_wide, desc


def run_backtest_report(
    factor_names: tuple[str, ...] | None = None,
    top_n: int = 5,
    rebalance_days: int = 20,
    direction: str = "auto",
    buffer_n: int | None = None,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
    fwd_days: int = 5,
    output: Path | None = None,
    workspace: Path | None = None,
    data: Path | None = None,
    top_symbols: int | None = None,
    workdir: Path | None = None,
) -> Path:
    """執行回測並寫出報告，回傳報告路徑。

    ``data``／``top_symbols``：指定價格資料集（或在資料集內取前 N 檔子集）時，
    因子值會在**同一宇宙**上重跑（result.h5 的既有因子值僅涵蓋 debug 20 檔）。
    """
    factor_wides, price_df, close_wide, data_desc = _prepare_universe(
        factor_names, data, top_symbols, workdir, workspace=workspace
    )
    results, combined, bench, direction_used = evaluate_portfolios(
        factor_wides,
        close_wide,
        price_df=price_df,
        top_n=top_n,
        rebalance_days=rebalance_days,
        direction=direction,
        buffer_n=buffer_n,
        commission=commission,
        tax=tax,
        slippage_rate=slippage_rate,
        fwd_days=fwd_days,
    )
    params = {
        "因子": "、".join(r.factor_name for r in results),
        "top_n": str(top_n),
        "再平衡間隔": f"{rebalance_days} 個交易日",
        "方向": direction,
        "緩衝帶": f"{buffer_n} 名（跌出才換）" if buffer_n else "無",
        "資料來源": data_desc,
        "手續費（單邊）": f"{commission:.4%}",
        "證交稅（賣出）": f"{tax:.4%}",
        "滑價（單邊）": f"{slippage_rate:.4%}",
        "IC 前瞻天數": str(fwd_days),
        "標的數": str(close_wide.shape[1]),
        "期間": f"{close_wide.index.min()} ~ {close_wide.index.max()}",
    }
    report = build_report(
        results,
        combined,
        bench,
        direction_used,
        params,
        bench_daily=benchmark_daily_returns(close_wide),
    )
    output = output or (kstock_settings.data_dir / "qlab" / BACKTEST_REPORT_FILENAME)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def run_oos_report(
    factor_names: tuple[str, ...] | None = None,
    top_n: int = 5,
    rebalance_days: int = 20,
    buffer_n: int | None = None,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
    fwd_days: int = 5,
    is_ratio: float = 0.7,
    data: Path | None = None,
    top_symbols: int | None = None,
    workdir: Path | None = None,
    output: Path | None = None,
) -> Path:
    """樣本外驗證：方向只用 IS 期間 IC 決定，OOS 期間以固定方向回測。

    切分：交易日序列前 ``is_ratio`` 為 IS、其餘為 OOS。因子皆為 trailing 計算，
    全期一次算完再切片不會引入前視；唯一 in-sample 元素（方向選擇）被隔離在 IS。
    回傳報告路徑。
    """
    factor_wides, price_df, close_wide, data_desc = _prepare_universe(
        factor_names, data, top_symbols, workdir
    )
    dates = close_wide.index
    split = int(len(dates) * is_ratio)
    is_dates, oos_dates = dates[:split], dates[split:]
    if len(is_dates) < 60 or len(oos_dates) < 60:
        raise ValueError(f"樣本太短無法切分（IS={len(is_dates)}、OOS={len(oos_dates)} 日）")

    def _slice_df(df: pd.DataFrame, dts) -> pd.DataFrame:
        return df[df.index.get_level_values("datetime").isin(dts)]

    # 1) 方向只由 IS 決定
    directions: dict[str, str] = {}
    is_ic: dict[str, float] = {}
    for name, fw in factor_wides.items():
        used, ic_mean = auto_direction(fw.loc[is_dates], _slice_df(price_df, is_dates), fwd_days=fwd_days)
        directions[name] = used
        is_ic[name] = ic_mean

    # 2) IS / OOS 各回測一次（同方向）；免成本版另跑 OOS
    def _run(fws: dict[str, pd.DataFrame], cw: pd.DataFrame) -> tuple[list, PortfolioResult, dict[str, float]]:
        res, comb, _bench, _dirs = evaluate_portfolios(
            fws,
            cw,
            top_n=top_n,
            rebalance_days=rebalance_days,
            direction=directions,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
        )
        return res, comb, _bench

    is_fws = {n: fw.loc[is_dates] for n, fw in factor_wides.items()}
    oos_fws = {n: fw.loc[oos_dates] for n, fw in factor_wides.items()}
    is_res, is_comb, is_bench = _run(is_fws, close_wide.loc[is_dates])
    oos_res, oos_comb, oos_bench = _run(oos_fws, close_wide.loc[oos_dates])
    # 免成本 OOS（訊號有效性對照）
    if commission or tax or slippage_rate:
        gross_res, gross_comb, _, _ = evaluate_portfolios(
            oos_fws,
            close_wide.loc[oos_dates],
            top_n=top_n,
            rebalance_days=rebalance_days,
            direction=directions,
            buffer_n=buffer_n,
            commission=0.0,
            tax=0.0,
            slippage_rate=0.0,
        )
    else:
        gross_res, gross_comb = oos_res, oos_comb

    # 3) OOS IC（診斷訊號是否還活著）
    oos_ic: dict[str, tuple[float, float]] = {}
    fwd = forward_returns(_slice_df(price_df, oos_dates), days=fwd_days)
    for name, fw in oos_fws.items():
        s = daily_rank_ic(fw, fwd)
        if len(s) > 10 and s.std() > 0:
            oos_ic[name] = (float(s.mean()), float(s.mean() / (s.std() / np.sqrt(len(s)))))

    is_by = {r.factor_name: r for r in is_res}
    oos_by = {r.factor_name: r for r in oos_res}
    gross_by = {r.factor_name: r for r in gross_res}

    lines = [
        "# fin_factor 樣本外（OOS）驗證報告",
        "",
        f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 資料：{data_desc}",
        f"- 切分：IS {is_dates.min().date()} ~ {is_dates.max().date()}（{len(is_dates)} 日，{is_ratio:.0%}）"
        f" / OOS {oos_dates.min().date()} ~ {oos_dates.max().date()}（{len(oos_dates)} 日）",
        "- 方向只由 IS 期間 IC 決定，OOS 期間固定方向執行；因子皆為 trailing 計算，無前視",
        f"- 成本：來回 {2 * commission + tax:.3%}（手續費 {commission:.4%}×2 + 證交稅 {tax:.1%}）"
        + (f"；緩衝帶 {buffer_n} 名" if buffer_n else "")
        + ("；**評估以 OOS 淨績效為準**" if (commission or tax) else "；⚠️ 免成本模式"),
        "",
        "| 投組 | 方向 | OOS 淨年化 | OOS 夏普 | OOS 回撤 | OOS 免成本年化 | IS 淨年化 | OOS 基準年化 | OOS 超額(淨) | OOS IC(t) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in oos_res:
        name = r.factor_name
        g = gross_by.get(name)
        i = is_by.get(name)
        ic_t = f"{oos_ic[name][0]:.4f}（{oos_ic[name][1]:.2f}）" if name in oos_ic else "—"
        lines.append(
            f"| `{name}` | {directions[name]} | {r.annual_return:+.1%} | {r.sharpe:.2f} "
            f"| {r.max_drawdown:.1%} | {(g.annual_return if g else float('nan')):+.1%} "
            f"| {(i.annual_return if i else float('nan')):+.1%} | {oos_bench['annual_return']:+.1%} "
            f"| {r.annual_return - oos_bench['annual_return']:+.1%} | {ic_t} |"
        )
    gc, ic_, g = oos_comb, is_comb, gross_comb
    lines.append(
        f"| 組合 | mixed | {gc.annual_return:+.1%} | {gc.sharpe:.2f} | {gc.max_drawdown:.1%} "
        f"| {g.annual_return:+.1%} | {ic_.annual_return:+.1%} | {oos_bench['annual_return']:+.1%} "
        f"| {gc.annual_return - oos_bench['annual_return']:+.1%} | - |"
    )
    lines.append(
        f"| 基準（等權買入持有，免成本） | - | - | {oos_bench['sharpe']:.2f} | {oos_bench['max_drawdown']:.1%} | - | - | - | - | - |"
    )
    lines += [
        "",
        f"- IS 期間各因子 IC（方向依據）："
        + "、".join(f"`{n}`={v:.4f}" for n, v in is_ic.items()),
        "",
        "判讀：OOS 淨年化 > OOS 基準且 OOS IC 同號顯著 → 訊號有樣本外價值；",
        "若只有免成本為正而淨值轉負 → 成本侵蝕；若 OOS IC 變號 → 訊號不穩定，勿交易。",
    ]
    out_path = output or (kstock_settings.data_dir / "qlab" / "oos_backtest_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

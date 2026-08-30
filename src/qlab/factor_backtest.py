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
) -> tuple[pd.DataFrame, float]:
    """由因子值建立每日持倉訊號（1=持有多方 slot、0=空手）。

    再平衡日 = 因子索引每隔 ``rebalance_days`` 列；訊號在再平衡日收盤後產生，
    由回測引擎延後一天生效，因此此處不需再 shift。
    回傳 (訊號表, 平均每次再平衡換倉比例)。
    """
    if direction not in ("top", "bottom"):
        raise ValueError(f"direction 必須是 top/bottom，收到 {direction!r}")
    sig = pd.DataFrame(0.0, index=factor_wide.index, columns=factor_wide.columns)
    turnovers: list[float] = []
    prev_picks: set | None = None
    row_pos = list(range(0, len(factor_wide.index), rebalance_days))
    for k, start in enumerate(row_pos):
        end = row_pos[k + 1] if k + 1 < len(row_pos) else len(factor_wide.index)
        dt = factor_wide.index[start]
        row = factor_wide.iloc[start].dropna()
        if row.empty:
            prev_picks = set()
            continue
        picks = row.nlargest(top_n).index if direction == "top" else row.nsmallest(top_n).index
        sig.iloc[start:end, sig.columns.get_indexer(picks)] = 1.0
        if prev_picks is not None:
            overlap = len(prev_picks & set(picks))
            turnovers.append(1.0 - overlap / top_n)
        prev_picks = set(picks)
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
    direction: str = "auto",
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
        if direction == "auto":
            if price_df is None:
                raise ValueError("direction=auto 需要 price_df（daily_pv 長表）以計算 IC")
            used, ic_mean = auto_direction(fw, price_df, fwd_days=fwd_days)
            direction_used[name] = f"{used}（auto，全樣本 IC={ic_mean:.4f}）"
        else:
            used = direction
            direction_used[name] = used
        sig, avg_turnover = build_signals(fw, top_n=top_n, rebalance_days=rebalance_days, direction=used)
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


def build_report(
    results: list[PortfolioResult],
    combined: PortfolioResult,
    bench: dict[str, float],
    direction_used: dict[str, str],
    params: dict[str, str],
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
        "- ⚠️ 目前因子跑在 debug 子集（20 檔），結果僅供管線驗證",
        "",
    ]
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
    return "\n".join(lines) + "\n"


def run_backtest_report(
    factor_names: tuple[str, ...] | None = None,
    top_n: int = 5,
    rebalance_days: int = 20,
    direction: str = "auto",
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
    fwd_days: int = 5,
    output: Path | None = None,
    workspace: Path | None = None,
) -> Path:
    """執行回測並寫出報告，回傳報告路徑。"""
    factor_wides = load_factor_wides(factor_names, workspace=workspace)
    price_df = _load_price_data()
    if price_df is None:
        raise FileNotFoundError("找不到 daily_pv.h5（factor_source_data_tw_debug / tw / 內建 A 股皆無）")
    close_wide = price_df["$close"].unstack("instrument")
    # 壞資料防禦：0 或負值（停牌誤植等）視為缺價，避免產生 inf 日報酬毒化整條權益曲線
    close_wide = close_wide.mask(close_wide <= 0.0)
    results, combined, bench, direction_used = evaluate_portfolios(
        factor_wides,
        close_wide,
        price_df=price_df,
        top_n=top_n,
        rebalance_days=rebalance_days,
        direction=direction,
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
        "手續費（單邊）": f"{commission:.4%}",
        "證交稅（賣出）": f"{tax:.4%}",
        "滑價（單邊）": f"{slippage_rate:.4%}",
        "IC 前瞻天數": str(fwd_days),
        "標的數": str(close_wide.shape[1]),
        "期間": f"{close_wide.index.min()} ~ {close_wide.index.max()}",
    }
    report = build_report(results, combined, bench, direction_used, params)
    output = output or (kstock_settings.data_dir / "qlab" / BACKTEST_REPORT_FILENAME)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output

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

from qlab.factor_report import (
    _load_price_data,
    _normalize_factor_frame,
    collect_results,
    daily_rank_ic,
    forward_returns,
)

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
    direction: str  # top=買因子最大者、bottom=買因子最小者、short=放空因子最大者
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


def ic_stability(
    factor_wides: dict[str, pd.DataFrame],
    price_df: pd.DataFrame,
    fwd_days: int = 5,
    is_ratio: float = 0.7,
) -> dict[str, tuple[float, float, float]]:
    """IC 樣本穩定性：以時間前 ``is_ratio`` 切分 IS/OOS，回傳 {因子: (IS IC, OOS IC, OOS t 值)}。

    因子皆為 trailing 計算，切片不引入前視；OOS IC 與 IS IC 變號或 |t|<2
    代表全樣本 auto 方向的 alpha 證據薄弱。
    """
    dates = price_df.index.get_level_values("datetime").unique().sort_values()
    split = int(len(dates) * is_ratio)
    is_dates, oos_dates = set(dates[:split]), set(dates[split:])
    fwd = forward_returns(price_df, days=fwd_days)
    out: dict[str, tuple[float, float, float]] = {}
    for name, fw in factor_wides.items():
        is_ic = daily_rank_ic(fw[fw.index.isin(is_dates)], fwd)
        oos_ic = daily_rank_ic(fw[fw.index.isin(oos_dates)], fwd)
        if len(oos_ic) < 10 or oos_ic.std() == 0:
            out[name] = (float(is_ic.mean()) if len(is_ic) else float("nan"), float("nan"), float("nan"))
            continue
        t = float(oos_ic.mean() / (oos_ic.std() / np.sqrt(len(oos_ic))))
        out[name] = (float(is_ic.mean()) if len(is_ic) else float("nan"), float(oos_ic.mean()), t)
    return out


def _resolve_directions(
    factor_wides: dict[str, pd.DataFrame],
    price_df: pd.DataFrame | None,
    direction: str | dict[str, str],
    fwd_days: int,
) -> dict[str, str]:
    """把 direction 參數解析成逐因子乾淨方向（top/bottom），供免成本重跑沿用同訊號。"""
    if isinstance(direction, dict):
        return dict(direction)
    if direction == "auto":
        if price_df is None:
            raise ValueError("direction=auto 需要 price_df（daily_pv 長表）以計算 IC")
        return {name: auto_direction(fw, price_df, fwd_days=fwd_days)[0] for name, fw in factor_wides.items()}
    return {name: direction for name in factor_wides}


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

    ``direction`` 僅接受 top/bottom（long-only 訊號）；做空由
    :func:`evaluate_portfolios` 以 direction=short（top 訊號 + 日報酬取負）表達。
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


TRADE_LEDGER_COLUMNS = [
    "symbol",  # 標的（{market}{symbol}）
    "status",  # closed=已平倉、open=樣本結束仍持倉（exit_* 為最後標記值）
    "entry_date",  # 建倉訊號日（以當日收盤價成交＝台股盤後定價 13:40–14:00 口徑）
    "entry_price",  # 建倉成交價（未復權原始收盤，實際金額）
    "exit_date",  # 出場訊號日（open 時為空）
    "exit_price",  # 出場成交價（原始價；open 時為最後標記價）
    "holding_days",  # 持有交易日（標的壓縮序列：缺價日不計）
    "factor_value_at_entry",  # 建倉日因子值
    "rank_at_entry",  # 建倉日橫截面排名（bottom：1=因子最小）
    "universe_size_at_entry",  # 建倉日有因子值的檔數
    "gross_return",  # 毛報酬（復權口徑、含股息，≠ exit_price/entry_price − 1）
    "net_return",  # 淨報酬（復權口徑 + 來回成本，與回測引擎逐槽複利一致）
    "exit_reason",  # 出場原因
]


def _symbol_trades(
    symbol: str,
    sig_s: pd.Series,
    adj_s: pd.Series,
    raw_s: pd.Series,
    factor_wide: pd.DataFrame,
    direction: str,
    band: int,
    fee_rate: float,
) -> list[dict]:
    """單一標的壓縮序列上的訊號 → 逐筆交易（與 :func:`run_backtest` 的部位/成本時點一致）。

    成交慣例（引擎語意）：**訊號變化日的收盤價成交**——t 日訊號買進、以 close(t) 成交並賺
    close(t)→close(t+1)；出場訊號日以 close(出場日) 賣出。進場費記在進場後首日、出場費記在
    出場後首日（引擎 strat 的 turnover 項），淨報酬即逐槽複利結果。
    成交價（``entry_price``/``exit_price``）為**原始價**（實際金額）；
    毛淨報酬以**復權序列**計算（含股息，除權息日不視為虧損）。
    """
    c = adj_s.to_numpy(dtype=float)
    p = raw_s.to_numpy(dtype=float)
    dts = adj_s.index
    s = sig_s.to_numpy(dtype=int)
    n = len(s)
    ascending = direction == "bottom"
    trades: list[dict] = []
    i = 0
    while i < n:
        if s[i] != 1 or (i > 0 and s[i - 1] == 1):
            i += 1
            continue
        j = i
        while j + 1 < n and s[j + 1] == 1:
            j += 1
        # 進場價 = 訊號日 close[i]；部位最早 i+1 日生效 → 訊號落在末列時引擎不會建立部位
        if i + 1 < n:
            is_open = j == n - 1
            exit_i = None if is_open else j + 1  # 出場價 = 出場訊號日收盤
            mark_i = (n - 1) if is_open else exit_i
            entry_row = factor_wide.loc[dts[i]]
            net_mult = c[i + 1] / c[i] - fee_rate  # 進場日：價差 − 單邊成本
            if mark_i > i + 1:
                net_mult *= float(np.prod(c[i + 2 : mark_i + 1] / c[i + 1 : mark_i]))
            if not is_open:
                net_mult *= 1.0 - fee_rate  # 出場日：單邊成本
            if is_open:
                reason = ""
            else:
                exit_row = factor_wide.loc[dts[exit_i]]
                if pd.isna(exit_row.get(symbol, np.nan)):
                    reason = "因子值缺失（停牌或壞資料）"
                else:
                    r_exit = exit_row.rank(ascending=ascending).get(symbol, np.nan)
                    reason = (
                        f"跌出前 {band} 名（排名第 {int(r_exit)}）"
                        if pd.notna(r_exit) and r_exit > band
                        else "再平衡換倉"
                    )
            trades.append(
                {
                    "symbol": symbol,
                    "status": "open" if is_open else "closed",
                    "entry_date": dts[i],
                    "entry_price": float(p[i]),
                    "exit_date": None if is_open else dts[exit_i],
                    "exit_price": float(p[mark_i]),
                    "holding_days": int(mark_i - i),
                    "factor_value_at_entry": float(entry_row.get(symbol, np.nan)),
                    "rank_at_entry": float(entry_row.rank(ascending=ascending).get(symbol, np.nan)),
                    "universe_size_at_entry": int(entry_row.notna().sum()),
                    "gross_return": float(c[mark_i] / c[i] - 1.0),
                    "net_return": float(net_mult - 1.0),
                    "exit_reason": reason,
                }
            )
        i = j + 1
    return trades


def build_trade_ledger(
    factor_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    direction: str,
    top_n: int,
    rebalance_days: int,
    buffer_n: int | None = None,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
    adj_close_wide: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """因子訊號 → 實際進出倉明細表（一列＝一筆交易；樣本末端未出場者 status=open）。

    訊號以 :func:`build_signals` 重建（與回測同一份），或以 ``signals`` 直接提供
    （執行時點變換等場景：明細必須與變換後訊號一致）。逐標的在其「壓縮序列」（缺價日剔除，
    與 :func:`portfolio_daily_returns` 口徑一致）上拆出進出場：

    - ``close_wide``：**原始收盤價**（成交價顯示用）；``adj_close_wide``：**復權收盤價**
      （報酬計算用，含股息；壓縮序列必須與回測引擎一致）。``adj_close_wide=None`` 時
      退回用 ``close_wide`` 本身計算報酬（純價差、不含股息）。
    - 建倉：訊號首日**收盤價**（台股盤後定價以收盤價成交的口徑；量能因子需收盤後才確定，
      盤中無法搶先）。
    - 出場：出場訊號日收盤價；原因為跌出緩衝帶，或因子值缺失（停牌/壞資料被剔出選股）。
    - ``net_return``：進場日 (1+r−fee) × 持有日 (1+r) × 出場日 (1−fee)，其中
      fee = commission + tax/2 + slippage（與引擎單邊費率一致，來回恰為 2×commission + tax）。
    """
    if direction not in ("top", "bottom"):
        raise ValueError(f"direction 必須是 top/bottom，收到 {direction!r}")
    if signals is None:
        sig, _ = build_signals(
            factor_wide, top_n=top_n, rebalance_days=rebalance_days, direction=direction, buffer_n=buffer_n
        )
    else:
        sig = signals.copy()
    band = buffer_n or top_n
    fee_rate = commission + tax / 2.0 + slippage_rate
    adj_wide = close_wide if adj_close_wide is None else adj_close_wide
    rows: list[dict] = []
    for symbol in sig.columns:
        if symbol not in adj_wide.columns:
            continue
        adj_s = adj_wide[symbol].dropna()
        if len(adj_s) < 3:
            continue
        sig_s = sig[symbol].reindex(adj_s.index).fillna(0.0).astype(int)
        if adj_close_wide is None or symbol not in close_wide.columns:
            raw_s = adj_s
        else:
            raw_s = close_wide[symbol].reindex(adj_s.index)
        rows.extend(
            _symbol_trades(symbol, sig_s, adj_s, raw_s, factor_wide, direction, band, fee_rate)
        )
    return pd.DataFrame(rows, columns=TRADE_LEDGER_COLUMNS)


def export_trade_ledgers(
    factor_wides: dict[str, pd.DataFrame],
    close_wide: pd.DataFrame,
    directions: dict[str, str],
    top_n: int,
    rebalance_days: int,
    buffer_n: int | None,
    commission: float,
    tax: float,
    slippage_rate: float,
    output_dir: Path,
    prefix: str = "",
    adj_close_wide: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """逐因子輸出交易明細 CSV（``<prefix>trades_<因子>.csv``，utf-8-sig 供 Excel 直接開啟）。

    ``close_wide`` 為原始價、``adj_close_wide`` 為復權價（含股息、與回測引擎同序列）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ledgers: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for name, fw in factor_wides.items():
        df = build_trade_ledger(
            fw,
            close_wide,
            direction=directions[name],
            top_n=top_n,
            rebalance_days=rebalance_days,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
            adj_close_wide=adj_close_wide,
        )
        path = output_dir / f"{prefix}trades_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        ledgers[name], paths[name] = df, path
    return ledgers, paths


def _ledger_summary_section(
    ledgers: dict[str, pd.DataFrame], csv_paths: dict[str, Path]
) -> list[str]:
    """交易明細摘要（附於回測報告尾端）。"""
    lines = [
        "",
        "## 交易明細",
        "",
        "成交慣例：建倉/出場皆以**訊號日收盤價**成交（台股盤後定價 13:40–14:00 口徑，"
        "量能因子需收盤後才確定）；成交價為原始價，毛/淨報酬以**復權價**計（含股息）；"
        "淨報酬含來回成本，與回測引擎逐槽複利一致；"
        "持有交易日以標的自身壓縮序列計（缺價日不計）。",
        "",
    ]
    for name, df in ledgers.items():
        path = csv_paths.get(name)
        suffix = f" → `{path.name}`" if path else ""
        if df.empty:
            lines.append(f"- `{name}`：期間內無交易{suffix}")
            continue
        closed = df[df["status"] == "closed"]
        n_open = int((df["status"] == "open").sum())
        win = float((closed["net_return"] > 0).mean()) if len(closed) else float("nan")
        mean_net = float(closed["net_return"].mean()) if len(closed) else float("nan")
        avg_hold = float(closed["holding_days"].mean()) if len(closed) else float("nan")
        mean_s = f"{mean_net:+.2%}" if np.isfinite(mean_net) else "-"
        win_s = f"{win:.0%}" if np.isfinite(win) else "-"
        hold_s = f"{avg_hold:.1f} 個交易日" if np.isfinite(avg_hold) else "-"
        lines.append(
            f"- `{name}`：{len(df)} 筆（已平倉 {len(closed)}、持倉中 {n_open}）；"
            f"單筆淨報酬平均 {mean_s}、勝率 {win_s}、平均持有 {hold_s}{suffix}"
        )
    return lines


def _open_exec_overrides(
    sig_s: pd.Series,
    adj_open_s: pd.Series,
    raw_open_s: pd.Series,
) -> tuple[list, dict, dict]:
    """掃描訊號 run，產生 V1（t+1 開盤進出）所需的序列覆寫。

    引擎語意等價：把進場訊號日 a 的收盤替換成**次日開盤**（adj[a] = open(a+1)），並把
    出場訊號日 b 的訊號延長一天、b+1 日收盤替換成**當日開盤**（adj[b+1] = open(b+1)），
    使 :func:`run_backtest` 在變換後序列上的逐日損益恰為「開盤進、開盤出」——
    進場首日賺 open→close、末日在 open(b+1) 出場。開盤缺值（≤0/NaN）時不覆寫，
    該筆交易退化為 t 收盤口徑。同標的再進場至少隔一個再平衡窗，覆寫日不會相鄰衝突。

    回傳 (flip_dates, entry_overrides, exit_overrides)；
    overrides 為 {日期: (復權開盤, 原始開盤)}。
    """
    s = sig_s.to_numpy(dtype=int)
    ao = adj_open_s.to_numpy(dtype=float)
    ro = raw_open_s.to_numpy(dtype=float)
    dts = sig_s.index
    n = len(s)
    flips: list = []
    entry: dict = {}
    exit_: dict = {}
    i = 0
    while i < n:
        if s[i] != 1 or (i > 0 and s[i - 1] == 1):
            i += 1
            continue
        j = i
        while j + 1 < n and s[j + 1] == 1:
            j += 1
        b = j + 1  # 出場訊號日（壓縮索引）
        if (
            i + 1 < n
            and np.isfinite(ao[i + 1]) and ao[i + 1] > 0
            and np.isfinite(ro[i + 1]) and ro[i + 1] > 0
        ):
            entry[dts[i]] = (float(ao[i + 1]), float(ro[i + 1]))
        if b < n - 1 and np.isfinite(ao[b + 1]) and ao[b + 1] > 0 and np.isfinite(ro[b + 1]) and ro[b + 1] > 0:
            flips.append(dts[b])
            exit_[dts[b + 1]] = (float(ao[b + 1]), float(ro[b + 1]))
        i = j + 1
    return flips, entry, exit_


def execution_timing_variants(
    factor_wide: pd.DataFrame,
    adj_close_wide: pd.DataFrame,
    raw_close_wide: pd.DataFrame,
    adj_open_wide: pd.DataFrame,
    raw_open_wide: pd.DataFrame,
    direction: str,
    top_n: int,
    rebalance_days: int,
    buffer_n: int | None = None,
    commission: float = 0.001425,
    tax: float = 0.003,
    slippage_rate: float = 0.0,
) -> list[dict]:
    """同一份訊號、三種成交時點的對照（回報告用列格式）。

    - ``t 收盤``：回測假設＝台股盤後定價（13:40–14:00 以收盤價撮合）；進場 close(t)、出場 close(t′)
    - ``t+1 開盤``：進場 open(t+1)、出場 open(t′+1)——錯過訊號夜 jump，但流動性最佳
    - ``t+1 收盤``：進場 close(t+1)、出場 close(t′+1)——完全錯過 t→t+1 段（最保守）

    三者訊號、再平衡、緩衝帶、成本完全相同；差異只在成交價基準。
    t收盤 − t+1開盤 的年化差＝**隔夜段貢獻**；t+1開盤 − t+1收盤＝**日內段貢獻**。
    """
    if direction not in ("top", "bottom"):
        raise ValueError(
            f"exec-timing 僅支援 top/bottom（做空為獨立診斷），收到 {direction!r}"
        )
    fee_rate = commission + tax / 2.0
    sig, _ = build_signals(
        factor_wide, top_n=top_n, rebalance_days=rebalance_days, direction=direction, buffer_n=buffer_n
    )
    sig2 = sig.shift(1).fillna(0.0)  # t+1 收盤＝訊號再延一天生效（引擎語意）

    sig1 = sig.copy()
    c1 = adj_close_wide.copy()
    p1 = raw_close_wide.copy()
    for sym in sig.columns:
        if sym not in adj_close_wide.columns:
            continue
        adj_s = adj_close_wide[sym].dropna()
        if len(adj_s) < 3 or sym not in adj_open_wide.columns:
            continue
        sig_s = sig[sym].reindex(adj_s.index).fillna(0.0)
        ao_s = adj_open_wide[sym].reindex(adj_s.index)
        if sym in raw_open_wide.columns:
            ro_s = raw_open_wide[sym].reindex(adj_s.index)
        else:
            ro_s = pd.Series(np.nan, index=adj_s.index)
        flips, entry_ov, exit_ov = _open_exec_overrides(sig_s, ao_s, ro_s)
        for d in flips:
            sig1.loc[d, sym] = 1.0
        for d, (va, _vr) in entry_ov.items():
            c1.loc[d, sym] = va
        for d, (va, _vr) in exit_ov.items():
            c1.loc[d, sym] = va
        for d, (_va, vr) in {**entry_ov, **exit_ov}.items():
            if d in p1.index:
                p1.loc[d, sym] = vr

    plan = [
        ("t 收盤（盤後定價，回測假設）", "close(t)", "close(t′)", sig, adj_close_wide, raw_close_wide),
        ("t+1 開盤", "open(t+1)", "open(t′+1)", sig1, c1, p1),
        ("t+1 收盤", "close(t+1)", "close(t′+1)", sig2, adj_close_wide, raw_close_wide),
    ]
    rows: list[dict] = []
    for vname, epx, xpx, sigx, adjx, rawx in plan:
        daily, n_trades = portfolio_daily_returns(
            sigx, adjx, top_n=top_n, fee_rate=fee_rate, slippage_rate=slippage_rate
        )
        m = _metrics(daily)
        led = build_trade_ledger(
            factor_wide,
            rawx,
            direction=direction,
            top_n=top_n,
            rebalance_days=rebalance_days,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
            adj_close_wide=adjx,
            signals=sigx,
        )
        closed = led[led["status"] == "closed"] if len(led) else led
        rows.append(
            {
                "執行時點": vname,
                "進場價": epx,
                "出場價": xpx,
                "淨年化": m["annual_return"],
                "夏普": m["sharpe"],
                "最大回撤": m["max_drawdown"],
                "進出場次數": n_trades,
                "單筆淨報酬平均": float(closed["net_return"].mean()) if len(closed) else float("nan"),
                "勝率": float((closed["net_return"] > 0).mean()) if len(closed) else float("nan"),
                "平均持有日": float(closed["holding_days"].mean()) if len(closed) else float("nan"),
            }
        )
    return rows


def benchmark_daily_returns(close_wide: pd.DataFrame) -> pd.Series:
    """基準：所有標的等權、**每日再平衡**的組合日報酬（不含成本）。

    注意：這是「每日再平衡等權」，不是靜態買入持有——標籤以此為準，
    兩者的再平衡溢價不同，比較時須知悉。
    """
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
    ``direction`` 支援 top/bottom/short/auto：``short``＝放空因子最大者（做空），
    以「top 訊號、日報酬取負」表達——成本結構與做多對稱（進出各一次手續費、
    賣出側證交稅），借券費依評估規則不計。
    回傳 (各因子結果, 因子組合結果, 基準指標, 各因子實際使用的方向與其 IC)。
    """
    if direction not in ("top", "bottom", "short", "auto") and not isinstance(direction, dict):
        raise ValueError(f"direction 必須是 top/bottom/short/auto，收到 {direction!r}")
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
        if used not in ("top", "bottom", "short"):
            raise ValueError(f"direction 必須是 top/bottom/short，收到 {used!r}")
        sig, avg_turnover = build_signals(
            fw,
            top_n=top_n,
            rebalance_days=rebalance_days,
            direction="top" if used == "short" else used,
            buffer_n=buffer_n,
        )
        daily, trades = portfolio_daily_returns(
            sig, close_wide, top_n=top_n, fee_rate=fee_rate, slippage_rate=slippage_rate
        )
        if used == "short":
            daily = -daily
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
        df = _normalize_factor_frame(df)
        if df is None:
            raise RuntimeError(f"因子 {name} 重跑後的 result.h5 index 無法辨識（需 (datetime, instrument)）")
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
    bench_daily: pd.Series | None,
) -> list[str]:
    """逐年表現區塊：檢視是否單一年度（暴漲/暴跌）貢獻了全部報酬。"""
    bench_stats = _yearly_stats(bench_daily) if bench_daily is not None and len(bench_daily) else {}
    port_stats = {name: _yearly_stats(daily) for name, daily in portfolios}
    years = sorted(set(bench_stats) | {y for stats in port_stats.values() for y in stats})
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
        b_cell = f"{b_ret:+.1%}" if np.isfinite(b_ret) else "-"
        best = max((port_stats[n].get(y, (float("nan"),))[0] for n, _ in portfolios), default=float("nan"))
        lines.append(f"| {y} | " + " | ".join(cells) + f" | {b_cell} | {best - b_ret:+.1%} |")
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
        b_cell = f"{b_mdd:.1%}" if np.isfinite(b_mdd) else "-"
        lines.append(f"| {y} | " + " | ".join(cells) + f" | {b_cell} |")
    return lines


def _robustness_section(
    results: list[PortfolioResult],
    combined: PortfolioResult,
    gross_results: list[PortfolioResult] | None,
    gross_combined: PortfolioResult | None,
    ic_stability_map: dict[str, tuple[float, float, float]] | None,
) -> list[str]:
    """成本與穩健性對照區塊：免成本年化、成本侵蝕、IS/OOS IC。"""
    gross_map = {r.factor_name: r for r in (gross_results or [])}
    if gross_combined is not None:
        gross_map[combined.factor_name] = gross_combined
    stab = ic_stability_map or {}
    lines = [
        "",
        "## 成本與穩健性對照",
        "",
        "免成本＝同訊號、零成本重跑（僅驗證訊號有無 alpha）；IC 以時間前 70% 為 IS、其餘為 OOS。",
        "",
        "| 投組 | 淨年化 | 免成本年化 | 成本侵蝕 | IS IC | OOS IC（t 值） |",
        "|---|---|---|---|---|---|",
    ]
    for r in [*results, combined]:
        g = gross_map.get(r.factor_name)
        gross_ann = g.annual_return if g is not None else float("nan")
        erosion = (gross_ann - r.annual_return) * 100.0
        s = stab.get(r.factor_name)
        is_ic = f"{s[0]:+.4f}" if s and np.isfinite(s[0]) else "-"
        oos = f"{s[1]:+.4f}（{s[2]:+.2f}）" if s and np.isfinite(s[1]) else "-"
        lines.append(
            f"| `{r.factor_name}` | {r.annual_return:+.1%} | {gross_ann:+.1%} | {erosion:+.1f}pp | {is_ic} | {oos} |"
        )
    return lines


def _interpretation(
    combined: PortfolioResult,
    bench: dict[str, float],
    bench_daily: pd.Series | None,
    gross_results: list[PortfolioResult] | None,
    gross_combined: PortfolioResult | None,
    ic_stability_map: dict[str, tuple[float, float, float]] | None,
) -> list[str]:
    """自動判讀：成本侵蝕、OOS IC 穩定性、報酬集中度、回撤對比。"""
    lines = ["", "## 判讀", ""]
    stab = ic_stability_map or {}
    for name, (is_ic, oos_ic, t) in stab.items():
        if not np.isfinite(oos_ic):
            lines.append(f"- `{name}`：OOS IC 樣本不足，無法評估穩定性")
        elif np.isfinite(is_ic) and (is_ic >= 0) != (oos_ic >= 0):
            lines.append(
                f"- `{name}`：⚠️ OOS IC（{oos_ic:+.4f}）與 IS IC（{is_ic:+.4f}）**變號** → 訊號不穩定，勿交易"
            )
        elif abs(t) < 2:
            lines.append(
                f"- `{name}`：OOS IC 同號但**不顯著**（|t|={abs(t):.2f}<2）→ 樣本外 alpha 證據薄弱，勿以全樣本績效下結論"
            )
        else:
            lines.append(f"- `{name}`：OOS IC 同號且顯著（t={t:+.2f}）→ 有樣本外證據")
    gc = gross_combined
    if gc is not None:
        pp = (gc.annual_return - combined.annual_return) * 100.0
        if pp > 3:
            lines.append(
                f"- 成本侵蝕 {pp:.1f}pp/年（淨 {combined.annual_return:+.1%} vs 免成本 {gc.annual_return:+.1%}）："
                "顯著，優先拉長再平衡間隔或加大緩衝帶"
            )
        elif pp > 0:
            lines.append(f"- 成本侵蝕 {pp:.1f}pp/年：影響有限")
        else:
            lines.append("- 免成本年化未高於淨年化（無成本侵蝕問題）")
    if bench_daily is not None and len(combined.daily) and len(bench_daily):
        port_stats = _yearly_stats(combined.daily)
        bench_stats = _yearly_stats(bench_daily)
        excess = {
            y: v[0] - bench_stats.get(y, (float("nan"),))[0] for y, v in port_stats.items()
        }
        excess = {y: e for y, e in excess.items() if np.isfinite(e)}
        if excess:
            y_best = max(excess, key=lambda y: excess[y])
            lines.append(
                f"- 報酬集中度：超額報酬最集中於 {y_best}（{excess[y_best]:+.1%}）；"
                "若單一年度貢獻過半，需警惕對特定行情的依賴"
            )
    if combined.max_drawdown > bench["max_drawdown"]:
        lines.append(
            f"- 回撤：策略 MDD {combined.max_drawdown:.1%} **高於**基準 {bench['max_drawdown']:.1%}，"
            "風險調整後優勢需審視"
        )
    else:
        lines.append(
            f"- 回撤：策略 MDD {combined.max_drawdown:.1%} 低於基準 {bench['max_drawdown']:.1%}"
        )
    return lines


def build_report(
    results: list[PortfolioResult],
    combined: PortfolioResult,
    bench: dict[str, float],
    direction_used: dict[str, str],
    params: dict[str, str],
    bench_daily: pd.Series | None = None,
    gross_results: list[PortfolioResult] | None = None,
    gross_combined: PortfolioResult | None = None,
    ic_stability_map: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    """組裝 Markdown 回測報告。

    ``gross_results``／``gross_combined``：同訊號零成本對照（量化成本侵蝕）；
    ``ic_stability_map``：{因子: (IS IC, OOS IC, OOS t 值)}（檢驗 auto 方向的穩健性）。
    """
    lines = [
        "# fin_factor 台股投組回測報告",
        "",
        f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **評估準則：一律以含成本（淨）績效為準**；免成本結果僅供訊號有效性驗證",
        "- 成本模型（純現金）：手續費單邊 commission、賣出證交稅 tax、滑價 slippage"
        "（台股預設來回 0.585% = 0.1425%×2 + 0.3%；借貸成本不計）",
        "- 無前視偏差：t 日收盤後訊號，t+1 日才開始計損益（kstock 回測引擎保證）",
        "- 報酬口徑：**復權價**（$close × $factor，向後調整、含股息；除權息不再視為虧損；"
        "殘留減資/分割復權跳動為已知限制）",
    ]
    if any("（auto" in str(v) for v in direction_used.values()):
        lines.append(
            "- ⚠️ 方向=auto 為**全樣本 in-sample 判斷**（前視）：OOS IC 對照見「成本與穩健性對照」，"
            "正式評估請用 `--oos 0.7` 或明確指定 --direction top/bottom"
        )
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
        f"| 基準（全標的等權、每日再平衡，免成本） | - | {bench['total_return']:+.1%} | {bench['annual_return']:+.1%} "
        f"| {bench['sharpe']:.2f} | {bench['max_drawdown']:.1%} | {bench['volatility']:.1%} | - | - | - |"
    )
    lines += _robustness_section(results, combined, gross_results, gross_combined, ic_stability_map)
    lines += _yearly_section(
        [(r.factor_name, r.daily) for r in results] + [(combined.factor_name, combined.daily)],
        bench_daily,
    )
    lines += _interpretation(combined, bench, bench_daily, gross_results, gross_combined, ic_stability_map)
    return "\n".join(lines) + "\n"


def _price_wides(price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """daily_pv 長表 → (復權收盤 wide, 原始收盤 wide)。

    回測與 IC 一律以**復權價**（$close × $factor，向後調整、含股息）計算報酬——
    除息日原始價自然回落不會被誤判為虧損；交易明細的成交價則用原始價（實際金額）。
    兩個 wide 套用**同一個**缺價遮罩（$close ≤ 0 停牌誤植、非有限值），
    保證逐標的壓縮序列完全一致，明細與回測引擎可逐筆對帳。
    """
    raw = price_df["$close"].unstack("instrument")
    if "$factor" in price_df.columns:
        adj = raw * price_df["$factor"].unstack("instrument")
    else:
        adj = raw
    invalid = (raw <= 0.0) | ~np.isfinite(adj) | (adj <= 0.0)
    return adj.mask(invalid), raw.mask(invalid)


def _prepare_universe(
    factor_names: tuple[str, ...] | None,
    data: Path | None,
    top_symbols: int | None,
    workdir: Path | None,
    workspace: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """載入（或重跑）因子值與價格，回傳 (因子 wides, 價格長表, 復權 close wide, 原始 close wide, 資料描述)。"""
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
    close_wide, raw_close_wide = _price_wides(price_df)
    return factor_wides, price_df, close_wide, raw_close_wide, desc


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
    trades: bool = False,
) -> Path:
    """執行回測並寫出報告，回傳報告路徑。

    ``data``／``top_symbols``：指定價格資料集（或在資料集內取前 N 檔子集）時，
    因子值會在**同一宇宙**上重跑（result.h5 的既有因子值僅涵蓋 debug 20 檔）。
    ``trades``：額外輸出實際進出倉明細 CSV（``trades_<因子>.csv``，與報告同目錄），
    並在報告尾端附摘要；方向沿用回測解析結果（auto 亦同），確保明細與淨值曲線一致。
    """
    factor_wides, price_df, close_wide, raw_close_wide, data_desc = _prepare_universe(
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
    # 免成本對照（同訊號、零成本）：量化成本侵蝕；方向沿用淨值版以確保訊號一致
    gross_dirs = _resolve_directions(factor_wides, price_df, direction, fwd_days)
    gross_results, gross_combined, _gross_bench, _ = evaluate_portfolios(
        factor_wides,
        close_wide,
        top_n=top_n,
        rebalance_days=rebalance_days,
        direction=gross_dirs,
        buffer_n=buffer_n,
        commission=0.0,
        tax=0.0,
        slippage_rate=0.0,
        fwd_days=fwd_days,
    )
    stab = ic_stability(factor_wides, price_df, fwd_days=fwd_days)
    params = {
        "因子": "、".join(r.factor_name for r in results),
        "top_n": str(top_n),
        "再平衡間隔": f"{rebalance_days} 個交易日",
        "方向": direction,
        "緩衝帶": f"{buffer_n} 名（跌出才換）" if buffer_n else "無",
        "資料來源": data_desc,
        "價格口徑": "復權收盤（$close × $factor，含股息）；交易明細成交價為原始價",
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
        gross_results=gross_results,
        gross_combined=gross_combined,
        ic_stability_map=stab,
    )
    if trades:
        dirs_clean = _resolve_directions(factor_wides, price_df, direction, fwd_days)
        ledgers, tpaths = export_trade_ledgers(
            factor_wides,
            raw_close_wide,
            dirs_clean,
            top_n=top_n,
            rebalance_days=rebalance_days,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
            output_dir=output.parent,
            adj_close_wide=close_wide,
        )
        report += "\n".join(_ledger_summary_section(ledgers, tpaths)) + "\n"
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
    trades: bool = False,
) -> Path:
    """樣本外驗證：方向只用 IS 期間 IC 決定，OOS 期間以固定方向回測。

    切分：交易日序列前 ``is_ratio`` 為 IS、其餘為 OOS。因子皆為 trailing 計算，
    全期一次算完再切片不會引入前視；唯一 in-sample 元素（方向選擇）被隔離在 IS。
    ``trades``：額外輸出 **OOS 期間**的交易明細 CSV（``oos_trades_<因子>.csv``）。
    回傳報告路徑。
    """
    factor_wides, price_df, close_wide, raw_close_wide, data_desc = _prepare_universe(
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
        "；報酬以**復權價**計（$close × $factor，含股息）"
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
        f"| 基準（等權、每日再平衡，免成本） | - | - | {oos_bench['sharpe']:.2f} | {oos_bench['max_drawdown']:.1%} | - | - | - | - | - |"
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
    if trades:
        ledgers, tpaths = export_trade_ledgers(
            oos_fws,
            raw_close_wide.loc[oos_dates],
            directions,
            top_n=top_n,
            rebalance_days=rebalance_days,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
            output_dir=out_path.parent,
            prefix="oos_",
            adj_close_wide=close_wide.loc[oos_dates],
        )
        lines += _ledger_summary_section(ledgers, tpaths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_execution_timing_report(
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
    """執行時點對照報告：同一訊號以 t 收盤／t+1 開盤／t+1 收盤三種成交價回測。

    診斷回測隱含的「t 日收盤價成交」（台股盤後定價口徑）退化到次日用什麼價成交時，
    edge 還剩多少——若 alpha 集中在 t→t+1 隔夜/開盤反應段，策略對「收盤後消息、價已先動」
    最脆弱，實務執行必須靠盤後定價或開盤卡位。回傳報告路徑。
    """
    factor_wides, price_df, close_wide, raw_close_wide, data_desc = _prepare_universe(
        factor_names, data, top_symbols, workdir, workspace=workspace
    )
    dirs_clean = _resolve_directions(factor_wides, price_df, direction, fwd_days)
    if any(v == "short" for v in dirs_clean.values()):
        raise ValueError(
            "exec-timing 不支援 direction=short（做空與執行時點對照為獨立診斷；請用 top/bottom/auto）"
        )
    raw_open_wide = price_df["$open"].unstack("instrument")
    if "$factor" in price_df.columns:
        adj_open_wide = raw_open_wide * price_df["$factor"].unstack("instrument")
    else:
        adj_open_wide = raw_open_wide

    rows: list[dict] = []
    for name, fw in factor_wides.items():
        for r in execution_timing_variants(
            fw,
            close_wide,
            raw_close_wide,
            adj_open_wide,
            raw_open_wide,
            direction=dirs_clean[name],
            top_n=top_n,
            rebalance_days=rebalance_days,
            buffer_n=buffer_n,
            commission=commission,
            tax=tax,
            slippage_rate=slippage_rate,
        ):
            r["因子"] = name
            rows.append(r)

    out_path = output or (kstock_settings.data_dir / "qlab" / "exec_timing_report.md")
    params = {
        "因子": "、".join(factor_wides),
        "top_n": str(top_n),
        "再平衡間隔": f"{rebalance_days} 個交易日",
        "方向（auto 為 in-sample）": "、".join(f"{k}={v}" for k, v in dirs_clean.items()),
        "緩衝帶": f"{buffer_n} 名（跌出才換）" if buffer_n else "無",
        "資料來源": data_desc,
        "手續費（單邊）": f"{commission:.4%}",
        "證交稅（賣出）": f"{tax:.1%}",
        "滑價（單邊）": f"{slippage_rate:.4%}",
        "標的數": str(close_wide.shape[1]),
        "期間": f"{close_wide.index.min()} ~ {close_wide.index.max()}",
    }
    lines = [
        "# 執行時點對照報告",
        "",
        f"- 產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **目的**：回測隱含「t 日收盤價成交」（台股盤後定價 13:40–14:00 口徑，量能因子需收盤後才確定）。"
        "以**同一份訊號**比較三種成交時點，檢驗 alpha 是否集中在 t→t+1 的隔夜/開盤反應段——"
        "愈依賴隔夜段，對「收盤後消息、價已先動」愈脆弱",
        "- 成本：來回 "
        f"{2 * commission + tax:.3%}（手續費 {commission:.4%}×2 + 證交稅 {tax:.1%}）；"
        "報酬以**復權價**計（含股息）；三者的訊號、再平衡、緩衝帶、成本完全相同",
        "- ⚠️ 方向若為 auto 屬 in-sample 判斷；本報告為**執行風險診斷**，非正式績效評估",
        "",
        "## 參數",
        "",
    ]
    lines += [f"- {k}：{v}" for k, v in params.items()]
    lines += [
        "",
        "## 對照總表（含成本淨值）",
        "",
        "| 因子 | 執行時點 | 進場價 | 出場價 | 淨年化 | 夏普 | 最大回撤 | 進出場次數 | 單筆淨報酬平均 | 勝率 | 平均持有日 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['因子']}` | {r['執行時點']} | {r['進場價']} | {r['出場價']} "
            f"| {r['淨年化']:+.1%} | {r['夏普']:.2f} | {r['最大回撤']:.1%} | {r['進出場次數']} "
            f"| {r['單筆淨報酬平均']:+.2%} | {r['勝率']:.0%} | {r['平均持有日']:.1f} |"
        )
    # 隔夜段／日內段貢獻拆解與判讀
    seen: dict[str, dict[str, dict]] = {}
    for r in rows:
        seen.setdefault(r["因子"], {})[r["執行時點"]] = r
    lines += ["", "## 判讀", ""]
    for name, v in seen.items():
        v0 = v.get("t 收盤（盤後定價，回測假設）")
        v1 = v.get("t+1 開盤")
        v2 = v.get("t+1 收盤")
        if not (v0 and v1 and v2):
            continue
        overnight = (v0["淨年化"] - v1["淨年化"]) * 100.0
        intraday = (v1["淨年化"] - v2["淨年化"]) * 100.0
        lines.append(
            f"- `{name}`：**隔夜段**（t收盤 − t+1開盤）貢獻 {overnight:+.1f}pp/年、"
            f"**日內段**（t+1開盤 − t+1收盤）貢獻 {intraday:+.1f}pp/年"
        )
        share = v2["淨年化"] / v0["淨年化"] if v0["淨年化"] > 0 else float("nan")
        share_s = f"{share:.0%}" if np.isfinite(share) else "-"
        if np.isfinite(share) and share < 0.5:
            lines.append(
                f"  - ⚠️ t+1 收盤年化僅剩 {v2['淨年化']:+.1%}（保住 {share_s}）——"
                "alpha 高度依賴 t→t+1 段，實務必須靠盤後定價或開盤執行，且對收盤後消息脆弱，"
                "建議搭配事件排除驗證"
            )
        else:
            lines.append(
                f"  - t+1 收盤仍保有 {v2['淨年化']:+.1%}（保住 {share_s}）——執行時點風險有限"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

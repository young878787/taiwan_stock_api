"""向量化回測引擎（第一版：單標的，做多/空手/放空）。

避免前視偏差：t 日收盤後產生的訊號，最早 t+1 日才開始持倉。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: list[float]
    daily_returns: list[float]
    positions: list[int]
    total_return: float
    annual_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))


def run_backtest(
    signals,
    close,
    fee_rate: float = 0.001425,
    slippage_rate: float = 0.0,
    freq: int = TRADING_DAYS,
) -> BacktestResult:
    """訊號向量化回測。

    參數:
        signals: 每日收盤後的訊號（1=做多、0=空手、-1=放空），長度 = close 長度。
        close: 每日收盤價序列。
        fee_rate: 單邊手續費率（台股預設 0.1425%）。
        slippage_rate: 單邊滑價比例（於換倉日按換手量扣除，持有期間不計）。
        freq: 年化交易日數（預設 252）。
    """
    if not close:
        raise ValueError("close 不可以為空")
    prices = _floats(close)
    sig = [int(s) for s in signals]
    if len(sig) != len(prices):
        raise ValueError(f"signals 與 close 長度必須一致（{len(sig)} != {len(prices)}）")

    positions = [0] * len(prices)
    for i in range(1, len(sig)):
        positions[i] = sig[i - 1]

    strat: list[float] = [0.0] * len(prices)
    for i in range(1, len(prices)):
        if prices[i - 1] == 0.0:
            continue
        raw_return = prices[i] / prices[i - 1] - 1.0
        turnover = abs(positions[i] - positions[i - 1])
        # 成本只在換倉日按換手量扣除：fee 與滑價皆為單邊費率，來回 = 2×(fee+slippage)
        strat[i] = positions[i] * raw_return - turnover * (fee_rate + slippage_rate)

    equity: list[float] = [1.0]
    for r in strat[1:]:
        equity.append(equity[-1] * (1.0 + r))

    n = len(prices)
    total_return = equity[-1] - 1.0 if n > 1 else 0.0
    annual_return = (1.0 + total_return) ** (freq / n) - 1.0 if n > 2 else 0.0

    active = [r for r in strat if r != 0.0]
    vol = _std(active) * math.sqrt(freq)
    sharpe = (_mean(active) / _std(active) * math.sqrt(freq)) if _std(active) > 0 else 0.0

    peak = 1.0
    max_drawdown = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            max_drawdown = max(max_drawdown, 1.0 - e / peak)

    trade_count = 0
    for i in range(1, len(positions)):
        if positions[i] != positions[i - 1]:
            trade_count += 1

    winners = sum(1 for r in active if r > 0)
    win_rate = winners / len(active) if active else 0.0

    return BacktestResult(
        equity=equity,
        daily_returns=strat,
        positions=positions,
        total_return=total_return,
        annual_return=annual_return,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        win_rate=win_rate,
    )
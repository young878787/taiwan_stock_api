from __future__ import annotations


def back_adjusted_close(closes: list[float], event_ratios: dict[int, float]) -> list[float]:
    """後向還原收盤價。

    event_ratios: {event_index: ratio}，ratio = 事件後價格 / 事件前價格
    （例如 除權 2:1 → ratio=0.5；減資 1/... → ratio<1）。

    以「事件之後」的收盤價為基準，把「事件之前」的價格乘上，使其可比。
    """
    n = len(closes)
    if n == 0:
        return []
    scales = [1.0] * n
    acc = 1.0
    for i in range(n - 1, -1, -1):
        scales[i] = acc
        acc *= event_ratios.get(i, 1.0)
    out = []
    for price, scale in zip(closes, scales):
        out.append(round(price * scale, 2))
    return out


def forward_adjusted_close(closes: list[float], dividends: list[float]) -> list[float]:
    """前向還原收盤價（近似）：在該日支付則今日收盤價 = close - 累積已配息。

    dividends: {index: 現金股利金額}，假設除息後股價自然下跌。
    """
    n = len(closes)
    if n == 0:
        return []
    out: list[float] = []
    cumulative = 0.0
    for i in range(n):
        cumulative += dividends[i] if i in dividends else 0.0
        out.append(round(closes[i] - cumulative, 2))
    return out
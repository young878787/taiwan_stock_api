from kstock.normalizers.price import back_adjusted_close, forward_adjusted_close


def test_back_adjusted_no_events():
    assert back_adjusted_close([100, 105, 110], {}) == [100, 105, 110]


def test_back_adjust_split_2to1():
    closes = [200.0, 190.0, 100.0, 95.0]  # 第 2 日發生 2:1 分割
    adj = back_adjusted_close(closes, {2: 0.5})
    assert adj == [100.0, 95.0, 100.0, 95.0]


def test_back_adjust_clears_gap():
    closes = [100.0, 90.0, 45.0, 44.0]
    adj = back_adjusted_close(closes, {2: 0.5})
    assert adj[0] == 50.0
    assert adj[1] == 45.0


def test_forward_adjusted_subtracts_cumulative_dividend():
    closes = [100.0, 98.0, 99.0, 97.0]
    dividends = {1: 2.0}
    adjusted = forward_adjusted_close(closes, dividends)
    assert adjusted == [100.0, 96.0, 97.0, 95.0]


def test_forward_adjusted_multiple_dividends():
    closes = [100.0, 98.0, 99.0]
    dividends = {1: 1.0, 2: 0.5}
    adjusted = forward_adjusted_close(closes, dividends)
    assert adjusted == [100.0, 97.0, 97.5]


def test_back_original_vs_adjusted_order_preserved():
    raw = [90.0, 92.0, 88.0, 60.0, 61.0]
    adjusted = back_adjusted_close(raw, {3: 0.68})
    assert [round(a, 2) for a in adjusted] == [61.2, 62.56, 59.84, 60.0, 61.0]
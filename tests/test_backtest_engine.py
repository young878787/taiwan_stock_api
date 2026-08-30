import pytest

from kstock.backtest.engine import run_backtest


def test_no_signals_flat_equity():
    res = run_backtest([0] * 5, [100, 101, 103, 102, 105])
    assert res.total_return == pytest.approx(0.0, abs=1e-9)
    assert res.trade_count == 0
    assert all(e == pytest.approx(1.0) for e in res.equity)


def test_buy_and_hold_returns_price_move():
    close = [100, 101, 103, 103, 110]
    res = run_backtest([1] * 5, close, fee_rate=0.0)
    assert res.total_return == pytest.approx(110 / 100 - 1, rel=1e-6)
    assert res.equity[-1] == pytest.approx(1.10, rel=1e-6)


def test_no_lookahead_signal_taken_next_day():
    close = [100, 140, 130, 120, 110]
    res = run_backtest([1, 0, 0, 0, 0], close, fee_rate=0.0)
    assert res.positions[1] == 1  # 第 0 日的 1 訊號只在第1日起生效
    assert res.equity[1] == pytest.approx(1.4)
    assert res.positions[0] == 0
    assert res.positions[2] == 0


def test_fee_reduces_return():
    close = [100, 101, 103, 103, 110]
    no_fee = run_backtest([1, 1, 1, 1, 1], close, fee_rate=0.0)
    with_fee = run_backtest([1, 1, 1, 1, 1], close, fee_rate=0.001425)
    assert with_fee.total_return < no_fee.total_return


def test_slippage_charged_on_trade_day_only():
    """滑價只在換倉日按換手量扣一次，持有期間不得重複扣。"""
    close = [100, 101, 103, 103, 110]
    res = run_backtest([1] * 5, close, fee_rate=0.0, slippage_rate=0.001)
    # 進場日（第 1 日）扣一次 0.001，之後持有日不再扣
    assert res.equity[1] == pytest.approx(1.01 - 0.001)
    assert res.equity[-1] == pytest.approx((1.01 - 0.001) * (103 / 101) * (110 / 103))


def test_slippage_round_trip_two_sides():
    close = [100, 100, 100, 100, 100]
    res = run_backtest([1, 1, 1, 0, 0], close, fee_rate=0.0, slippage_rate=0.001)
    # 進場 + 出場各扣一次滑價，價格無變動
    assert res.total_return == pytest.approx((1 - 0.001) ** 2 - 1, rel=1e-9)


def test_no_slippage_without_trades():
    close = [100, 100, 100, 100, 100]
    res = run_backtest([0] * 5, close, fee_rate=0.0, slippage_rate=0.001)
    assert res.total_return == pytest.approx(0.0)


def test_trade_count_counts_position_changes():
    res = run_backtest([1, 0, 1, 0, 1], [100] * 5, fee_rate=0.0)
    assert res.trade_count == 4


def test_short_position_benefits_from_decline():
    close = [100, 95, 90, 85, 80]
    res = run_backtest([-1] * 5, close, fee_rate=0.0)
    assert res.equity[1] == pytest.approx(1.05)
    assert res.total_return == pytest.approx(1.2352941176 - 1, rel=1e-6, abs=1e-6)


def test_max_drawdown_computed():
    close = [100, 120, 60, 60, 120]
    res = run_backtest([1] * 5, close, fee_rate=0.0)
    assert res.equity[2] == pytest.approx(0.6)
    assert res.max_drawdown == pytest.approx(0.5, abs=1e-9)


def test_mismatched_length_raises():
    with pytest.raises(ValueError):
        run_backtest([1, 0], [100, 101, 102])


def test_sharpe_bounded_assets():
    res = run_backtest([1] * 5, [100, 101, 103, 103, 110])
    assert isinstance(res.sharpe, float)
    assert res.win_rate >= 0.0
    assert res.win_rate <= 1.0
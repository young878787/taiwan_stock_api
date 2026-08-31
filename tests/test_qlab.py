import numpy as np
import pandas as pd
import polars as pl
import pytest

from qlab.config import QlabSettings, qlab_settings
from qlab.export import QlibDataExporter

from conftest import make_daily_bars
from kstock.config.settings import Settings


def test_qlab_settings_inherits_kstock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    s = qlab_settings(Settings())  # 重新建構以讀取當下 env
    assert s.openai_api_key == "sk-test"
    assert s.deepseek_api_key == "ds-test"
    assert s.provider_dir.name == "qlib_data"


def test_qlab_provider_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("QLAB_PROVIDER_DIR", str(tmp_path / "custom_provider"))
    s = QlabSettings()
    assert s.provider_dir == tmp_path / "custom_provider"


def test_openrouter_key_inherits_and_runner_maps(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)

    s = qlab_settings(Settings())
    assert s.openrouter_api_key == "or-test"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"

    from qlab.rdagent_runner import build_rdagent_env

    env = build_rdagent_env(Settings())  # 重新建構以讀取當下 env
    assert env["OPENROUTER_API_KEY"] == "or-test"
    assert env["OPENAI_API_BASE"] == "https://openrouter.ai/api/v1"
    assert env["OPENAI_API_KEY"] == "or-test"
    assert env["CHAT_MODEL"] == "openai/gpt-4o-mini"


def test_export_writes_qlib_bin_layout(store, test_settings):
    store.write_normalized(
        "daily", make_daily_bars("2330", ["2024-01-02", "2024-01-03", "2024-01-04"], [590.0, 598.0, 601.0])
    )
    store.write_normalized(
        "daily", make_daily_bars("0050", ["2024-01-02", "2024-01-04"], [30.0, 30.5])
    )
    s = qlab_settings(test_settings)
    report = QlibDataExporter(s).export()

    assert report.n_symbols == 2
    assert report.n_calendar_days == 3

    # 交易日曆
    cal = (s.provider_dir / "calendars" / "day.txt").read_text().split()
    assert cal == ["2024-01-02", "2024-01-03", "2024-01-04"]

    # instruments（依符號排序，起訖為該符號的實際資料區間）
    lines = (s.provider_dir / "instruments" / "all.txt").read_text().strip().splitlines()
    assert lines[0] == "0050\t2024-01-02\t2024-01-04"
    assert lines[1] == "2330\t2024-01-02\t2024-01-04"

    # bin 檔格式：float32，[start_idx, 對齊日曆的值...]，缺日為 NaN
    raw = np.fromfile(s.provider_dir / "features" / "2330" / "close.day.bin", dtype="<f4")
    assert raw[0] == 0.0  # start index in calendar
    close = raw[1:]
    assert close[0] == pytest.approx(590.0)
    assert close[2] == pytest.approx(601.0)

    # 0050 缺 2024-01-03 → 對應位置應為 NaN
    raw_0050 = np.fromfile(s.provider_dir / "features" / "0050" / "close.day.bin", dtype="<f4")
    assert np.isnan(raw_0050[2])  # [0]=header，[1]=01-02，[2]=01-03
    assert raw_0050[3] == pytest.approx(30.5)

    # volume 欄位沿用「股」
    vol = np.fromfile(s.provider_dir / "features" / "2330" / "volume.day.bin", dtype="<f4")
    assert vol[1] == pytest.approx(1_000_000.0)


def test_export_raises_when_no_data(tmp_path):
    from kstock.config.settings import Settings

    s = qlab_settings(Settings(project_root=tmp_path, data_dir=tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        QlibDataExporter(s).export()


# ---------------------------------------------------------------------------
# factor_backtest：因子 → 投組回測
# ---------------------------------------------------------------------------


def _flat_close(n_days=10, n_sym=4, start=100.0):
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    cols = [f"S{i}" for i in range(n_sym)]
    return pd.DataFrame(start, index=idx, columns=cols, dtype=float)


def test_build_signals_rebalance_and_bottom_picks():
    from qlab.factor_backtest import build_signals

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    factor = pd.DataFrame(
        {"A": [1.0, 1, 1, 1, 1, 1], "B": [0.0, 0, 0, 0, 0, 0], "C": [-1.0, -1, -1, -1, -1, -1]},
        index=idx,
    )
    sig, turnover = build_signals(factor, top_n=1, rebalance_days=3, direction="bottom")

    # 每個再平衡窗（3 列）只選因子最小者：C → C
    assert (sig["C"] == 1.0).all()
    assert (sig["A"] == 0.0).all() and (sig["B"] == 0.0).all()
    # 兩次再平衡，第二次完全重疊 → 平均換倉比例 0
    assert turnover == pytest.approx(0.0)

    sig_top, _ = build_signals(factor, top_n=1, rebalance_days=3, direction="top")
    assert (sig_top["A"] == 1.0).all()
    assert (sig_top["C"] == 0.0).all()


def test_portfolio_backtest_no_lookahead():
    """改 t 日因子值，投組在 t+1 日（含）之前報酬不得改變（引擎延後一天生效）。"""
    from qlab.factor_backtest import portfolio_daily_returns

    close = _flat_close(n_days=10)
    close["S0"] = 100.0 * (1.10 ** np.arange(10))  # S0 每日 +10%
    close["S1"] = 100.0 * (0.90 ** np.arange(10))  # S1 每日 -10%
    factor = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    factor.iloc[5] = [0.0, 0.0, 0.0, 0.0]
    base = factor.copy()
    base.iloc[5, 0] = 1.0  # 第 5 日（再平衡日）持有 S0
    variant = factor.copy()
    variant.iloc[5, 1] = 1.0  # 改持有 S1

    daily_base, _ = portfolio_daily_returns(base, close, top_n=1, fee_rate=0.0)
    daily_var, _ = portfolio_daily_returns(variant, close, top_n=1, fee_rate=0.0)

    # 訊號在第 5 列產生，引擎 shift 一天 → 第 6 列才反映差異
    assert np.allclose(daily_base.iloc[:6].values, daily_var.iloc[:6].values)
    assert not np.isclose(daily_base.iloc[6], daily_var.iloc[6])
    assert daily_base.iloc[6] == pytest.approx(0.10)  # base 持有 S0
    assert daily_var.iloc[6] == pytest.approx(-0.10)  # variant 持有 S1


def test_portfolio_backtest_fees_reduce_return():
    from qlab.factor_backtest import portfolio_daily_returns

    close = _flat_close(n_days=12)
    close["S0"] = 100.0 * (1.05 ** np.arange(12))
    factor = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    factor["S0"] = 1.0  # 永遠持有 S0（進場一次）

    free, _ = portfolio_daily_returns(factor, close, top_n=1, fee_rate=0.0)
    costed, _ = portfolio_daily_returns(factor, close, top_n=1, fee_rate=0.002925)
    assert costed.sum() < free.sum()
    # 進場一次、之後無換倉 → 只在進場日扣一次費
    assert free.sum() - costed.sum() == pytest.approx(0.002925, rel=1e-6)


def test_evaluate_portfolios_smoke():
    from qlab.factor_backtest import evaluate_portfolios

    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, (60, 8)), axis=0), index=idx, columns=cols
    )
    factor = pd.DataFrame(rng.normal(size=(60, 8)), index=idx, columns=cols)

    results, combined, bench, dirs = evaluate_portfolios(
        {"f1": factor}, close, top_n=3, rebalance_days=5, direction="bottom"
    )
    assert len(results) == 1 and results[0].factor_name == "f1"
    assert dirs["f1"] == "bottom"
    assert combined.factor_name.startswith("組合")
    assert np.isfinite(combined.total_return)
    # 全部標的長期上漲 → 等權基準總報酬為正
    assert bench["total_return"] > 0
    # 換倉比例介於 0~1
    assert 0.0 <= results[0].avg_turnover <= 1.0


def test_build_signals_buffer_keeps_holding():
    """緩衝帶：持倉跌出 top_n 但仍在 buffer_n 內 → 續抱；跌出 buffer_n → 換股。"""
    from qlab.factor_backtest import build_signals

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    # A 最強 → C 最弱（bottom=買最小者 → C 排名 1）
    factor = pd.DataFrame(
        {"A": [9.0, 9, 9, 9], "B": [5.0, 5, 5, 5], "C": [1.0, 1, 1, 1], "D": [2.0, 2, 2, 2], "E": [3.0, 3, 3, 3]},
        index=idx,
    )
    # 無緩衝（top_n=1）：每次都抱 C，換倉 0
    sig0, t0 = build_signals(factor, top_n=1, rebalance_days=1, direction="bottom")
    assert (sig0["C"] == 1.0).all() and t0 == pytest.approx(0.0)

    # 有緩衝（top_n=1, buffer=3）：第 2 天 C 變成第 4 名（跌出緩衝帶）→ 換成 B
    f2 = factor.copy()
    f2.iloc[1:, :] = 0.0
    f2.iloc[1:, 0] = [9.0, 9.0, 9.0]  # A
    f2.iloc[1:, 1] = [3.0, 3.0, 3.0]  # B 第 2 名
    f2.iloc[1:, 2] = [8.0, 8.0, 8.0]  # C 掉到第 5 名（跌出緩衝帶 3）
    f2.iloc[1:, 3] = [1.0, 1.0, 1.0]  # D 第 1 名
    f2.iloc[1:, 4] = [2.0, 2.0, 2.0]  # E
    sig1, t1 = build_signals(f2, top_n=1, rebalance_days=1, direction="bottom", buffer_n=3)
    assert sig1["C"].iloc[0] == 1.0  # 第一個再平衡抱 C
    assert sig1["D"].iloc[1] == 1.0  # C 跌出緩衝帶 → 換成排名 1 的 D
    assert t1 == pytest.approx(1.0 / 3.0)  # 3 次有前倉的再平衡只換 1 次

    # 緩衝帶內（C 只掉到第 3 名）→ 續抱，換倉 0
    f3 = factor.copy()
    f3.iloc[1:, 2] = [4.0, 4.0, 4.0]  # C 值變 4 → 排名第 3（D=1、E=2、C=3），仍在緩衝帶 3
    sig2, t2 = build_signals(f3, top_n=1, rebalance_days=1, direction="bottom", buffer_n=3)
    assert (sig2["C"] == 1.0).all()
    assert t2 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# factor_backtest：交易明細表（--trades）
# ---------------------------------------------------------------------------


def test_trade_ledger_round_trip_matches_engine():
    """明細表的進出場時點、價格與淨報酬必須與回測引擎逐槽一致。"""
    from kstock.backtest.engine import run_backtest

    from qlab.factor_backtest import build_trade_ledger

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 105], "B": [200.0] * 6}, index=idx)
    # bottom：第 0 日選 A（1<2）、第 3 日換 B（A 變 2、B 變 1）
    factor = pd.DataFrame({"A": [1.0, 1, 1, 2, 2, 2], "B": [2.0, 2, 2, 1, 1, 1]}, index=idx)
    led = build_trade_ledger(factor, close, direction="bottom", top_n=1, rebalance_days=3)
    assert len(led) == 2
    a = led[led["symbol"] == "A"].iloc[0]
    b = led[led["symbol"] == "B"].iloc[0]

    # A：第 0 日訊號買進（收盤 100）、第 3 日訊號賣出（收盤 103）
    assert a["status"] == "closed"
    assert a["entry_date"] == idx[0] and a["exit_date"] == idx[3]
    assert a["entry_price"] == pytest.approx(100.0) and a["exit_price"] == pytest.approx(103.0)
    assert a["holding_days"] == 3
    assert a["rank_at_entry"] == pytest.approx(1.0) and a["universe_size_at_entry"] == 2
    assert a["gross_return"] == pytest.approx(103 / 100 - 1)
    assert "跌出前 1 名" in a["exit_reason"]

    # 淨報酬與引擎逐槽權益一致（進場日 1+r−fee、出場日 1−fee）
    fee = 0.001425 + 0.003 / 2
    engine_a = run_backtest([1, 1, 1, 0, 0, 0], [100.0, 101, 102, 103, 104, 105], fee_rate=fee)
    assert a["net_return"] == pytest.approx(engine_a.equity[4] - 1.0)
    # 滑價計入單邊費率（與引擎 fee+slippage 一致）
    led_slip = build_trade_ledger(
        factor, close, direction="bottom", top_n=1, rebalance_days=3, slippage_rate=0.001
    )
    engine_slip = run_backtest(
        [1, 1, 1, 0, 0, 0], [100.0, 101, 102, 103, 104, 105], fee_rate=fee, slippage_rate=0.001
    )
    assert led_slip[led_slip["symbol"] == "A"].iloc[0]["net_return"] == pytest.approx(
        engine_slip.equity[4] - 1.0
    )

    # B：第 3 日訊號買進、樣本結束仍持倉（標記價 = 最後收盤、無出場成本）
    assert b["status"] == "open"
    assert pd.isna(b["exit_date"])
    assert b["entry_date"] == idx[3] and b["entry_price"] == pytest.approx(200.0)
    assert b["holding_days"] == 2
    engine_b = run_backtest([0, 0, 0, 1, 1, 1], [200.0] * 6, fee_rate=fee)
    assert b["net_return"] == pytest.approx(engine_b.equity[-1] - 1.0)

    # 筆數與引擎換倉次數對帳：已平倉 ×2 + 持倉中 ×1
    from qlab.factor_backtest import build_signals, portfolio_daily_returns

    sig, _ = build_signals(factor, top_n=1, rebalance_days=3, direction="bottom")
    _, total_trades = portfolio_daily_returns(sig, close, top_n=1, fee_rate=fee)
    assert total_trades == 2 * int((led["status"] == "closed").sum()) + int(
        (led["status"] == "open").sum()
    )


def test_trade_ledger_buffer_no_churn():
    """緩衝帶內續抱不應產生交易列（一筆長持有的 open 交易）。"""
    from qlab.factor_backtest import build_trade_ledger

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [10.0] * 6, "B": [20.0] * 6, "C": [30.0] * 6}, index=idx)
    # 第 3 日 A 掉到第 2 名（仍在緩衝帶 3 內）→ 續抱；B 雖是第 1 名也不補進（slot 未空出）
    factor = pd.DataFrame(
        {"A": [1.0, 1, 1, 2, 2, 2], "B": [3.0, 3, 3, 1, 1, 1], "C": [2.0, 2, 2, 3, 3, 3]},
        index=idx,
    )
    led = build_trade_ledger(
        factor, close, direction="bottom", top_n=1, rebalance_days=3, buffer_n=3
    )
    assert len(led) == 1
    assert led.iloc[0]["symbol"] == "A" and led.iloc[0]["status"] == "open"
    assert led.iloc[0]["holding_days"] == 5


def test_trade_ledger_skips_signal_on_last_row():
    """訊號落在樣本末列時引擎不會建立部位，明細不得出現該筆交易。"""
    from qlab.factor_backtest import build_trade_ledger

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [10.0] * 4, "B": [20.0] * 4}, index=idx)
    factor = pd.DataFrame({"A": [1.0, 1, 1, 2.0], "B": [2.0, 2, 2, 1.0]}, index=idx)
    led = build_trade_ledger(factor, close, direction="bottom", top_n=1, rebalance_days=3)
    assert set(led["symbol"]) == {"A"}
    a = led.iloc[0]
    assert a["status"] == "closed" and a["exit_date"] == idx[3]


def test_trade_ledger_exit_reason_missing_factor():
    """再平衡日因子值缺失（停牌/壞資料）→ 被剔出選股，出場原因需標註。"""
    from qlab.factor_backtest import build_trade_ledger

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [10.0] * 6, "B": [20.0] * 6}, index=idx)
    factor = pd.DataFrame(
        {"A": [1.0, 1, 1, np.nan, np.nan, np.nan], "B": [2.0, 2, 2, 1.0, 1, 1]}, index=idx
    )
    led = build_trade_ledger(factor, close, direction="bottom", top_n=1, rebalance_days=3)
    a = led[led["symbol"] == "A"].iloc[0]
    assert a["status"] == "closed"
    assert "因子值缺失" in a["exit_reason"]


def test_ledger_summary_section_lists_paths(tmp_path):
    from qlab.factor_backtest import _ledger_summary_section

    df = pd.DataFrame(
        [
            {"symbol": "A", "status": "closed", "net_return": 0.05, "holding_days": 3},
            {"symbol": "B", "status": "open", "net_return": -0.01, "holding_days": 2},
        ]
    )
    lines = _ledger_summary_section({"f1": df}, {"f1": tmp_path / "trades_f1.csv"})
    text = "\n".join(lines)
    assert "## 交易明細" in text
    assert "trades_f1.csv" in text
    assert "已平倉 1、持倉中 1" in text


# ---------------------------------------------------------------------------
# 復權口徑：報酬以 $close × $factor（含息）計算
# ---------------------------------------------------------------------------


def test_forward_returns_uses_adjusted_close():
    """除息日原始價下跌不應計為虧損：前瞻報酬須以 $close × $factor（含息）計算。"""
    from qlab.factor_report import forward_returns

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    price = pd.DataFrame(
        {
            "$close": [100.0, 100.0, 97.0, 98.0],  # 第 2 列之後為除息 3 元後的原始價
            "$factor": [0.97, 0.97, 1.0, 1.0],
            "instrument": ["TSE0000"] * 4,
        },
        index=idx,
    ).set_index("instrument", append=True)
    fwd = forward_returns(price, days=1)
    # 復權價 = [97, 97, 97, 98] → 除息日報酬 0（非 -3%）
    assert fwd.iloc[0, 0] == pytest.approx(0.0)
    assert fwd.iloc[1, 0] == pytest.approx(0.0)
    assert fwd.iloc[2, 0] == pytest.approx(98 / 97 - 1)

    # 無 $factor 欄（舊資料）→ 退回原始價
    fwd_raw = forward_returns(price.drop(columns="$factor"), days=1)
    assert fwd_raw.iloc[0, 0] == pytest.approx(100 / 100 - 1)
    assert fwd_raw.iloc[1, 0] == pytest.approx(97 / 100 - 1)  # 原始口徑會把除息當虧損


def test_price_wides_adjusted_and_same_mask():
    """復權/原始 wide 需共用同一缺價遮罩，保證明細與引擎壓縮序列一致。"""
    from qlab.factor_backtest import _price_wides

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    pdf = pd.DataFrame(
        {
            "$close": [100.0, 0.0, 50.0],  # 第 2 列停牌零列
            "$factor": [0.5, 1.0, 1.0],
            "instrument": ["TSE0000"] * 3,
        },
        index=idx,
    ).set_index("instrument", append=True)
    adj, raw = _price_wides(pdf)
    assert adj.iloc[0, 0] == pytest.approx(50.0)
    assert raw.iloc[0, 0] == pytest.approx(100.0)
    assert pd.isna(adj.iloc[1, 0]) and pd.isna(raw.iloc[1, 0])  # 同一遮罩

    # 無 $factor 欄 → 復權價退回原始價
    adj2, _raw2 = _price_wides(pdf.drop(columns="$factor"))
    assert adj2.iloc[0, 0] == pytest.approx(100.0)


def test_trade_ledger_dividend_credited_in_returns():
    """持有期間除息：成交價為原始價，但毛/淨報酬以復權價計（股息計入）。"""
    from kstock.backtest.engine import run_backtest

    from qlab.factor_backtest import build_trade_ledger

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    # A：第 2 列除息 3 元（原始價 100→97），factor 向後調整前兩列 0.97
    raw = pd.DataFrame({"A": [100.0, 100, 97, 98, 99, 100], "B": [200.0] * 6}, index=idx)
    fac = pd.DataFrame({"A": [0.97, 0.97, 1.0, 1.0, 1.0, 1.0], "B": [1.0] * 6}, index=idx)
    adj = raw * fac  # [97, 97, 97, 98, 99, 100]
    # bottom：第 0 日選 A（1<2）、第 3 日換 B
    factor = pd.DataFrame({"A": [1.0, 1, 1, 2, 2, 2], "B": [2.0, 2, 2, 1, 1, 1]}, index=idx)
    led = build_trade_ledger(
        factor, raw, direction="bottom", top_n=1, rebalance_days=3, adj_close_wide=adj
    )
    a = led[led["symbol"] == "A"].iloc[0]
    assert a["status"] == "closed"
    # 成交價為原始價
    assert a["entry_price"] == pytest.approx(100.0) and a["exit_price"] == pytest.approx(98.0)
    # 毛報酬以復權價計（97→98 含息 +1/97），非原始價差的 −2%
    assert a["gross_return"] == pytest.approx(98 / 97 - 1)
    assert a["gross_return"] > 0

    # 淨報酬與引擎（以復權價回測）逐槽一致
    fee = 0.001425 + 0.003 / 2
    engine_a = run_backtest([1, 1, 1, 0, 0, 0], [97.0, 97, 97, 98, 99, 100], fee_rate=fee)
    assert a["net_return"] == pytest.approx(engine_a.equity[4] - 1.0)


# ---------------------------------------------------------------------------
# 執行時點對照（exec-timing）：t 收盤 / t+1 開盤 / t+1 收盤
# ---------------------------------------------------------------------------


def test_open_exec_overrides_flips_and_prices():
    """V1 序列變換：進場訊號日收盤→次日開盤、出場訊號延長一天→當日開盤。"""
    from qlab.factor_backtest import _open_exec_overrides

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    sig = pd.Series([1.0, 1, 1, 0, 0, 1, 1, 0], index=idx)
    adj_open = pd.Series([10.0, 11, 12, 13, 14, 15, 16, 17], index=idx)
    raw_open = adj_open * 2.0

    flips, entry, exit_ = _open_exec_overrides(sig, adj_open, raw_open)
    # 第 1 個 run（0-2）：進場覆寫在第 0 日（值=次日開盤 11/22）
    assert entry[idx[0]] == pytest.approx((11.0, 22.0))
    # 出場訊號日在 idx[3]：延長一天 + 第 4 日覆寫為當日開盤（14/28）
    assert flips == [idx[3]]
    assert exit_[idx[4]] == pytest.approx((14.0, 28.0))
    # 第 2 個 run（5-6）：進場覆寫在第 5 日（值=次日開盤 16/32）
    assert entry[idx[5]] == pytest.approx((16.0, 32.0))
    # 出場訊號日在末列（idx[7]）：無次日開盤 → 不延長、不覆寫
    assert idx[7] not in flips and idx[7] not in entry and all(k != idx[7] for k in exit_)


def test_execution_timing_variants_overnight_capture():
    """漲幅全在隔夜段（open=close）時：t 收盤進場吃到跳空，t+1 開盤/收盤都吃不到。"""
    from qlab.factor_backtest import execution_timing_variants

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    close = pd.DataFrame(
        {"A": [100.0, 110, 110, 110, 110, 110, 110, 110], "B": [200.0] * 8}, index=idx
    )
    # 開盤=當日收盤：所有漲幅都發生在前一日收盤→當日開盤之間（純隔夜跳空）
    open_ = close.copy()
    # bottom：第 0 日選 A（1<2）、第 4 日換 B
    factor = pd.DataFrame({"A": [1.0, 1, 1, 1, 2, 2, 2, 2], "B": [2.0, 2, 2, 2, 1, 1, 1, 1]}, index=idx)
    rows = execution_timing_variants(
        factor,
        close,  # adj_close
        close,  # raw_close
        open_,  # adj_open
        open_,  # raw_open
        direction="bottom",
        top_n=1,
        rebalance_days=4,
        commission=0.0,
        tax=0.0,
    )
    by_name = {r["執行時點"]: r for r in rows}
    v0, v1, v2 = (
        by_name["t 收盤（盤後定價，回測假設）"],
        by_name["t+1 開盤"],
        by_name["t+1 收盤"],
    )
    # t 收盤進場（100）吃到 +10% 隔夜跳空；t+1 開盤/收盤進場（110）完全錯過
    assert v0["淨年化"] > v1["淨年化"] == v2["淨年化"]  # 免成本下 V1/V2 淨值全平
    assert v1["淨年化"] == pytest.approx(0.0, abs=1e-9)

    # 明細層驗證：進場價與毛報酬
    from qlab.factor_backtest import build_trade_ledger

    led0 = build_trade_ledger(
        factor, close, direction="bottom", top_n=1, rebalance_days=4, commission=0.0, tax=0.0,
        adj_close_wide=close,
    )
    a0 = led0[led0["symbol"] == "A"].iloc[0]
    assert a0["entry_price"] == pytest.approx(100.0) and a0["gross_return"] == pytest.approx(0.10)

    # V1：以變換後序列建明細（訊號延長一天、進場價=次日開盤）
    from qlab.factor_backtest import _open_exec_overrides

    sig = pd.DataFrame({"A": [1.0, 1, 1, 1, 0, 0, 0, 0], "B": [0.0, 0, 0, 0, 1, 1, 1, 1]}, index=idx)
    sig1 = sig.copy()
    c1 = close.copy()
    p1 = close.copy()
    flips, entry_ov, exit_ov = _open_exec_overrides(sig1["A"], open_["A"], open_["A"])
    assert flips == [idx[4]]
    assert entry_ov[idx[0]] == pytest.approx((110.0, 110.0))
    assert exit_ov[idx[5]] == pytest.approx((110.0, 110.0))
    for d in flips:
        sig1.loc[d, "A"] = 1.0
    for d, (va, vr) in entry_ov.items():
        c1.loc[d, "A"] = va
        p1.loc[d, "A"] = vr
    for d, (va, vr) in exit_ov.items():
        c1.loc[d, "A"] = va
        p1.loc[d, "A"] = vr
    led1 = build_trade_ledger(
        factor, p1, direction="bottom", top_n=1, rebalance_days=4, commission=0.0, tax=0.0,
        adj_close_wide=c1, signals=sig1,
    )
    a1 = led1[led1["symbol"] == "A"].iloc[0]
    # 進場價 = t+1 開盤 110；出場價 = 延長日開盤 110；全程 0 報酬（隔夜段被跳過）
    assert a1["entry_price"] == pytest.approx(110.0)
    assert a1["exit_price"] == pytest.approx(110.0)
    assert a1["gross_return"] == pytest.approx(0.0)
    assert a1["holding_days"] == 5  # 訊號延長一天 → 帳面持有 0..5


def test_execution_timing_variants_rejects_short():
    from qlab.factor_backtest import execution_timing_variants

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [100.0] * 6}, index=idx)
    with pytest.raises(ValueError):
        execution_timing_variants(
            pd.DataFrame({"A": [1.0] * 6}, index=idx),
            close, close, close, close,
            direction="short", top_n=1, rebalance_days=3,
        )


def test_dividend_factor_column_formula():
    """FinMind 除權息事件 → 向後調整因子：除權息日 adj 應平滑。"""
    from qlab.export_h5 import _dividend_factor_column

    dates = pd.date_range("2024-06-20", periods=10, freq="D")
    # S1：ex-day 2024-06-25 現金股利 2 元，除息前收盤 100 → f = 1 - 2/100 = 0.98
    closes = [100.0, 101.0, 99.0, 100.0, 100.0, 98.0, 99.0, 100.0, 101.0, 100.0]
    pdf = pd.DataFrame(
        {
            "datetime": list(dates) * 2,
            "instrument": ["TSE0001"] * 10 + ["TSE0002"] * 10,
            "close": closes + [50.0] * 10,
        }
    )
    events = {"TSE0001": [(pd.Timestamp("2024-06-25").date(), 2.0, 0.0)]}
    fac = _dividend_factor_column(pdf, events)

    got = fac[fac.index[:10]]
    assert got.iloc[-5:].tolist() == [1.0] * 5  # 除息日（含）之後 factor=1
    assert got.iloc[:5].tolist() == [0.98] * 5  # 除息日之前全部乘 0.98（向後調整）
    # adj = close × factor 在除息日平滑：100×0.98 = 98 = 98×1.0
    adj = pdf["close"].iloc[:10] * got.to_numpy()
    assert adj.iloc[4] == pytest.approx(adj.iloc[5])
    # 無事件標的全為 1
    assert (fac[fac.index[10:]] == 1.0).all()

    # 股票股利：配股率 0.1 → f = 1/1.1
    events2 = {"TSE0001": [(pd.Timestamp("2024-06-25").date(), 0.0, 0.1)]}
    fac2 = _dividend_factor_column(pdf, events2)
    assert fac2[fac2.index[0]] == pytest.approx(1 / 1.1)
    assert fac2[fac2.index[9]] == pytest.approx(1.0)


def test_auto_direction_uses_ic_sign():
    from qlab.factor_backtest import auto_direction

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    cols = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    # 讓 A..C 連續上漲、D..F 連續下跌 → 動能為正（因子大者未來漲）
    close.loc[:, ["A", "B", "C"]] = 100.0 * (1.01 ** np.arange(30))[:, None]
    close.loc[:, ["D", "E", "F"]] = 100.0 * (0.99 ** np.arange(30))[:, None]
    price_df = close.stack().rename("$close").rename_axis(["datetime", "instrument"]).to_frame()
    # 因子相對排序固定：A..C 恆為高值、D..F 恆為低值（與價格路徑方向一致）
    factor = pd.DataFrame(
        {c: (1.0 if c in ("A", "B", "C") else 0.0) for c in cols},
        index=idx,
    )
    direction, ic = auto_direction(factor, price_df, fwd_days=5)
    assert direction == "top"
    assert ic > 0


def test_rerun_factors_on_data(tmp_path):
    """在指定資料集上重跑 factor.py（RD-Agent 約定：讀 daily_pv.h5、寫 result.h5）。"""
    import textwrap

    from qlab.factor_backtest import rerun_factors_on_data

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    cols = [f"S{i}" for i in range(5)]
    price = pd.DataFrame(
        100.0 * (1.0 + 0.01 * np.sin(np.arange(30)))[:, None].repeat(5, axis=1),
        index=idx,
        columns=cols,
    )
    long = price.stack().rename("$close").rename_axis(["datetime", "instrument"]).to_frame()
    data_h5 = tmp_path / "daily_pv.h5"
    long.to_hdf(data_h5, key="data")

    # 假工作區：一個「LLM 生成」的 factor.py + 既有 result.h5（collect_results 靠它識別因子名）
    ws = tmp_path / "RD-Agent_workspace" / "abc123"
    ws.mkdir(parents=True)
    (ws / "factor.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def run():
                df = pd.read_hdf('daily_pv.h5')
                df['f'] = df.groupby(level='instrument')['$close'].pct_change()
                df[['f']].to_hdf('result.h5', key='data')

            if __name__ == '__main__':
                run()
            """
        ),
        encoding="utf-8",
    )
    prior = long.copy()
    prior["f"] = prior.groupby(level="instrument")["$close"].pct_change()
    prior[["f"]].to_hdf(ws / "result.h5", key="data")

    out = rerun_factors_on_data(
        ("f",), data_h5, tmp_path / "work", workspace=tmp_path / "RD-Agent_workspace"
    )
    assert "f" in out
    wide = out["f"]
    assert wide.shape == (30, 5)
    assert wide.notna().sum().sum() == 145  # 每檔第一日 pct_change 為 NaN（5 檔 × 29）


def test_forward_returns_masks_nonpositive_close():
    """close<=0（壞資料）視為缺價：前瞻報酬不得出現 inf（與回測端口徑一致）。"""
    from qlab.factor_report import forward_returns

    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    long = pd.DataFrame(
        {
            "datetime": list(idx) * 2,
            "instrument": ["A"] * 6 + ["B"] * 6,
            "$close": [10.0, 11.0, 0.0, 12.0, 13.0, 14.0] + [20.0] * 6,
        }
    ).set_index(["datetime", "instrument"])
    fwd = forward_returns(long, days=2)
    assert np.isinf(fwd.to_numpy()).sum() == 0
    # A 的 0 價日與其前 2 日（前瞻落入 0 價）皆為 NaN
    assert fwd["A"].isna().sum() >= 3


def test_ic_stability_split_signs():
    """ic_stability：IS/OOS 切分回傳 (IS IC, OOS IC, OOS t)；穩定因子兩段同號。"""
    from qlab.factor_backtest import ic_stability

    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    cols = ["A", "B", "C", "D", "E", "F"]
    # 各標的不同成長率（A 最強 → F 最弱），確保前瞻報酬排序穩定
    growth = np.array([1.010, 1.008, 1.006, 0.994, 0.992, 0.990])
    close = pd.DataFrame(
        100.0 * np.cumprod(np.tile(growth, (60, 1)), axis=0), index=idx, columns=cols
    )
    price_df = close.stack().rename("$close").rename_axis(["datetime", "instrument"]).to_frame()
    # 因子 = 排序基底 + 雜訊（IC 逐日變動但均值為正）
    rng = np.random.default_rng(0)
    base = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    factor = pd.DataFrame(
        rng.normal(0, 0.5, (60, 6)) + base, index=idx, columns=cols
    )

    stab = ic_stability({"f": factor}, price_df, fwd_days=5)
    is_ic, oos_ic, t = stab["f"]
    assert is_ic > 0 and oos_ic > 0  # 全期動能方向一致 → 兩段同號
    assert np.isfinite(t)


def test_resolve_directions_auto_and_fixed():
    from qlab.factor_backtest import _resolve_directions

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    cols = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    close.loc[:, ["A", "B", "C"]] = 100.0 * (1.01 ** np.arange(30))[:, None]
    close.loc[:, ["D", "E", "F"]] = 100.0 * (0.99 ** np.arange(30))[:, None]
    price_df = close.stack().rename("$close").rename_axis(["datetime", "instrument"]).to_frame()
    factor = pd.DataFrame({c: (1.0 if c in ("A", "B", "C") else 0.0) for c in cols}, index=idx)

    fixed = _resolve_directions({"f": factor}, None, "bottom", 5)
    assert fixed == {"f": "bottom"}
    auto = _resolve_directions({"f": factor}, price_df, "auto", 5)
    assert auto == {"f": "top"}  # 動能為正 → auto 選 top


def test_build_report_robustness_and_interpretation():
    """報告應含免成本對照、OOS IC 欄位、auto 警告與判讀區塊；基準標籤為每日再平衡。"""
    from qlab.factor_backtest import build_report, evaluate_portfolios

    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, (80, 8)), axis=0), index=idx, columns=cols
    )
    factor = pd.DataFrame(rng.normal(size=(80, 8)), index=idx, columns=cols)

    results, combined, bench, dirs = evaluate_portfolios(
        {"f": factor}, close, top_n=3, rebalance_days=5, direction="bottom"
    )
    gross_results, gross_combined, _gb, _gd = evaluate_portfolios(
        {"f": factor}, close, top_n=3, rebalance_days=5, direction="bottom",
        commission=0.0, tax=0.0,
    )
    stab = {"f": (0.05, -0.02, -0.8)}  # 刻意讓 OOS 與 IS 變號
    params = {"因子": "f", "top_n": "3", "標的數": "8"}

    report = build_report(
        results, combined, bench, dirs, params,
        bench_daily=close.pct_change(fill_method=None).mean(axis=1),
        gross_results=gross_results,
        gross_combined=gross_combined,
        ic_stability_map=stab,
    )
    assert "成本與穩健性對照" in report
    assert "免成本年化" in report and "成本侵蝕" in report
    assert "每日再平衡" in report and "買入持有" not in report
    assert "變號" in report  # OOS IC 與 IS 變號的判讀
    assert "報酬集中度" in report and "判讀" in report

    # auto 方向 → 報告頭部需有 in-sample 警告
    results_a, combined_a, bench_a, dirs_a = evaluate_portfolios(
        {"f": factor}, close, price_df=close.stack().rename("$close").rename_axis(
            ["datetime", "instrument"]).to_frame(),
        top_n=3, rebalance_days=5, direction="auto",
    )
    report_a = build_report(results_a, combined_a, bench_a, dirs_a, params)
    assert "in-sample" in report_a and "auto" in report_a


def test_evaluate_portfolios_short_negates_top():
    """direction=short：訊號取 top（因子最大者）、日報酬取負（做空）。

    S0 每日 -10%（因子最大）、S1 每日 +10% → 做空 S0 應獲利；
    免成本下 short 日報酬必須恰為 top（做多）日報酬的負值。
    """
    from qlab.factor_backtest import evaluate_portfolios

    n = 12
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.DataFrame({"S0": 100.0 * (0.90 ** np.arange(n)), "S1": 100.0 * (1.10 ** np.arange(n))}, index=idx)
    factor = pd.DataFrame({"S0": 1.0, "S1": 0.0}, index=idx)  # S0 恆為因子最大 → short 目標

    top_res, _c, _b, top_dirs = evaluate_portfolios(
        {"f": factor}, close, top_n=1, rebalance_days=n, direction="top", commission=0.0, tax=0.0
    )
    short_res, _c2, _b2, short_dirs = evaluate_portfolios(
        {"f": factor}, close, top_n=1, rebalance_days=n, direction="short", commission=0.0, tax=0.0
    )
    assert top_dirs["f"] == "top" and short_dirs["f"] == "short"
    assert np.allclose(short_res[0].daily.values, -top_res[0].daily.values)
    assert short_res[0].total_return > 0  # 做空下跌股獲利
    assert top_res[0].total_return < 0

    # 成本對稱性：short 淨損益 = -（top 淨損益）不成立（複利非線性），但進出次數應一致
    short_fee, _c3, _b3, _d3 = evaluate_portfolios(
        {"f": factor}, close, top_n=1, rebalance_days=n, direction="short", commission=0.001425, tax=0.003
    )
    assert short_fee[0].trade_count == top_res[0].trade_count


def test_build_signals_rejects_short():
    """short 不是 long-only 訊號層的概念：build_signals 應拒絕，由 evaluate_portfolios 處理。"""
    from qlab.factor_backtest import build_signals

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    factor = pd.DataFrame({"A": [1.0, 1, 1, 1], "B": [0.0, 0, 0, 0]}, index=idx)
    with pytest.raises(ValueError, match="top/bottom"):
        build_signals(factor, top_n=1, rebalance_days=2, direction="short")


def test_slice_top_symbols(tmp_path):
    """slice_top_symbols：依 instrument 排序切前 N 檔（OTC < TSE），與回測口徑一致。"""
    from qlab.export_h5 import slice_top_symbols

    src = tmp_path / "daily_pv.h5"
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["TSE2330", "OTC1234", "TSE0050", "OTC5678"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {"$open": 1.0, "$close": 1.0, "$high": 1.0, "$low": 1.0, "$volume": 1000.0, "$factor": 1.0}, index=idx
    )
    df.to_hdf(src, key="data")

    rep = slice_top_symbols(src=src, n=2, output_dir=tmp_path / "out")
    assert rep.n_symbols == 2 and rep.path.exists() and (rep.path.parent / "README.md").exists()
    out = pd.read_hdf(rep.path, key="data")
    got = sorted(out.index.get_level_values("instrument").unique())
    assert got == ["OTC1234", "OTC5678"]  # 排序前 2 檔（OTC1234 < OTC5678 < TSE…）
    assert len(out) == 4  # 2 檔 × 2 日


def test_slice_top_symbols_missing_src(tmp_path):
    from qlab.export_h5 import slice_top_symbols

    with pytest.raises(FileNotFoundError):
        slice_top_symbols(src=tmp_path / "nope.h5", n=100, output_dir=tmp_path / "out")


def test_build_rdagent_env_guidance_and_universe(test_settings, tmp_path):
    """--guidance short 注入假設生成類別覆寫；--universe tw100 需 tw100 資料集存在。"""
    from qlab.rdagent_runner import build_rdagent_env

    env = build_rdagent_env(test_settings, universe="full", guidance="short")
    assert env["QLIB_FACTOR_HYPOTHESIS_GEN"] == "qlab.short_factor_proposal.GuidedShortFactorHypothesisGen"
    # 無引導時不應注入
    env_plain = build_rdagent_env(test_settings, universe="full", guidance="none")
    assert "QLIB_FACTOR_HYPOTHESIS_GEN" not in env_plain

    # tw100 資料集不存在 → 明確報錯（先跑 export-tw100）
    with pytest.raises(FileNotFoundError):
        build_rdagent_env(test_settings, universe="tw100", guidance="short")

    # tw100 資料集存在 → DEBUG 目錄指到 tw100（fin_factor 因子執行固定用 DEBUG 目錄）
    qlab_data = tmp_path / "data" / "qlab"
    (qlab_data / "factor_source_data_tw").mkdir(parents=True)  # 需與 debug 目錄並存
    (qlab_data / "factor_source_data_tw" / "daily_pv.h5").write_bytes(b"x")
    tw100 = qlab_data / "factor_source_data_tw100"
    tw100.mkdir(parents=True)
    (tw100 / "daily_pv.h5").write_bytes(b"x")
    env_tw = build_rdagent_env(test_settings, universe="tw100", guidance="none")
    assert env_tw["FACTOR_COSTEER_DATA_FOLDER_DEBUG"].endswith("factor_source_data_tw100")


def test_short_factor_proposal_guidance_injection():
    """做空引導應附加在 hypothesis_specification 與 RAG，並硬排 volume_change_5d 反向。"""
    from qlab.short_factor_proposal import GuidedShortFactorHypothesisGen

    gen = GuidedShortFactorHypothesisGen(None)  # prepare_context 不用 scen，None 即可
    context, ok = gen.prepare_context(type("T", (), {"hist": []})())
    assert ok
    assert "SHORT-SELLING" in context["hypothesis_specification"]
    assert "volume_change_5d" in context["hypothesis_specification"]
    assert "FORBIDDEN" in context["hypothesis_specification"]
    assert "SHORT-selling" in context["RAG"]

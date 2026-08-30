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

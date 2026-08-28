import numpy as np
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

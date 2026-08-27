import pytest

from kstock.config.settings import PROJECT_ROOT, Settings


def test_default_resolved_from_project_root():
    s = Settings()
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.normalized_dir == s.data_dir / "normalized"
    assert s.raw_dir == s.data_dir / "raw"


def test_env_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("KSTOCK_DATA_DIR", str(tmp_path / "custom"))
    s = Settings()
    assert s.data_dir == tmp_path / "custom"
    assert s.normalized_dir == tmp_path / "custom" / "normalized"


def test_constructor_values_honored_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("FINMIND_VOLUME_UNIT", raising=False)
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    monkeypatch.delenv("KSTOCK_DATA_DIR", raising=False)
    s = Settings(data_dir=tmp_path / "x", finmind_token="abc", finmind_volume_unit="shares")
    assert s.data_dir == tmp_path / "x"
    assert s.finmind_token == "abc"
    assert s.finmind_volume_unit == "shares"


def test_finmind_data_url(tmp_path):
    s = Settings(finmind_base_url="https://api.finmindtrade.com/api/v4")
    assert s.finmind_data_url == "https://api.finmindtrade.com/api/v4/data"


def test_keys_placeholder_allowed_empty():
    s = Settings()
    assert isinstance(s.shioaji_key, str)
    assert isinstance(s.finmind_token, str)
from kstock.adapters.twse import TWSEAdapter, roc_date_to_iso


def test_roc_date_to_iso():
    assert roc_date_to_iso("112/01/02") == "2023-01-02"
    assert roc_date_to_iso("113/12/31") == "2024-12-31"


def _day_payload() -> dict:
    return {
        "stat": "OK",
        "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"],
        "data": [
            ["112/01/02", "324123", "72,392,000,000", "590.00", "600.00", "585.00", "598.00", "+8.00", "18234"],
            ["112/01/03", "300000", "66,000,000,000", "600.00", "601.00", "590.00", "590.00", "-8.00", "15000"],
        ],
    }


def test_get_daily_bars_maps_and_converts(monkeypatch):
    adapter = TWSEAdapter(base_url="https://www.twse.com.tw")
    monkeypatch.setattr(adapter, "_get_json", lambda url, params: _day_payload())

    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df.height == 2
    assert df["source"].to_list() == ["twse", "twse"]
    assert df["date"].to_list() == [__date("2023-01-02"), __date("2023-01-03")]
    assert df["close"].to_list() == [598.0, 590.0]
    assert df["volume_shares"].to_list() == [324_123, 300_000]
    assert df["trade_count"].to_list() == [18_234, 15_000]
    assert df["turnover_twd"].to_list() == [72_392_000_000, 66_000_000_000]


def test_multi_month_iteration(monkeypatch):
    adapter = TWSEAdapter()

    def fake(url, params):
        date_param = params["date"]
        if date_param == "20240101":
            return _day_payload()
        return {"stat": "OK", "fields": ["日期", "成交股數"], "data": [["112/02/02", "10"]]}

    monkeypatch.setattr(adapter, "_get_json", fake)
    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-02-29")
    assert df.height == 3


def test_stat_not_ok_returns_empty(monkeypatch):
    adapter = TWSEAdapter()
    monkeypatch.setattr(adapter, "_get_json", lambda url, params: {"stat": "Not Found"})
    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df.height == 0


def __date(iso: str):
    from datetime import date

    return date(int(iso[:4]), int(iso[5:7]), int(iso[8:10]))
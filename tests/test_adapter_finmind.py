import datetime

from kstock.adapters.finmind import FinMindAdapter

import polars as pl


def _price_rows() -> list[dict]:
    return [
        {
            "stock_id": "2330",
            "date": "2024-01-02",
            "open": 590.0,
            "max": 600.0,
            "min": 585.0,
            "close": 598.0,
            "Trading_Volume": 12345,
            "Trading_money": 7_391_310_000,
            "Trading_tickets": 18234,
        },
        {
            "stock_id": "2330",
            "date": "2024-01-03",
            "open": 600.0,
            "max": 601.0,
            "min": 590.0,
            "close": 590.0,
            "Trading_Volume": 10000,
            "Trading_money": 5_950_000_000,
            "Trading_tickets": 15000,
        },
    ]


def _make(token: str = "demo-token", volume_unit: str = "lots") -> FinMindAdapter:
    return FinMindAdapter(token=token, volume_unit=volume_unit)


def test_get_daily_bars_normalized(monkeypatch):
    adapter = _make()
    monkeypatch.setattr(adapter, "_get_json", lambda params: _price_rows())

    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df.height == 2
    assert df["symbol"].to_list() == ["2330", "2330"]
    assert df["market"].to_list() == ["TSE", "TSE"]
    assert df["source"].to_list() == ["finmind", "finmind"]
    assert df["volume_shares"].to_list() == [12_345_000, 10_000_000]
    assert df["close"].to_list() == [598.0, 590.0]
    assert df["trade_count"].to_list() == [18_234, 15_000]
    assert df["date"].dtype == pl.Date


def test_get_daily_bars_empty_returns_typed_schema(monkeypatch):
    adapter = _make()
    monkeypatch.setattr(adapter, "_get_json", lambda params: [])
    df = adapter.get_daily_bars("9999", "2024-01-01", "2024-01-31")
    assert df.height == 0
    assert "volume_shares" in df.columns
    assert "source" in df.columns


def test_get_daily_bars_drops_suspension_zero_rows(monkeypatch):
    """FinMind 停牌日回傳 OHLC/量全 0 的列（非缺列）→ adapter 應略過，不入庫。"""
    adapter = _make()
    rows = [
        *_price_rows(),
        {
            "stock_id": "2330",
            "date": "2024-01-04",
            "open": 0.0,
            "max": 0.0,
            "min": 0.0,
            "close": 0.0,
            "Trading_Volume": 0,
            "Trading_money": 0,
            "Trading_tickets": 0,
        },
        {  # close 缺值（_to_float → 0）同樣視為無效列
            "stock_id": "2330",
            "date": "2024-01-05",
            "open": 500.0,
            "max": 505.0,
            "min": 495.0,
            "close": None,
            "Trading_Volume": 1000,
            "Trading_money": 500_000,
            "Trading_tickets": 100,
        },
    ]
    monkeypatch.setattr(adapter, "_get_json", lambda params: rows)
    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df.height == 2  # 只留兩個有效交易日
    assert df["close"].to_list() == [598.0, 590.0]
    assert df["date"].to_list() == [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]


def test_get_daily_bars_repairs_zero_open(monkeypatch):
    """open=0/負值但 high/low/close/量正常 → adapter 以 close 修補 open，不丟列。"""
    adapter = _make()
    rows = [
        *_price_rows(),
        {
            "stock_id": "2330",
            "date": "2024-01-04",
            "open": 0.0,
            "max": 605.0,
            "min": 595.0,
            "close": 600.0,
            "Trading_Volume": 8000,
            "Trading_money": 4_800_000,
            "Trading_tickets": 900,
        },
        {
            "stock_id": "2330",
            "date": "2024-01-05",
            "open": None,
            "max": 610.0,
            "min": 600.0,
            "close": 610.0,
            "Trading_Volume": 5000,
            "Trading_money": 3_000_000,
            "Trading_tickets": 600,
        },
    ]
    monkeypatch.setattr(adapter, "_get_json", lambda params: rows)
    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df.height == 4
    assert df["open"].to_list() == [590.0, 600.0, 600.0, 610.0]
    assert (df["open"] <= 0).sum() == 0


def test_volume_unit_shares_no_multiply(monkeypatch):
    adapter = _make(volume_unit="shares")
    monkeypatch.setattr(adapter, "_get_json", lambda params: _price_rows())
    df = adapter.get_daily_bars("2330", "2024-01-01", "2024-01-31")
    assert df["volume_shares"].to_list() == [12_345, 10_000]


def test_field_names_introspection(monkeypatch):
    adapter = _make()
    monkeypatch.setattr(adapter, "_get_json", lambda params: _price_rows())
    assert adapter.field_names("TaiwanStockPrice") == list(_price_rows()[0].keys())


def test_get_instruments(monkeypatch):
    """2026 改版後 TaiwanStockInfo 僅 5 欄（industry_category/stock_id/stock_name/type/date）：
    無 delist_date、date 是異動快照日≠list_date、type 含 emerging。"""
    adapter = _make()
    monkeypatch.setattr(
        adapter,
        "_get_json",
        lambda params: [
            {
                "industry_category": "半導體",
                "stock_id": "2330",
                "stock_name": "台積電",
                "type": "twse",
                "date": "2026-09-11",
            },
            {
                "industry_category": "ETF",
                "stock_id": "0050",
                "stock_name": "元大台灣50",
                "type": "tpex",
                "date": "2026-09-11",
            },
            {
                "industry_category": "生技業",
                "stock_id": "1234",
                "stock_name": "興櫃範例",
                "type": "emerging",
                "date": "2026-09-11",
            },
        ],
    )
    df = adapter.get_instruments()
    assert df.height == 3
    assert df.columns == [
        "symbol", "name", "market", "industry", "list_date", "delist_date", "status",
        "snapshot_date",
    ]
    rows = {r["symbol"]: r for r in df.iter_rows(named=True)}
    assert rows["2330"]["market"] == "TSE"
    assert rows["0050"]["market"] == "OTC"
    assert rows["1234"]["market"] == "EMG"
    assert rows["2330"]["name"] == "台積電"
    assert rows["2330"]["industry"] == "半導體"
    assert rows["2330"]["list_date"] is None  # date 是異動快照日，不可誤用為上市日
    assert rows["1234"]["list_date"] is None
    assert rows["1234"]["delist_date"] is None  # 欄位已從 dataset 消失，防禦性保留
    assert rows["1234"]["status"] == "active"
    assert rows["2330"]["snapshot_date"] == datetime.date(2026, 9, 11)  # date 保留為 snapshot_date


def test_get_instruments_empty(monkeypatch):
    adapter = _make()
    monkeypatch.setattr(adapter, "_get_json", lambda params: [])
    df = adapter.get_instruments()
    assert df.height == 0
    assert df["list_date"].dtype == pl.Date


def _institutional_rows() -> list[dict]:
    return [
        {
            "date": "2024-08-01",
            "stock_id": "2330",
            "Foreign_Investor_buy": 10_000_000,
            "Foreign_Investor_sell": 5_000_000,
            "Investment_Trust_buy": 800_000,
            "Investment_Trust_sell": 100_000,
            "Dealer_buy": 0,
            "Dealer_sell": 0,
            "Dealer_self_buy": 500_000,
            "Dealer_self_sell": 300_000,
            "Dealer_Hedging_buy": 200_000,
            "Dealer_Hedging_sell": 100_000,
        }
    ]


def test_get_institutional_maps_wide(monkeypatch):
    adapter = _make()
    monkeypatch.setattr(adapter, "_get_json", lambda params: _institutional_rows())
    df = adapter.get_institutional("2330", "2024-08-01", "2024-08-31")

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["symbol"] == "2330"
    assert row["date"].isoformat() == "2024-08-01"
    assert row["foreign_buy"] == 10_000_000
    assert row["foreign_sell"] == 5_000_000
    assert row["investment_trust_buy"] == 800_000
    assert row["dealer_buy"] == 700_000  # dealer_self + hedging
    assert row["dealer_sell"] == 400_000
    assert row["source"] == "finmind"


def test_get_margin_lots_to_shares(monkeypatch):
    inst = _make()
    rows = [
        {
            "date": "2024-08-01",
            "stock_id": "2330",
            "MarginPurchaseBuy": 1914,
            "MarginPurchaseSell": 1269,
            "MarginPurchaseTodayBalance": 26285,
            "ShortSaleBuy": 24,
            "ShortSaleSell": 398,
            "ShortSaleTodayBalance": 156,
        }
    ]
    monkeypatch.setattr(inst, "_get_json", lambda params: rows)
    df = inst.get_margin("2330", "2024-08-01", "2024-08-31")

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["margin_buy"] == 1_914_000
    assert row["margin_sell"] == 1_269_000
    assert row["margin_balance"] == 26_285_000
    assert row["short_sell"] == 398_000
    assert row["short_cover"] == 24_000
    assert row["short_balance"] == 156_000


def test_get_institutional_empty(monkeypatch):
    inst = _make()
    monkeypatch.setattr(inst, "_get_json", lambda params: [])
    df = inst.get_institutional("9999")
    assert df.height == 0
    assert df.columns == [
        "symbol", "date", "foreign_buy", "foreign_sell",
        "investment_trust_buy", "investment_trust_sell",
        "dealer_buy", "dealer_sell", "source",
    ]
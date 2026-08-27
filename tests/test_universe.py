import polars as pl

from kstock.universe.twse_whole_market import (
    WholeMarketQuotes,
    _extract_table,
    _num,
    select_top_liquid,
)


def test_num_parsing():
    assert _num("1,234.5") == 1234.5
    assert _num("---") is None
    assert _num("") is None
    assert _num("456") == 456.0


def test_extract_table_old_and_new_format():
    old = {"fields": ["證券代號", "證券名稱"], "data": [["2330", "台積電"]]}
    assert _extract_table(old)["fields"] == ["證券代號", "證券名稱"]

    new = {"tables": [{"fields": ["其他"], "data": []}, {"fields": ["證券代號"], "data": [["2330"]]}]}
    assert _extract_table(new)["data"] == [["2330"]]

    assert _extract_table({"stat": "OK"}) is None


def _quote_frame(symbol: str, name: str, turnover: float, close: float = 100.0, day: str = "2024-08-05") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "name": [name],
            "close": [close],
            "volume_shares": [10_000],
            "turnover_twd": [turnover],
            "source_date": [day],
        }
    )


def test_select_top_liquid_ranks_by_turnover():
    quotes = [
        pl.concat(
            [
                _quote_frame("2330", "台積電", 1e11),
                _quote_frame("0050", "台灣50", 5e9),
                _quote_frame("1101", "台泥", 1e8, close=30.0),
                _quote_frame("1111", "太便宜", 9e9, close=2.0),  # 低於 min_price
            ]
        )
    ]
    ranked = select_top_liquid(quotes, n=2)
    assert ranked["symbol"].to_list() == ["2330", "1101"]  # 0050(ETF前綴)與低價股被排除


def test_fetch_day_parses_rows(monkeypatch):
    adapter = WholeMarketQuotes()
    payload = {
        "stat": "OK",
        "tables": [
            {
                "fields": ["證券代號", "證券名稱", "成交股數", "成交金額", "收盤價"],
                "data": [
                    ["0050", "元大台灣50", "12,345,000", "1,234,500,000", "100.5"],
                    ["2330", "台積電", "50,257,000", "48,211,390,158", "960"],
                    ["----", "無效列", "", "", ""],
                ],
            }
        ],
    }
    monkeypatch.setattr(adapter, "_get_json", lambda url, params: payload)
    df = adapter.fetch_day("2024-08-05")
    assert df.height == 2
    row = df.filter(pl.col("symbol") == "2330").row(0, named=True)
    assert row["close"] == 960.0
    assert row["volume_shares"] == 50_257_000
    assert row["turnover_twd"] == 48_211_390_158
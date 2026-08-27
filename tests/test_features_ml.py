from datetime import date

import polars as pl
import pytest

from kstock.features.ml import FEATURE_ORDER, build_ml_features


def _daily(symbol: str, dates: list[date], closes: list[float], vols: list[int]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume_shares": vols,
            "turnover_twd": [float(v) * 30 for v in vols],
        }
    )


def _inst(symbol: str, dates: list[date], foreign_net_lots: list[float]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "foreign_buy": [max(0.0, x * 1000) for x in foreign_net_lots],
            "foreign_sell": [abs(min(0.0, x * 1000)) for x in foreign_net_lots],
            "investment_trust_buy": [1000.0] * n,
            "investment_trust_sell": [500.0] * n,
            "dealer_buy": [2000.0] * n,
            "dealer_sell": [1000.0] * n,
        }
    )


def _margin(symbol: str, dates: list[date], balance: list[float]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "margin_balance": balance,
            "short_balance": [10000.0] * n,
        }
    )


def _dates(n: int, start_day: int = 2) -> list[date]:
    return [date(2024, 1, start_day + i) for i in range(n)]


def test_feature_columns_and_sorting():
    d = _daily("2330", _dates(5), [100.0, 101.0, 102.0, 103.0, 104.0], [1_000_000] * 5)
    out = build_ml_features(d, institutional=None, margin=None)
    assert out.columns == FEATURE_ORDER
    assert out.height == 5
    # 無籌碼資料 → 全 null 但保留列
    assert out["foreign_net_lots"].is_null().all()
    assert out["margin_change_pct"].is_null().all()


def test_price_features_values():
    dates = _dates(6)
    closes = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    out = build_ml_features(_daily("2330", dates, closes, [1_000_000] * 6))
    r = out.row(1, named=True)
    assert r["return_1d"] == pytest.approx(0.10)
    label = out["label_ret_1f"].to_list()
    assert label[0] == pytest.approx(110 / 100 - 1)
    assert label[-1] is None  # 最後一天沒有次日報酬


def test_institutional_feature_ratio():
    dates = _dates(3)
    daily = _daily("2330", dates, [100.0] * 3, [1_000_000] * 3)
    inst = _inst("2330", dates, [50.0, -20.0, 0.0])
    out = build_ml_features(daily, inst, None)
    ratio = out["foreign_net_ratio"].to_list()
    assert ratio[0] == pytest.approx(50 * 1000 / 1_000_000)   # 淨買 5 萬股 / 百萬股
    assert ratio[1] == pytest.approx(-20 * 1000 / 1_000_000)
    tii = out["tii_net_lots"].to_list()
    # trust 固定淨買 500 股、dealer 淨買 1000 股 → 0.5 張 + 1 張 + 外資
    assert tii[0] == pytest.approx(50.0 + 0.5 + 1.0)


def test_margin_change_pct():
    dates = _dates(3)
    balances = [20_000.0, 30_000.0, 30_000.0]
    out = build_ml_features(
        _daily("2330", dates, [100.0] * 3, [1_000_000] * 3), None, _margin("2330", dates, balances)
    )
    mc = out["margin_change_pct"].to_list()
    assert mc[0] is None
    assert mc[1] == pytest.approx(0.5)
    assert mc[2] == pytest.approx(0.0)


def test_per_symbol_restart():
    dates = _dates(4)
    a = build_ml_features(
        _daily("2330", dates, [100.0, 101.0, 102.0, 103.0], [1_000_000] * 4),
        _inst("2330", dates, [1.0] * 4),
        None,
    )
    b = build_ml_features(
        _daily("2317", dates, [50.0, 55.0, 60.0, 65.0], [2_000_000] * 4),
        _inst("2317", dates, [-2.0] * 4),
        None,
    )
    merged = pl.concat([a, b])
    sub2317 = merged.filter(pl.col("symbol") == "2317")
    # 各標的各自的 pct_change / windows 從零開始
    assert sub2317["return_1d"].to_list()[0] is None
    assert sub2317["return_1d"].to_list()[1] == pytest.approx(55 / 50 - 1)
    assert sub2317["foreign_net_lots"].to_list()[0] == pytest.approx(-2.0)

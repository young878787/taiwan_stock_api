import polars as pl
import pytest

from kstock.features.technical import add_returns, add_rsi, add_sma, add_volume_ratio


def _d(y, m, day):
    from datetime import date

    return date(y, m, day)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["2330"] * 6,
            "date": [_d(2024, 1, i + 1) for i in range(6)],
            "close": [100.0, 101.0, 103.0, 103.0, 102.0, 106.0],
            "volume_shares": [1_000_000, 1_200_000, 900_000, 1_000_000, 1_000_000, 2_000_000],
        }
    )


def test_returns_one_frame():
    out = add_returns(_frame(), n=1)
    r = out["return_1d"].to_list()
    assert r[0] is None
    assert r[1] == pytest.approx(0.01)
    assert r[2] == pytest.approx(103 / 101 - 1)
    assert r[3] == pytest.approx(0.0)


def test_returns_per_symbol_restart():
    b = _frame().with_columns(
        pl.Series("symbol", ["0050"] * 6),
        (pl.col("close") + 200.0).alias("close"),
    )
    merged = pl.concat([_frame(), b])
    out = add_returns(merged, n=1)
    for sym, first, second in (("2330", 101 / 100 - 1, 0), ("0050", 301 / 300 - 1, 0)):
        sub = out.filter(pl.col("symbol") == sym)
        values = sub["return_1d"].to_list()
        assert values[0] is None
        assert values[1] == pytest.approx(first)


def test_sma_window():
    out = add_sma(_frame(), window=3)
    values = out["sma_3"].to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx((100 + 101 + 103) / 3)
    assert values[5] == pytest.approx((103 + 102 + 106) / 3)


def test_rsi_in_bounds_and_direction():
    up = add_rsi(_frame(), n=2)["rsi_2"].to_list()
    assert all(v is None or 0.0 <= v <= 100.0 for v in up)
    assert up[-1] > 50.0

    down = _frame().with_columns(pl.Series("close", [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]))
    down_rsi = add_rsi(down, n=2)["rsi_2"].to_list()
    assert down_rsi[-1] < 50.0


def test_rsi_flat_series_is_50():
    flat = _frame().with_columns(pl.Series("close", [100.0] * 6))
    rsi = add_rsi(flat, n=2)["rsi_2"].to_list()
    assert rsi[-1] == pytest.approx(50.0)


def test_volume_ratio():
    out = add_volume_ratio(_frame(), window=5)
    ratios = out["volume_ratio_5"].to_list()
    assert ratios[0] is None
    assert ratios[5] == pytest.approx(2_000_000 / ((1_200_000 + 900_000 + 1_000_000 + 1_000_000 + 2_000_000) / 5))
from conftest import make_daily_bars

from kstock.storage.parquet import ParquetStore


def test_write_and_read_partitioned(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0]))
    store.write_normalized("daily", make_daily_bars("2330", ["2025-02-04"], [600.0]))

    parts = store.normalized_files("daily")
    assert len(parts) == 2
    for p in parts:
        assert "year=2024" in str(p) or "year=2025" in str(p)

    read = store.read_normalized("daily")
    assert read.height == 3


def test_partition_dirs_use_hive_layout(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    folders = [p.parent.name for p in store.normalized_files("daily")]
    assert folders == ["year=2024"]


def test_read_normalized_symbol_and_date_filters(store: ParquetStore):
    store.write_normalized("daily", make_daily_bars("2330", ["2024-06-01", "2024-07-02"], [100.0, 101.0]))
    store.write_normalized("daily", make_daily_bars("0050", ["2024-06-15"], [30.0]))

    only_2330 = store.read_normalized("daily", symbols=["2330"])
    assert only_2330.height == 2

    window = store.read_normalized("daily", start_date="2024-06-15", end_date="2024-06-30")
    assert window.height == 1
    assert window["symbol"].to_list() == ["0050"]


def test_write_raw_appends_and_dedups(store: ParquetStore):
    store.write_raw("finmind", "daily", make_daily_bars("2330", ["2024-01-02"], [590.0]))
    store.write_raw(
        "finmind",
        "daily",
        make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [591.0, 592.0]),
    )
    raw = store.read_raw("finmind", "daily")
    assert raw.height == 2


def test_read_empty_table_returns_typed(store: ParquetStore):
    df = store.read_normalized("daily")
    assert df.height == 0
    assert "volume_shares" in df.columns
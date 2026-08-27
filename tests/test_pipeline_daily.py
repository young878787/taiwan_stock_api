import polars as pl

from conftest import make_daily_bars
from kstock.adapters.base import DataSourceAdapter
from kstock.pipeline.daily import DailyUpdatePipeline
from kstock.storage.duckdb import DuckStore
from kstock.storage.parquet import ParquetStore

INSTRUMENT_FRAME = pl.DataFrame(
    {
        "symbol": ["2330"],
        "name": ["台積電"],
        "market": ["TSE"],
        "industry": ["半導體"],
        "list_date": [None],
        "delist_date": [None],
        "status": ["active"],
    }
)


class FakeAdapter(DataSourceAdapter):
    name = "fake"

    def __init__(self, bars):
        self._bars = bars

    def get_daily_bars(self, symbol, start_date=None, end_date=None, market="TSE"):
        return self._bars

    def get_instruments(self):
        return IN


def test_pipeline_writes_to_parquet_and_duck(store: ParquetStore, duck: DuckStore):
    bars = make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0])
    pipeline = DailyUpdatePipeline(adapter=FakeAdapter(bars), store=store, duck=duck)
    results = pipeline.run(["2330"], "2024-01-01", "2024-01-31")

    assert len(results) == 1
    assert results[0].symbol == "2330"
    assert results[0].row_count == 2
    assert results[0].validation.passed is False

    assert len(store.normalized_files("daily")) == 1
    read = store.read_normalized("daily", symbols=["2330"])
    assert read.height == 2

    queried = duck.query("SELECT COUNT(*) AS n FROM daily WHERE symbol = '2330'")
    assert queried.row(0)[0] == 2


def test_pipeline_validates_against_secondary(store: ParquetStore):
    bars = make_daily_bars("2330", ["2024-01-02"], [590.0])
    other = make_daily_bars("2330", ["2024-01-02"], [595.0])
    pipeline = DailyUpdatePipeline(
        adapter=FakeAdapter(bars), store=store, validator_adapter=FakeAdapter(other)
    )
    results = pipeline.run(["2330"], "2024-01-01", "2024-01-31", validate=True)
    assert results[0].validation.passed is False
    assert results[0].validation.mismatches[0]["field"] == "close"


def test_pipeline_skip_raw_when_configured(store: ParquetStore):
    bars = make_daily_bars("2330", ["2024-01-02"], [590.0])
    pipeline = DailyUpdatePipeline(adapter=FakeAdapter(bars), store=store)
    results = pipeline.run(["2330"], "2024-01-01", "2024-01-31", write_raw=False)
    assert results[0].raw_path == ""
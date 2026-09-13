"""kstock.instruments CLI 離線測試（fake adapter + tmp_path 隔離 store）。"""

import polars as pl

from kstock import instruments
from kstock.models.schema import empty_dataframe
from kstock.storage.parquet import ParquetStore


def _instrument_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["2330", "0050"],
            "name": ["台積電", "元大台灣50"],
            "market": ["TSE", "TSE"],
            "industry": ["半導體", "ETF"],
            "list_date": [None, None],
            "delist_date": [None, None],
            "status": ["active", "active"],
        },
        schema_overrides={"list_date": pl.Utf8, "delist_date": pl.Utf8},
    )


class _FakeAdapter:
    def get_instruments(self) -> pl.DataFrame:
        return _instrument_df()


class _EmptyAdapter:
    def get_instruments(self) -> pl.DataFrame:
        return empty_dataframe("instrument")


def test_run_instruments_writes_and_rerun_idempotent(store: ParquetStore):
    n = instruments.run_instruments(adapter=_FakeAdapter(), store=store)
    assert n == 2
    target = store.normalized_dir / "instrument" / "data.parquet"
    assert target.exists()
    assert pl.read_parquet(target).height == 2

    # 重跑冪等：同 symbol upsert，不 append
    n2 = instruments.run_instruments(adapter=_FakeAdapter(), store=store)
    assert n2 == 2
    assert pl.read_parquet(target).height == 2


def test_run_instruments_empty_returns_zero(store: ParquetStore):
    assert instruments.run_instruments(adapter=_EmptyAdapter(), store=store) == 0
    assert not (store.normalized_dir / "instrument").exists()


def test_main_walkthrough(store: ParquetStore, monkeypatch, capsys):
    monkeypatch.setattr(instruments, "FinMindAdapter", lambda: _FakeAdapter())
    monkeypatch.setattr(instruments, "ParquetStore", lambda: store)

    rc = instruments.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2" in out  # 列數
    assert "TSE" in out  # market 分布


def test_main_empty_returns_zero(store: ParquetStore, monkeypatch):
    monkeypatch.setattr(instruments, "FinMindAdapter", lambda: _EmptyAdapter())
    monkeypatch.setattr(instruments, "ParquetStore", lambda: store)
    assert instruments.main([]) == 0


def test_main_exception_returns_nonzero(store: ParquetStore, monkeypatch):
    class _Boom:
        def get_instruments(self):
            raise RuntimeError("http 402 too many requests")

    monkeypatch.setattr(instruments, "FinMindAdapter", lambda: _Boom())
    monkeypatch.setattr(instruments, "ParquetStore", lambda: store)
    assert instruments.main([]) == 1

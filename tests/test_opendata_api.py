"""作業下載／匯入／HTTP／Client 整合測試，全部使用隔離檔案與 mock HTTP。"""

from datetime import date
import json
import sqlite3

from fastapi.testclient import TestClient
import httpx
import polars as pl
import pytest

from kstock.adapters.twse import normalize_daily_snapshot
from kstock.api.app import create_app
from kstock.api.client import StockAPIClient, run_demo
from kstock.api.models import DailyBar
from kstock.models.schema import TABLE_DTYPES
from kstock.pipeline.opendata import prepare
from kstock.storage.sqlite import SQLiteStore


@pytest.fixture
def raw_rows():
    return [{"Date": "1150924", "Code": "2330", "Name": "台積電", "TradeVolume": "1,000",
             "TradeValue": "102,000", "OpeningPrice": "100", "HighestPrice": "105",
             "LowestPrice": "99", "ClosingPrice": "102", "Change": "2", "Transaction": "10"}]


@pytest.fixture
def record(raw_rows):
    return DailyBar.model_validate(normalize_daily_snapshot(raw_rows).to_dicts()[0]).model_dump(mode="json")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "api.sqlite3")) as session:
        yield session


def test_snapshot_matches_existing_schema(raw_rows):
    frame = normalize_daily_snapshot(raw_rows)
    assert frame.schema == pl.Schema(TABLE_DTYPES["daily"])
    assert frame["date"].to_list() == [date(2026, 9, 24)]
    assert frame["volume_shares"].to_list() == [1000]
    assert frame["turnover_twd"].to_list() == [102000.0]


def test_snapshot_missing_values_are_null(raw_rows):
    raw_rows[0].update(TradeVolume="", OpeningPrice="--", HighestPrice="--", LowestPrice="--", ClosingPrice="--")
    bar = DailyBar.model_validate(normalize_daily_snapshot(raw_rows).to_dicts()[0])
    assert bar.volume_shares is None
    assert bar.close is None


@pytest.mark.parametrize("kind", ["empty", "object", "field", "date", "duplicate", "number"])
def test_reject_invalid_snapshot(raw_rows, kind):
    if kind == "empty":
        raw_rows = []
    elif kind == "object":
        raw_rows = {"error": "not available"}
    elif kind == "field":
        del raw_rows[0]["ClosingPrice"]
    elif kind == "date":
        raw_rows[0]["Date"] = "1150230"
    elif kind == "duplicate":
        raw_rows = raw_rows * 2
    else:
        raw_rows[0]["TradeVolume"] = "invalid"
    with pytest.raises(ValueError):
        normalize_daily_snapshot(raw_rows)


def test_crud_contract_and_restart(client, record, tmp_path):
    path = "/api/v1/daily-bars/2330/2026-09-24"
    created = client.post("/api/v1/daily-bars", json=record)
    assert created.status_code == 201
    assert created.headers["Location"] == path
    assert created.json() == record
    assert client.post("/api/v1/daily-bars", json=record).status_code == 409
    assert client.get(path).json() == record
    values = {key: value for key, value in record.items() if key not in ("symbol", "date")}
    values.update(close=103.0, source="manual")
    replaced = client.put(path, json=values)
    assert replaced.status_code == 200
    assert replaced.json()["close"] == 103.0
    with TestClient(create_app(tmp_path / "api.sqlite3")) as restarted:
        assert restarted.get(path).json()["source"] == "manual"
        assert restarted.delete(path).status_code == 204
        assert restarted.get(path).status_code == 404
        assert restarted.put(path, json=values).status_code == 404
        assert restarted.delete(path).status_code == 404
    with TestClient(create_app(tmp_path / "api.sqlite3")) as restarted:
        assert restarted.get("/api/v1/daily-bars").json()["total"] == 0


def test_pagination_filters_and_empty_page(client, record):
    for symbol, day in [("2330", "2026-09-24"), ("0050", "2026-09-24"), ("2330", "2026-09-23")]:
        assert client.post("/api/v1/daily-bars", json={**record, "symbol": symbol, "date": day}).status_code == 201
    page = client.get("/api/v1/daily-bars", params={"limit": 2}).json()
    assert page["total"] == 3
    assert [bar["symbol"] for bar in page["items"]] == ["0050", "2330"]
    page = client.get("/api/v1/daily-bars", params={"offset": 2, "limit": 2}).json()
    assert page["items"][0]["date"] == "2026-09-23"
    page = client.get("/api/v1/daily-bars", params={"symbol": "2330", "start_date": "2026-09-24", "end_date": "2026-09-24"}).json()
    assert page["total"] == 1
    assert client.get("/api/v1/daily-bars", params={"offset": 99}).json()["items"] == []


@pytest.mark.parametrize("field,value", [("volume_shares", -1), ("volume_shares", 1.5),
    ("volume_shares", True), ("volume_shares", 2**63), ("high", 90), ("low", 110),
    ("close", "NaN"), ("close", "Infinity"), ("turnover_twd", -1), ("trade_count", -1),
    ("date", "2026-02-30"), ("symbol", "x' OR 1=1"), ("source", "unknown"),
    ("market", "OTC"), ("extra", "unexpected")])
def test_invalid_body_does_not_write(client, record, field, value):
    assert client.post("/api/v1/daily-bars", json={**record, field: value}).status_code == 422
    assert client.get("/api/v1/daily-bars").json()["total"] == 0


def test_put_requires_complete_body_and_immutable_key(client, record):
    client.post("/api/v1/daily-bars", json=record)
    path = "/api/v1/daily-bars/2330/2026-09-24"
    assert client.put(path, json={"close": 103}).status_code == 422
    assert client.put(path, json=record).status_code == 422
    assert client.get(path).json() == record


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_number_returns_422(client, record, value):
    response = client.post("/api/v1/daily-bars", content=json.dumps({**record, "close": value}),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "close"]
    assert client.get("/api/v1/daily-bars").json()["total"] == 0


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}, {"offset": 2**63},
    {"symbol": "' OR 1=1"}, {"start_date": "invalid"},
    {"start_date": "2026-09-25", "end_date": "2026-09-24"}])
def test_invalid_queries(client, params):
    assert client.get("/api/v1/daily-bars", params=params).status_code == 422


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]["/api/v1/daily-bars"]) == {"get", "post"}
    assert set(spec["paths"]["/api/v1/daily-bars/{symbol}/{date}"]) == {"get", "put", "delete"}
    assert "409" in spec["paths"]["/api/v1/daily-bars"]["post"]["responses"]


def test_import_idempotent_update_and_transaction(tmp_path, record):
    store = SQLiteStore(tmp_path / "data.sqlite3")
    store.initialize()
    store.import_daily([record])
    store.import_daily([{**record, "close": 103}])
    assert store.list(None, None, None, 100, 0)["total"] == 1
    assert store.get(record["symbol"], record["date"])["close"] == 103
    with pytest.raises(sqlite3.IntegrityError):
        store.import_daily([{**record, "close": 104}, {**record, "symbol": "0050", "volume_shares": -1}])
    assert store.get(record["symbol"], record["date"])["close"] == 103
    assert store.get("0050", record["date"]) is None


def test_prepare_offline_and_update(tmp_path, raw_rows, monkeypatch):
    dataset = tmp_path / "twse.json"
    dataset.write_text(json.dumps(raw_rows), encoding="utf-8")
    def forbid_network(*args, **kwargs):
        raise AssertionError("離線匯入不得連線")
    monkeypatch.setattr(httpx.Client, "get", forbid_network)
    db, spec = tmp_path / "api.sqlite3", tmp_path / "openapi.json"
    result = prepare(dataset, db, openapi_path=spec)
    assert result["imported_rows"] == 1
    assert result["downloaded"] is False
    raw_rows[0]["ClosingPrice"] = "103"
    dataset.write_text(json.dumps(raw_rows), encoding="utf-8")
    prepare(dataset, db, openapi_path=spec)
    assert SQLiteStore(db).get("2330", "2026-09-24")["close"] == 103
    assert json.loads(spec.read_text(encoding="utf-8"))["paths"]


def test_download_and_bad_refresh_preserves_snapshot(tmp_path, raw_rows, monkeypatch):
    dataset, db, spec = tmp_path / "twse.json", tmp_path / "api.sqlite3", tmp_path / "openapi.json"
    monkeypatch.setattr(httpx.Client, "get", lambda *args, **kwargs: httpx.Response(
        200, json=raw_rows, request=httpx.Request("GET", "https://example.test")))
    assert prepare(dataset, db, openapi_path=spec)["downloaded"] is True
    original = dataset.read_bytes()
    raw_rows[0]["TradeVolume"] = "-1"
    with pytest.raises(ValueError):
        prepare(dataset, db, download=True, openapi_path=spec)
    assert dataset.read_bytes() == original
    assert SQLiteStore(db).get("2330", "2026-09-24")["volume_shares"] == 1000


def test_python_client_demo(client):
    examples = run_demo(StockAPIClient(client))
    assert [example["response"]["status_code"] for example in examples] == [200, 201, 200, 200, 409, 200, 200, 422, 204, 404]
    assert client.get("/api/v1/daily-bars").json()["total"] == 0


def test_python_client_cleans_up_after_failure(client, monkeypatch):
    api = StockAPIClient(client)
    monkeypatch.setattr(api, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("模擬更新失敗")))
    with pytest.raises(RuntimeError, match="模擬更新失敗"):
        run_demo(api)
    assert client.get("/api/v1/daily-bars").json()["total"] == 0

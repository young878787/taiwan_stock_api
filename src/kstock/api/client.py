"""可繳交的 Python API Client，執行 CRUD 示範並保存最新實測結果。"""

import argparse
import json
from uuid import uuid4

import httpx

from kstock.config.settings import PROJECT_ROOT


class StockAPIClient:
    def __init__(self, session: httpx.Client) -> None:
        self.session = session
        self.examples: list[dict] = []

    def request(self, method: str, path: str, expected_status: int, **kwargs):
        response = self.session.request(method, path, **kwargs)
        self.examples.append({
            "request": {"method": method, "path": path, "params": kwargs.get("params"), "data": kwargs.get("json")},
            "response": {"status_code": response.status_code,
                         "output": response.json() if response.content else None},
        })
        if response.status_code != expected_status:
            raise RuntimeError(f"{method} {path}: 預期 {expected_status}，實際 {response.status_code}")
        return response

    def create(self, record: dict):
        return self.request("POST", "/api/v1/daily-bars", 201, json=record)

    def read(self, symbol: str, date: str):
        return self.request("GET", f"/api/v1/daily-bars/{symbol}/{date}", 200)

    def list(self, **params):
        return self.request("GET", "/api/v1/daily-bars", 200, params=params)

    def replace(self, symbol: str, date: str, values: dict):
        return self.request("PUT", f"/api/v1/daily-bars/{symbol}/{date}", 200, json=values)

    def delete(self, symbol: str, date: str):
        return self.request("DELETE", f"/api/v1/daily-bars/{symbol}/{date}", 204)


def run_demo(client: StockAPIClient) -> list[dict]:
    """只操作本次建立的示範主鍵，失敗時也嘗試清理。"""
    symbol, date = "DEMO" + uuid4().hex[:8].upper(), "2000-01-01"
    values = {"market": "TSE", "open": 100.0, "high": 105.0, "low": 99.0,
              "close": 102.0, "volume_shares": 1000, "turnover_twd": 102000.0,
              "trade_count": 10, "source": "manual"}
    record = {"symbol": symbol, "date": date, **values}
    client.list(symbol="2330", limit=2)
    client.create(record)
    deleted = False
    try:
        client.read(symbol, date)
        client.list(symbol=symbol)
        client.request("POST", "/api/v1/daily-bars", 409, json=record)
        client.replace(symbol, date, {**values, "close": 103.0})
        client.read(symbol, date)
        client.request("PUT", f"/api/v1/daily-bars/{symbol}/{date}", 422,
                       json={**values, "volume_shares": -1})
        client.delete(symbol, date)
        deleted = True
        client.request("GET", f"/api/v1/daily-bars/{symbol}/{date}", 404)
    finally:
        if not deleted:
            client.delete(symbol, date)
    return client.examples


def main() -> None:
    parser = argparse.ArgumentParser(description="執行台股 API CRUD 實測，結果寫入 docs/api_examples.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API Server 位置")
    arguments = parser.parse_args()
    with httpx.Client(base_url=arguments.base_url, timeout=10) as session:
        client = StockAPIClient(session)
        passed = False
        try:
            run_demo(client)
            passed = True
        finally:
            target = PROJECT_ROOT / "docs" / "api_examples.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"passed": passed, "base_url": arguments.base_url,
                                          "examples": client.examples}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"CRUD 實測通過（{len(client.examples)} 個請求）：{target}")


if __name__ == "__main__":
    main()

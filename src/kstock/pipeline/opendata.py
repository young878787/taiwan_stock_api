"""單一 Open Data JSON 下載、驗證、匯入 SQLite，並匯出 API 規格。"""

import argparse
import json
from pathlib import Path

import httpx

from kstock.adapters.twse import DAILY_SNAPSHOT_URL, normalize_daily_snapshot
from kstock.api.app import create_app
from kstock.api.models import DailyBar
from kstock.config.settings import PROJECT_ROOT, settings
from kstock.storage.sqlite import SQLiteStore

DATASET_PATH = PROJECT_ROOT / "datasets" / "twse_daily.json"
OPENAPI_PATH = PROJECT_ROOT / "docs" / "openapi.json"


def prepare(dataset_path: Path = DATASET_PATH, db_path: Path | None = None,
            download: bool = False, openapi_path: Path = OPENAPI_PATH) -> dict:
    """預設重用繳交快照；只有明確更新或缺檔才連線。"""
    downloaded = download or not dataset_path.exists()
    if downloaded:
        with httpx.Client(timeout=30) as client:
            response = client.get(DAILY_SNAPSHOT_URL)
            response.raise_for_status()
            rows = response.json()
    else:
        rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    frame = normalize_daily_snapshot(rows)
    # 所有列先通過與 API 相同的驗證，避免匯入部分或不可回傳的資料。
    records = [DailyBar.model_validate(row).model_dump(mode="json") for row in frame.to_dicts()]
    if downloaded:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = dataset_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(dataset_path)
    target = db_path if db_path is not None else settings.data_dir / "kstock.sqlite3"
    store = SQLiteStore(target)
    store.initialize()
    count = store.import_daily(records)
    openapi_path.parent.mkdir(parents=True, exist_ok=True)
    openapi_path.write_text(json.dumps(create_app(target).openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dataset": str(dataset_path), "database": str(target), "imported_rows": count,
            "start_date": frame["date"].min().isoformat(), "end_date": frame["date"].max().isoformat(),
            "downloaded": downloaded, "openapi": str(openapi_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="下載／匯入 TWSE 日成交資訊並產生 OpenAPI 規格")
    parser.add_argument("--download", action="store_true", help="重新下載官方最新快照；預設使用既有檔案")
    arguments = parser.parse_args()
    print(json.dumps(prepare(download=arguments.download), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

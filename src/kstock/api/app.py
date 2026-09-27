"""FastAPI 日 K CRUD；啟動時只建表，不下載或重新匯入資料。"""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import sqlite3
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kstock.api.models import APIError, DailyBar, DailyPage, DailyValues, Symbol
from kstock.config.settings import settings
from kstock.storage.sqlite import SQLiteStore


def create_app(db_path: Path | None = None) -> FastAPI:
    store = SQLiteStore(db_path if db_path is not None else settings.data_dir / "kstock.sqlite3")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.initialize()
        yield

    app = FastAPI(
        title="台股日成交資訊 API",
        description="TWSE Open Data 的本機日 K 管理。CRUD 只修改 SQLite，不修改官方來源或研究用 Parquet。",
        version="1.0.0",
        lifespan=lifespan,
    )
    missing = {404: {"model": APIError, "description": "找不到日 K"}}

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error: RequestValidationError):
        # 不回傳原始 input / ctx，避免 NaN / Infinity 讓錯誤回應無法序列化。
        details = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/api/v1/daily-bars", response_model=DailyPage, tags=["日 K"], summary="查詢日 K 列表")
    def list_bars(
        symbol: Symbol | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
    ):
        if start_date and end_date and start_date > end_date:
            raise HTTPException(422, "start_date 不得晚於 end_date")
        return store.list(symbol, start_date.isoformat() if start_date else None,
                          end_date.isoformat() if end_date else None, limit, offset)

    @app.post("/api/v1/daily-bars", response_model=DailyBar, status_code=201, tags=["日 K"],
              summary="新增日 K", responses={409: {"model": APIError, "description": "自然主鍵重複"}})
    def create_bar(bar: DailyBar, response: Response):
        try:
            store.create(bar.model_dump(mode="json"))
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorname != "SQLITE_CONSTRAINT_PRIMARYKEY":
                raise
            raise HTTPException(409, "相同 symbol 與 date 的日 K 已存在") from error
        response.headers["Location"] = f"/api/v1/daily-bars/{bar.symbol}/{bar.date.isoformat()}"
        return bar

    @app.get("/api/v1/daily-bars/{symbol}/{date}", response_model=DailyBar, tags=["日 K"],
             summary="讀取單筆日 K", responses=missing)
    def get_bar(symbol: Symbol, date: date):
        bar = store.get(symbol, date.isoformat())
        if bar is None:
            raise HTTPException(404, "找不到日 K")
        return bar

    @app.put("/api/v1/daily-bars/{symbol}/{date}", response_model=DailyBar, tags=["日 K"],
             summary="完整替換日 K 內容", responses=missing)
    def replace_bar(symbol: Symbol, date: date, values: DailyValues):
        record = values.model_dump(mode="json")
        if not store.replace(symbol, date.isoformat(), record):
            raise HTTPException(404, "找不到日 K")
        return {**record, "symbol": symbol, "date": date.isoformat()}

    @app.delete("/api/v1/daily-bars/{symbol}/{date}", status_code=204, tags=["日 K"],
                summary="刪除單筆日 K", responses=missing)
    def delete_bar(symbol: Symbol, date: date):
        if not store.delete(symbol, date.isoformat()):
            raise HTTPException(404, "找不到日 K")
        return Response(status_code=204)

    return app


app = create_app()

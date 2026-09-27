# RESTful API 作業：分析與實作決策

## 1. 現況與範圍

現有 `kstock` 是 Python 資料管道，而非 HTTP Server：

- `adapters` 取得 FinMind、TWSE、Yahoo 資料，輸出 Polars DataFrame。
- `models/schema.py` 定義日 K 的 11 個標準欄位。
- `DailyUpdatePipeline` 將標準資料寫入 Parquet，再交由 DuckDB 查詢。
- Parquet 使用 `(symbol, date)` 合併同一天的資料；研究、特徵與回測讀取這一層。

作業需要可下載的單一 Open Data 檔、可修改的資料庫、HTTP CRUD、Swagger 與 Client，因此新增以下支線：

```text
TWSE STOCK_DAY_ALL → datasets/twse_daily.json（原始 JSON）
                             ↓
                normalize_daily_snapshot（Polars / 既有 daily schema）
                             ↓
                  DailyBar 全批資料驗證
                             ↓
                    SQLiteStore / daily
                             ↕
                 FastAPI → Swagger / Python Client
```

SQLite 與原有 Parquet 是獨立儲存層。API 修改只影響 SQLite；沒有雙寫、同步排程或回測資料回寫。這讓作業可以明確展示可變更資源，也避免示範 CRUD 汙染研究資料。若需要接入研究流程，可先使用 `normalize_daily_snapshot()` 的 DataFrame，但本次沒有加入額外同步機制。

## 2. 資料來源

選用政府資料開放平臺的「盤後資訊 > 個股日成交資訊」：

- [資料集與授權資訊](https://data.gov.tw/dataset/11549)：免費、每日更新，政府資料開放授權條款第 1 版。
- [TWSE 官方 OpenAPI](https://openapi.twse.com.tw/)：`GET /v1/exchangeReport/STOCK_DAY_ALL`。
- JSON 保留官方所有欄位，包含名稱與漲跌價差；API 使用既有日 K Schema，不另外新增股票主檔資源。
- 官方 `Date` 為民國年月日字串，轉成 ISO 日期；不以下載日代替交易日。
- 官方 `TradeVolume` 已是股，不乘 1000。逗號分隔數字會移除逗號後解析。
- 空字串與 `--` / `---` 轉為 `null`，保留缺值，不推論為 0。
- 檔案是來源回傳的全市場快照，可能包含 ETF 等證券；不是完整歷史資料庫，也不限制成只有四碼股票。

隨附 dump 為 `datasets/twse_daily.json`，共 1,380 筆、434,759 bytes。本機保存時間為 **2026-09-27 14:37:39（Asia/Taipei）**；官方 `Date` 欄只有 **民國 115 年 9 月 24 日（2026-09-24）**，因此資料涵蓋範圍是 `2026-09-24` 至 `2026-09-24`，即單一交易日，不是歷史區間。原始 OpenAPI JSON 沒有下載時間欄；README 記錄的是取得並保存 dump 的本機時間，交易日則依官方資料欄位換算。若需要歷史區間，需改用支援歷史查詢的資料來源或逐日彙整資料，不能把這份快照描述成歷史日 K。

GitHub 提交原始 JSON 作為可檢視、可重建的作業資料檔；SQLite 留作產生物。助教 clone 後執行 `uv run python -m kstock.pipeline.opendata`，即可由附帶 JSON 離線建立 SQLite，不必下載資料，也不必先收到另一個 `.sqlite3` 檔案。

## 3. 為什麼 FastAPI + SQLite，暫不使用 Alembic

| 元件 | 本次用途 | 決策 |
|---|---|---|
| FastAPI | 路由、輸入驗證、HTTP 狀態碼、OpenAPI、Swagger UI | 使用 |
| Pydantic | 可預測的 Request / Response 契約，拒絕未知欄位與非法值 | 使用 |
| SQLite / Python `sqlite3` | 本機持久化、複合主鍵、交易、參數化查詢 | 使用 |
| SQLAlchemy / SQLModel | ORM 與資料庫抽象 | 本次一張表，直接 SQL 足夠 |
| Alembic | 跨版本資料表結構遷移 | 本次固定初始 schema，暫不使用 |

FastAPI 不強制使用特定資料庫或 ORM；Alembic 是以 SQLAlchemy 為基礎的結構遷移工具，不負責一般資料 CRUD 或日常下載更新。[FastAPI 資料庫說明](https://fastapi.tiangolo.com/tutorial/sql-databases/)、[Alembic 官方文件](https://alembic.sqlalchemy.org/en/latest/)。

啟動時 `CREATE TABLE IF NOT EXISTS` 只負責初始建表，**不會修改既有資料表結構**。日後若新增欄位、改型別、改約束且要保留既有資料，就需要明確的 migration；採用 SQLAlchemy 時可加入 Alembic。SQLite 較複雜的 schema 修改還可能需要 batch migration。[官方 batch migration 說明](https://alembic.sqlalchemy.org/en/latest/batch.html)。

## 4. Resource 與 REST 契約

只定義一個正式資源：`daily-bars`。一筆日 K 的自然主鍵是 `(symbol, date)`，不增加沒有用途的自增 ID。全部資料限 `market=TSE`，因此主鍵沿用現有專案。名稱與漲跌價差保留在原始 JSON，不混入既有日 K 欄位。

| Method | Path | 功能 | 成功 | 主要錯誤 |
|---|---|---|---|---|
| GET | `/api/v1/daily-bars` | 列表、代碼與日期區間篩選、分頁 | 200 | 422 |
| POST | `/api/v1/daily-bars` | 建立日 K | 201 + Location | 409 / 422 |
| GET | `/api/v1/daily-bars/{symbol}/{date}` | 讀取單筆 | 200 | 404 / 422 |
| PUT | `/api/v1/daily-bars/{symbol}/{date}` | 完整替換既有日 K 內容 | 200 | 404 / 422 |
| DELETE | `/api/v1/daily-bars/{symbol}/{date}` | 刪除單筆 | 204、無 body | 404 / 422 |

PUT 不建立不存在的資料、不更改主鍵；要更改主鍵需建立新資源再刪除舊資源。PUT 必填所有價格、成交量、成交金額與成交筆數欄位（可明確填 `null`）；`market` 預設 `TSE`、`source` 預設 `manual`。本次不加 PATCH，CRUD 已完整。

列表回傳 `{items, total, limit, offset}`；預設 `limit=100`、上限 500、`offset=0`。排序固定為日期新到舊，再依代碼升冪。日期區間包含首尾；沒有符合資料時為空列表，單筆不存在才回 404。日期倒置回 422。

## 5. 更新、交易與安全邊界

- 同一次匯入先驗證全部資料；SQLite 寫入在同一交易內，錯誤回滾整批。
- 相同自然主鍵匯入採更新，重跑不增加重複列；未出現在快照的舊資料保留。
- 明確執行匯入會以官方值覆蓋相同主鍵的本機修改，也會重新建立曾刪除的官方資料。重啟 Server 不會匯入。
- 每次 SQLite 操作建立並關閉獨立連線，避免共用連線跨 FastAPI worker thread。
- 查詢值均透過 SQL parameters 傳入；不將使用者輸入拼接為 SQL。
- 不接受 NaN、Infinity、負數、超出 SQLite 64-bit 整數的股數；已知價格需符合最高／最低價邊界。
- 原始快照只有通過驗證才取代舊檔。JSON 替換與 SQLite 匯入不是跨檔交易；若後續 DB 寫入失敗，可重跑匯入完成。
- 此為本機作業服務，預設綁定 `127.0.0.1`，未實作登入或存取權限。不得直接開放公開網路上的寫入服務。

## 6. 驗證與交付

離線測試檢查來源映射、缺值、無效快照、完整 CRUD、重啟持久化、分頁、邊界驗證、匯入冪等與回滾、Swagger 規格及 Client 失敗後清理。

實際 HTTP 驗證使用 `kstock.api.client`，建立本次專用 `DEMO...` 代碼，讀取、列表、測試 409、更新、確認更新、測試 422、刪除與確認 404，最後保存最新 `docs/api_examples.json`。只清理本次成功建立的示範列。

繳交項目與執行命令見根目錄 README；來源 JSON、OpenAPI 與實測範例皆採固定路徑，SQLite 可由 JSON 重建。

# kstock — 台股 Open Data RESTful API 與量化研究資料管道

本專案包含可繳交的 RESTful API 作業：從 TWSE 下載一份 Open Data JSON，沿用既有日 K Schema，以 SQLite 保存資料，透過 FastAPI 提供完整 CRUD，並附 Swagger UI、Python API Client 與離線測試。

```text
TWSE OpenAPI → 單一原始 JSON → Polars 正規化 → SQLite ↔ FastAPI ↔ Python Client
```

原有 FinMind / Parquet / DuckDB 研究流程保留在下方。API 的新增、修改與刪除只影響 SQLite。

## 作業快速開始

需要 Python ≥ 3.11 與 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在專案根目錄執行，不需設定 `.env` 或 `FINMIND_TOKEN`：

```powershell
uv sync
uv run python -m kstock.pipeline.opendata
uv run uvicorn kstock.api.app:app --host 127.0.0.1 --port 8000
```

第二個指令讀取附帶的 `datasets/twse_daily.json`，匯入 `data/kstock.sqlite3` 並產生 `docs/openapi.json`；缺少資料檔時才下載。第三個指令啟動 Server，保持該終端開啟，按 Ctrl+C 可停止。

- Swagger UI：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)，展開端點後選 **Try it out → Execute**。
- ReDoc：[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)。
- 線上 OpenAPI：[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)。

這些網址在本機 Server 運行時才有效；靜態 API 規格已附在 `docs/openapi.json`。Swagger UI / ReDoc 預設從 CDN 載入前端資源，瀏覽文件介面需要網路；附帶快照的匯入與 pytest 不需要網路。

另開終端執行 Client 與測試：

```powershell
uv run python -m kstock.api.client
uv run pytest
```

Client 會實際執行列表、新增、讀取、重複新增、更新、驗證錯誤、刪除及確認不存在，保存最新 `docs/api_examples.json`。示範使用本次專用 `DEMO...` 主鍵，最後清理該筆，不改動官方資料。Server 使用不同位置時可指定 `--base-url http://127.0.0.1:8001`。

## Open Data 來源與儲存

| 項目 | 內容 |
|---|---|
| 資料集 | [盤後資訊 > 個股日成交資訊](https://data.gov.tw/dataset/11549) |
| 官方 API | [TWSE STOCK_DAY_ALL](https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL) |
| 授權 | 政府資料開放授權條款第 1 版；使用時保留來源標示 |
| 單一資料檔 | `datasets/twse_daily.json`，完整保留官方 JSON 陣列與欄位 |
| 隨附快照 | 2026-09-27 下載，官方交易日期 2026-09-24，共 1,380 筆 |
| SQLite | `data/kstock.sqlite3`，可由快照重建，不入版控 |
| 資料表 | `daily`，自然主鍵 `(symbol, date)` |

發布來源為臺灣證券交易所，政府平臺列示提供機關為金融監督管理委員會證券期貨局。資料可能包含 ETF 等證券；這是一個全市場日成交快照，不是完整歷史資料。

民國日期轉成西元 ISO 日期，成交股數保持股，逗號數字轉成 Python 數字，官方空值保留為 `null`。API 回傳既有日 K 的 11 欄，名稱與漲跌價差留在原始 JSON。

需要更新來源時，先停止 Server，再明確重新下載與匯入：

```powershell
uv run python -m kstock.pipeline.opendata --download
```

重跑匯入依 `(symbol, date)` 更新，不累加重複列；其他日期／主鍵保留。**匯入會用官方值覆蓋相同主鍵的本機修改，並重新建立曾刪除的官方資料。** Server 重啟不會重新匯入，因此 CRUD 修改可持續保留。原始快照只在驗證通過後被取代；SQLite 整批寫入採單一交易。若 DB 寫入失敗，修正問題後重跑即可。

`KSTOCK_DATA_DIR` 可覆寫 SQLite 所在資料根目錄；資料集與文件仍固定放在專案的 `datasets/` 與 `docs/`。

## RESTful API 規格

正式資源為日 K `daily-bars`，路徑前綴 `/api/v1`。所有 CRUD 都針對本機資料庫。

| Method | URL Path | 功能 | 成功狀態 |
|---|---|---|---|
| GET | `/api/v1/daily-bars` | 列表與篩選、分頁 | 200 |
| POST | `/api/v1/daily-bars` | 新增日 K | 201 + `Location` |
| GET | `/api/v1/daily-bars/{symbol}/{date}` | 讀取單筆 | 200 |
| PUT | `/api/v1/daily-bars/{symbol}/{date}` | 完整替換既有內容 | 200 |
| DELETE | `/api/v1/daily-bars/{symbol}/{date}` | 刪除單筆 | 204，無 body |

列表參數：`symbol`、`start_date`、`end_date`、`limit`（預設 100，上限 500）、`offset`（預設 0）。日期區間含首尾，結果按日期降冪、代碼升冪排序；回傳 `{items, total, limit, offset}`。

單筆不存在回 404，新增重複主鍵回 409，非法日期／數值／未知欄位回 422。PUT 不建立新資料，也不修改主鍵；必須送完整內容。`market` 固定 `TSE`，`source` 只接受 `twse` 或 `manual`。價格、成交金額、股數及成交筆數可為 `null`；有值時不可為負數，價格須符合最高／最低價邊界，股數與筆數須為 SQLite 64-bit 範圍內的整數。

### Request / Response 範例

以下新增範例使用教學主鍵；完整真實請求與回應見 [docs/api_examples.json](docs/api_examples.json)。

```http
POST /api/v1/daily-bars
Content-Type: application/json

{
  "symbol": "DEMO0001",
  "market": "TSE",
  "date": "2000-01-01",
  "open": 100.0,
  "high": 105.0,
  "low": 99.0,
  "close": 102.0,
  "volume_shares": 1000,
  "turnover_twd": 102000.0,
  "trade_count": 10,
  "source": "manual"
}
```

回應 `201 Created`，`Location: /api/v1/daily-bars/DEMO0001/2000-01-01`，body 為上述資料。`GET /api/v1/daily-bars/DEMO0001/2000-01-01` 讀取同一筆。PUT 使用相同內容但**移除 `symbol`、`date`**，修改 `close` 為 103；DELETE 同一路徑回 204。刪除後再 GET：

```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{"detail": "找不到日 K"}
```

Python Client 可直接重用：

```python
import httpx
from kstock.api.client import StockAPIClient

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as session:
    client = StockAPIClient(session)
    print(client.list(symbol="2330", limit=5).json())
```

## FastAPI / Alembic 的選擇

本次使用 **FastAPI + Pydantic + Python 內建 sqlite3**。FastAPI 自動產生 OpenAPI 與 Swagger；SQLite 處理持久化與交易。固定一張資料表，目前不需要 SQLAlchemy 或 Alembic。

Alembic 是資料表結構遷移工具，適合日後要保留舊資料並增加欄位、改型別或約束時使用。一般 CRUD、下載與更新資料不需要 Alembic。`CREATE TABLE IF NOT EXISTS` 僅建立缺少的資料表，不會自動升級已存在的 schema。[Alembic 官方文件](https://alembic.sqlalchemy.org/en/latest/)

完整現況分析、API 決策、資料契約與風險見 [docs/rest_api_assignment.md](docs/rest_api_assignment.md)。服務預設只監聽本機，沒有認證或權限機制，適用作業展示；公開部署前需要另外設計存取限制。

## 繳交清單與驗證

| 作業要求 | 繳交位置 |
|---|---|
| Open Data 單一檔案 | `datasets/twse_daily.json` |
| Python 程式碼 | `src/kstock/`、`pyproject.toml` |
| API 規格與文件 | `docs/openapi.json`、Server 的 `/docs`、本 README |
| 實際 API Request / Response | `docs/api_examples.json` |
| Python API Client | `src/kstock/api/client.py` |
| 環境建置與執行 | 本 README |
| AI 輔助開發必要檔案 | `docs/ai_development.md`、設計分析與測試 |
| 可重現驗證 | `tests/`，`uv run pytest` |

SQLite 不需隨 GitHub 提交，助教可以用快照重建。`datasets/` 與 `docs/` 不在忽略清單內，提交時須一併包含。不要繳交 `.env`、金鑰或 `.venv`。若使用 Google Drive，保留同樣目錄結構並確認分享權限；目前只在本機整理，尚未上傳或發布。

實際驗證：全部 **132 個離線測試通過**；真實 TWSE 快照 **1,380 筆**匯入 SQLite；本機 Server 的 **10 個 Client HTTP 請求通過**，涵蓋 200 / 201 / 204 / 404 / 409 / 422；Chrome 確認 Swagger UI 正常顯示五個端點。現行 Starlette TestClient 對 `httpx` 有一則棄用警告，未影響測試通過。

## 原有研究流程

依循 `docs/taiwan_stock_api_architecture_summary.md` 的第一步實作：

```
FinMind → Adapter → Normalized Schema → Parquet → DuckDB → Backtest
```

## 快速開始

```bash
cp .env.example .env   # 填入 FINMIND_TOKEN（可後補）
uv sync                # 建立虛擬環境並安裝依賴（含 dev: pytest）
uv run pytest          # 執行測試（全部離線運行，不需網路/金鑰）
```

## 目錄結構

| 路徑 | 說明 |
|---|---|
| `src/kstock/adapters/` | FinMind / TWSE / Yahoo Finance 資料來源 Adapter |
| `src/kstock/api/` | FastAPI 日 K CRUD、HTTP 契約與 Python Client |
| `src/kstock/normalizers/` | 成交量單位、價格還原等標準化 |
| `src/kstock/storage/` | SQLite 作業儲存層 + Parquet 分層儲存 + DuckDB 查詢層 |
| `src/kstock/features/` | 技術指標（回報率、MA、RSI 等） |
| `src/kstock/backtest/` | 向量化回測引擎 |
| `src/kstock/pipeline/` | 資料更新管道（下載 → 標準化 → 寫入） |
| `datasets/` | 作業繳交用的單一 TWSE 原始 JSON |
| `docs/` | 架構與作業分析、OpenAPI、HTTP 實測、AI 開發紀錄 |
| `tests/` | 研究流程與 API 測試（全部離線，mock HTTP） |
| `data/` | raw / normalized / features（由程式產生，不入版控） |

## 標準 Schema（NORMALIZED 層）

統一輸出欄位（`src/kstock/models/schema.py`）：

- `daily_bar`：symbol, market, date, open/high/low/close, volume_shares, turnover_twd, trade_count, source
- `hourly_bar`：同 daily_bar 欄位，但 `date` 為 naive Datetime（台北時間的小時開始）
- `instrument`：symbol, name, market, industry, list_date, delist_date, status
- `institutional` / `margin`：見 schema.py

成交量一律標準化成 **股（volume_shares）**，策略只讀這個欄位。

## 環境變數

| 變數 | 用途 |
|---|---|
| `FINMIND_TOKEN` | FinMind API Token（留空時離線測試不受影響） |
| `FINMIND_BASE_URL` | FinMind API 位置（預設共用） |
| `FINMIND_VOLUME_UNIT` | FinMind 原始成交量單位（`lots`=張→×1000，`shares`=股） |
| `TWSE_BASE_URL` | TWSE OpenAPI（公開，無需金鑰） |
| `SHIOAJI_KEY` / `SHIOAJI_SECRET` | 第二階段使用，先保留 |
| `KSTOCK_DATA_DIR` | 資料根目錄（預設 `./data`） |

## AI 因子研究（qlab 子專案：Qlib + RD-Agent）

`src/qlab/` 與本套件共用 venv、`.env` 與 `data/`，把 normalized daily 轉成 Qlib bin 格式後進行 AI 因子分析（Alpha158 + LightGBM、RD-Agent 因子自動演化）。安裝、金鑰（支援 **OpenRouter** / DeepSeek / OpenAI）與使用說明見 **`docs/qlab_setup.md`**。

```bash
uv run python -m qlab export     # data/normalized/daily → data/qlab/qlib_data（Qlib bin 格式）
uv run python -m qlab ic --start 2021-01-01 --end 2026-08-26   # Alpha158 + LightGBM IC 分析
uv run python -m qlab rdagent quant   # RD-Agent（需 Docker + LLM 金鑰）

# fin_factor 因子演化（台股 daily_pv.h5；金鑰由 .env 自動注入）
uv run python -m qlab export-h5                    # 台股日K → daily_pv.h5（正式版 + debug 20 檔）
uv run python -m qlab export-tw100                 # 切出代碼前 100 檔 → factor_source_data_tw100/（code_first_n 正式口徑）
uv run python -m qlab rdagent fin_factor --universe tw100   # 演化宇宙=前 100 檔（與回測口徑一致）
uv run python -m qlab rdagent fin_factor --universe tw100 --guidance short
                                                   # 假設性引導：做空導向（高值→未來跌；
                                                   # 禁止 volume_change_5d 反向與既有因子翻版，
                                                   # 見 qlab/short_factor_proposal.py）
uv run python -m qlab factor-report --fwd-days 5   # 因子清單 + 截面 Rank IC → data/qlab/factor_report.md
uv run python -m qlab backtest --top-symbols 100 --direction short
                                                   # 做空回測（top 訊號、報酬取負、含成本淨值）
```

## 使用範例

```python
from kstock.config.settings import settings
from kstock.adapters.finmind import FinMindAdapter
from kstock.storage.parquet import ParquetStore
from kstock.storage.duckdb import DuckStore
from kstock.pipeline.daily import DailyUpdatePipeline

fm = FinMindAdapter(token=settings.finmind_token)
store = ParquetStore(settings.data_dir)
pipeline = DailyUpdatePipeline(adapter=fm, store=store, duck=DuckStore(settings))
pipeline.run(symbols=["2330"], start_date="2024-01-01", end_date="2026-12-31")

duck = DuckStore(settings)
duck.register_views()
df = duck.query("SELECT * FROM daily WHERE symbol='2330' ORDER BY date DESC LIMIT 5")
```

> 注意：FinMind 各 dataset 的欄位名稱可能隨官方更新，Adapter 已提供
> `FinMindAdapter.field_names("TaiwanStockMargin"...)` 方便除錯對齊。

### 法人 / 融資券（籌碼面）

```python
inst = fm.get_institutional("2330", "2024-08-01", "2024-08-31")  # 三大法人（股）
margin = fm.get_margin("2330", "2024-08-01", "2024-08-31")       # 融資融券（張→股）
store.write_normalized("institutional", inst)
store.write_normalized("margin", margin)
```

- 法人資料源 `TaiwanStockInstitutionalInvestorsBuySellWide`（寬表），自營商合計 = 自行買賣 + 避險。
- 融資券資料源 `TaiwanStockMarginPurchaseShortSale`，單位為張，Adapter 自動 ×1000 轉股。
- 皆輸出標準 schema（`models/schema.py`），可寫入 Parquet 或由 DuckDB 查詢。

### 選股宇宙（採集清單 / collection manifest）

```python
from kstock.universe.twse_whole_market import WholeMarketQuotes, select_top_liquid

quotes = WholeMarketQuotes()
frames = quotes.fetch_recent_days(sample_days=5)     # 近 5 個交易日全市場報表（TWSE）
top = select_top_liquid(frames, n=300)               # 依日均成交金額排名（排除 ETF、低價股）
top.write_csv("data/universe/top_liquidity_300.csv")
```

> **宇宙語意分工**（詳見 `docs/layering_and_universe_design.md`）：`data/universe/`
> 下的清單是**採集範圍 manifest**（決定 backfill 抓哪些代碼，如 top_liquidity_300.txt），
> 不是回測宇宙；回測宇宙由 `qlab.universe.UniverseSpec` 定義（`code_first_n` 正式口徑），
> 兩者不得混用。歷史宇宙回推（含下市候選清單）見「歷史宇宙回推」段。

```bash
uv run python -m kstock.instruments                   # TaiwanStockInfo → instrument 參考表（單檔 upsert）
uv run python -m kstock.universe_history --start 2008-01-01 --sleep 5   # MI_INDEX 月抽樣 + 官方終止上市清單 → manifest
uv run python -m kstock.backfill --delisted-csv data/universe/manifests/delisted_tw_YYYYMMDD.csv --start 2008-01-01 --tables daily   # 下市股回補（delist_date 裁切）
```

資料品質：`clean_daily(df)` 可在 ML 訓練前移除停牌等無效價格列；
OHLC 一致性檢查含四捨五入容差（≤ max(0.06 元, 0.2%×價格)）。

### ML 特徵管線（FEATURE 層）

```python
from kstock.features.ml import build_ml_features
from kstock.storage.parquet import ParquetStore

store = ParquetStore()
feat = build_ml_features(
    store.read_normalized("daily"),
    store.read_normalized("institutional"),
    store.read_normalized("margin"),
)
store.write_feature("ml_daily_v1", feat)   # -> data/features/ml_daily_v1.parquet
```

20 欄特徵：動能（return_1d/5d/20d、ma 比值）、波動、量比、RSI_14，
籌碼（外資/投信/三大法人淨買張數與淨買比率、融券增減率），以及**無洩漏**標籤 `label_ret_1f`。
所有欄位僅用當日（含）以前資訊。

### 小時K線（Yahoo Finance）

FinMind 免費等級無盤中資料（tick / KBar / 5 秒指標皆需 VIP），
小時K 改用 Yahoo Finance（`interval=60m`，歷史上限約 **730 天**）：

```python
from kstock.adapters.yfinance import YFinanceAdapter
from kstock.pipeline.hourly import HourlyUpdatePipeline
from kstock.storage.parquet import ParquetStore

store = ParquetStore(settings.data_dir)
pipeline = HourlyUpdatePipeline(adapter=YFinanceAdapter(), store=store)
pipeline.run(symbols=["2330"], start_date="2024-08-27")  # date 欄位 = naive 台北時間的小時開始
```

輸出表 `hourly`（schema 與 daily 同欄位，`date` 為 `Datetime`）。

**已知限制（Yahoo 源頭資料特性，程式端無法修復）：**

1. **成交量不可靠**：09:00 首根常為 0，小時量加總僅日K的 33%–86%
   → 勿用於量能策略；OHLC 價格不受影響。
2. **不含收盤競價**：台股 13:25–13:30 集合競價跳動不在最後一根 K 棒內，
   最後一根 close 與官方收盤價中位數差約 0.25%（P90 ≈ 5 元）。
3. 用途建議：日內走勢形態、時段效應研究；精確收/開盤分析請以 daily 層為準。

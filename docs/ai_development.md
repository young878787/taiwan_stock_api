# AI 輔助開發紀錄

本作業使用 Codex 協助需求分析、查閱官方來源、實作及驗證。這份文件連同程式與測試可作為 AI 輔助開發的必要檔案；執行 Server 不需要 AI API、帳號或金鑰。

## 提供給 AI 的需求

- 依既有台股資料流程，以 SQLite 儲存 Open Data。
- 使用 Python + FastAPI 提供符合資料主題的完整 RESTful CRUD。
- 自動產生 Swagger UI，提供可測試的 Python Client。
- 更新 README，整理資料檔、程式、API 規格、Client 與 AI 開發相關檔案。
- 評估 Alembic 是否必要，避免為單一固定資料表增加無必要依賴。

## 專案限制

- 使用繁體中文文件與 docstring、Polars 資料處理、`uv run` 執行 Python。
- 重用 `models/schema.py` 的日 K 欄位，成交量統一為股。
- 測試 HTTP 全 mock，不需網路或 `FINMIND_TOKEN`。
- 保留原有 Parquet / DuckDB 研究流程與使用者未提交修改。
- 不自行 commit、push、部署或修改外部服務。

## AI 完成的工作與可查證產物

| 工作 | 產物 |
|---|---|
| 追查來源與技術決策 | `docs/rest_api_assignment.md` |
| 官方 JSON 正規化 | `src/kstock/adapters/twse.py` 的 `normalize_daily_snapshot()` |
| SQLite 儲存與匯入 | `src/kstock/storage/sqlite.py`、`src/kstock/pipeline/opendata.py` |
| HTTP CRUD 與輸入驗證 | `src/kstock/api/app.py`、`src/kstock/api/models.py` |
| Python Client 與實測 | `src/kstock/api/client.py`、`docs/api_examples.json` |
| 離線回歸測試 | `tests/test_opendata_api.py` |

來源文件與實測結果用於檢查 AI 的產出；不把 AI 推測視為來源事實。實際下載辨識到缺值列，因此保留 `null`，而非虛構價格或成交量。完整測試命令與繳交方式見 README。

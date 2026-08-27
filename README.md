# kstock — 台股量化研究資料管道（第一版 MVP）

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
| `src/kstock/normalizers/` | 成交量單位、價格還原等標準化 |
| `src/kstock/storage/` | Parquet 分層儲存 + DuckDB 查詢層 |
| `src/kstock/features/` | 技術指標（回報率、MA、RSI 等） |
| `src/kstock/backtest/` | 向量化回測引擎 |
| `src/kstock/pipeline/` | 資料更新管道（下載 → 標準化 → 寫入） |
| `tests/` | 第一版測試（全部離線，mock HTTP） |
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
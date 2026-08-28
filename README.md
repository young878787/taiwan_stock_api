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

## AI 因子研究（qlab 子專案：Qlib + RD-Agent）

`src/qlab/` 與本套件共用 venv、`.env` 與 `data/`，把 normalized daily 轉成 Qlib bin 格式後進行 AI 因子分析（Alpha158 + LightGBM、RD-Agent 因子自動演化）。安裝、金鑰（支援 **OpenRouter** / DeepSeek / OpenAI）與使用說明見 **`docs/qlab_setup.md`**。

```bash
uv run python -m qlab export     # data/normalized/daily → data/qlab/qlib_data（Qlib bin 格式）
uv run python -m qlab ic --start 2021-01-01 --end 2026-08-26   # Alpha158 + LightGBM IC 分析
uv run python -m qlab rdagent quant   # RD-Agent（需 Docker + LLM 金鑰）
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

### 選股宇宙（ML 用前 N 大流動性）

```python
from kstock.universe.twse_whole_market import WholeMarketQuotes, select_top_liquid

quotes = WholeMarketQuotes()
frames = quotes.fetch_recent_days(sample_days=5)     # 近 5 個交易日全市場報表（TWSE）
top = select_top_liquid(frames, n=300)               # 依日均成交金額排名（排除 ETF、低價股）
top.write_csv("data/universe/top_liquidity_300.csv")
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
# 台股 API 研究架構總結

## 一、目標

建立一套適合台股量化研究、策略回測與後續自動交易的資料架構。

核心需求：

- 取得台股歷史價格與成交量
- 取得法人、融資融券、基本面等研究資料
- 取得盤中即時行情與分鐘級資料
- 支援歷史回測與因子研究
- 後續可直接接入自動下單
- 資料來源可交叉校驗
- 架構保持簡潔，不過度工程化
- 適合 Python、Polars、Pandas、Qlib、Backtrader 等研究工具

---

# 二、技術選型

整體採用：

```text
Shioaji
  └─ 即時行情 / 券商帳戶 / 自動下單

FinMind
  └─ 主要歷史與研究資料來源

TWSE OpenAPI
  └─ 官方輔助資料來源 / 校驗 / 補資料

Parquet
  └─ 原始與整理後歷史資料儲存

DuckDB
  └─ 查詢、研究、因子計算、回測資料存取
```

推薦核心組合：

> **Shioaji + FinMind + TWSE OpenAPI + Parquet + DuckDB**

---

# 三、各元件定位

## 1. Shioaji

用途：

- 即時行情
- Tick
- Bid / Ask
- KBar
- 帳戶資訊
- 庫存
- 委託查詢
- 下單
- 改單
- 刪單

Shioaji 不作為主要長期歷史資料庫來源，而是主要負責：

```text
Live Market Data
+
Execution
```

架構定位：

```text
Strategy
   │
   ▼
Risk Manager
   │
   ▼
Shioaji
   │
   ▼
Broker / TWSE
```

### 適合用途

- 實盤策略
- 模擬盤
- 即時訊號
- 即時分鐘 K
- Tick 收集
- 自動交易

---

# 四、FinMind

FinMind 作為主要研究資料來源。

主要負責：

## 價格資料

- 歷史日 K
- 還原價格
- Tick
- 分鐘 K

## 籌碼資料

- 三大法人
- 融資融券
- 借券
- 當沖

## 基本面

- 月營收
- 股利
- 財報相關資料
- PER / PBR 等估值資料

## 股票資訊

- 股票代碼
- 股票名稱
- 市場別
- 產業資訊

---

# 五、TWSE OpenAPI

TWSE OpenAPI 不作為主要資料來源。

定位是：

```text
Official Fallback
+
Validation Source
```

主要用途：

- 驗證 FinMind 資料
- 補最新交易日資料
- 查官方市場資訊
- 查融資融券
- 查三大法人
- 查成交統計
- 查本益比、殖利率
- 查注意股、處置股
- 當第三方 API 發生異常時使用

如果需要上櫃資料，可另外整合：

```text
TPEx OpenAPI
```

形成：

```text
TWSE
+
TPEx
```

作為官方資料層。

---

# 六、資料來源優先順序

推薦：

```text
Historical Research
        │
        ▼
     FinMind
        │
        ▼
   Parquet / DuckDB


Official Validation
        │
        ▼
   TWSE / TPEx


Realtime Market
        │
        ▼
     Shioaji


Order Execution
        │
        ▼
     Shioaji
```

---

# 七、資料庫設計

不採用傳統大型資料庫作為第一版核心。

第一版使用：

```text
Parquet + DuckDB
```

---

# 八、為什麼使用 Parquet

Parquet 適合股票歷史資料。

優點：

- Columnar Storage
- 壓縮率高
- 讀取速度快
- Polars 支援很好
- Pandas 支援
- DuckDB 可直接查詢
- 不需要 Database Server
- 適合大量歷史 OHLCV
- 適合時間序列資料

例如：

```text
data/
├─ raw/
│
├─ normalized/
│
└─ features/
```

---

# 九、為什麼使用 DuckDB

DuckDB 作為 Analysis Engine。

不用把資料全部 Import 進 DB。

可以直接查 Parquet：

```sql
SELECT *
FROM read_parquet('data/normalized/daily/*.parquet')
WHERE symbol = '2330'
ORDER BY date;
```

適合：

- Backtest
- Factor Analysis
- Strategy Research
- Feature Engineering
- Cross-sectional Query
- AI Agent 查詢
- Dataset Export

---

# 十、資料目錄設計

推薦：

```text
data/
│
├─ raw/
│   │
│   ├─ finmind/
│   │   ├─ stock_price/
│   │   ├─ institutional/
│   │   ├─ margin/
│   │   └─ fundamental/
│   │
│   ├─ twse/
│   │   ├─ daily/
│   │   ├─ institutional/
│   │   └─ margin/
│   │
│   └─ shioaji/
│       ├─ tick/
│       └─ realtime/
│
├─ normalized/
│   │
│   ├─ daily/
│   ├─ minute/
│   ├─ tick/
│   ├─ institutional/
│   ├─ margin/
│   ├─ fundamental/
│   └─ instruments/
│
├─ features/
│   ├─ technical/
│   ├─ factor/
│   └─ signals/
│
└─ reports/
```

---

# 十一、Parquet 分割方式

日 K：

```text
daily/
├─ year=2024/
├─ year=2025/
└─ year=2026/
```

或：

```text
daily/
├─ 2024.parquet
├─ 2025.parquet
└─ 2026.parquet
```

分鐘資料：

```text
minute/
├─ date=2026-08-25/
├─ date=2026-08-26/
└─ date=2026-08-27/
```

Tick：

```text
tick/
├─ date=2026-08-27/
│   ├─ 2330.parquet
│   ├─ 2317.parquet
│   └─ 2454.parquet
```

Tick 資料不要一開始就收全市場。

先只記錄策略關注標的。

---

# 十二、標準資料 Schema

## daily_bar

```text
symbol
market
date

open
high
low
close

volume_shares
turnover_twd
trade_count

adjusted
source
```

---

## minute_bar

```text
symbol
timestamp

open
high
low
close

volume_shares
turnover_twd

source
```

---

## tick

```text
symbol
timestamp

price
size_shares

tick_type

bid
ask

source
```

---

## instrument

```text
symbol
name
market
industry

list_date
delist_date

status
```

---

## institutional

```text
symbol
date

foreign_buy
foreign_sell

investment_trust_buy
investment_trust_sell

dealer_buy
dealer_sell
```

---

## margin

```text
symbol
date

margin_balance
margin_buy
margin_sell

short_balance
short_sell
short_cover
```

---

# 十三、成交量單位標準化

不同 API 的 volume 定義可能不同。

可能存在：

```text
股
張
成交筆數
```

資料進 normalized layer 時統一轉成：

```text
volume_shares
```

例如：

```text
1 張 = 1000 股
```

所有策略都只讀：

```text
volume_shares
```

不要讓策略自行判斷資料來源單位。

---

# 十四、價格還原

長期回測不能只用 raw close。

需要處理：

- 除權
- 除息
- 減資
- 股票分割
- Corporate Actions

因此保留：

```text
raw price
+
adjusted price
```

例如：

```text
close
adj_close
```

或使用：

```text
adjusted = true / false
```

---

# 十五、資料層設計

推薦採用三層：

```text
RAW
 ↓
NORMALIZED
 ↓
FEATURE
```

---

## RAW

完全保留 API 原始內容。

例如：

```text
raw/finmind/
raw/twse/
raw/shioaji/
```

不要直接修改。

用途：

- Debug
- API schema 變動
- 重建 normalized data
- 比對資料來源

---

## NORMALIZED

將所有 API 統一格式。

例如：

```text
Trading_Volume
volume
成交股數
```

全部轉成：

```text
volume_shares
```

---

## FEATURE

策略與模型使用的特徵。

例如：

```text
return_1d
return_5d

ma5
ma20

rsi14

volume_ratio

foreign_net_buy

margin_change

volatility_20d
```

---

# 十六、程式架構

推薦：

```text
src/
│
├─ adapters/
│   ├─ finmind.py
│   ├─ twse.py
│   └─ shioaji.py
│
├─ collectors/
│   ├─ daily.py
│   ├─ minute.py
│   └─ realtime.py
│
├─ normalize/
│   ├─ price.py
│   ├─ volume.py
│   └─ instrument.py
│
├─ storage/
│   ├─ parquet.py
│   └─ duckdb.py
│
├─ features/
│   ├─ technical.py
│   ├─ factor.py
│   └─ signals.py
│
├─ strategy/
│
├─ backtest/
│
├─ execution/
│   ├─ broker.py
│   ├─ risk.py
│   └─ shioaji.py
│
└─ config/
```

---

# 十七、Adapter 設計

策略層不要直接依賴 API。

錯誤：

```python
FinMind API
→ strategy.py
```

推薦：

```text
FinMind
TWSE
Shioaji
   │
   ▼
Data Adapter
   │
   ▼
Normalized Schema
   │
   ▼
Strategy
```

例如統一：

```python
get_daily_bars()

get_minute_bars()

get_ticks()

get_institutional()

get_margin()
```

這樣以後更換資料商不需要改策略。

---

# 十八、研究與實盤分離

這點非常重要。

## Research

```text
Parquet
   │
DuckDB
   │
Strategy
   │
Backtest
```

## Live

```text
Shioaji
   │
Realtime Adapter
   │
Strategy
   │
Risk Manager
   │
Execution
   │
Shioaji
```

Strategy Logic 可以共用。

但：

```text
Data Feed
Execution
```

必須分離。

---

# 十九、完整資料流

```text
                    FinMind
                       │
                       │
                       ▼
                 Historical Data
                       │
                       ▼
                    RAW
                       │
                       ▼
                 Normalize
                       │
                       ▼
                    Parquet
                       │
                       ▼
                    DuckDB
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Research     Backtest     AI / ML
          │
          ▼
       Strategy
          │
          ▼
      Signal Logic
          │
          │
          ├─────────────── Live ───────────────┐
          │                                    │
          ▼                                    ▼
       Shioaji                          Realtime Data
          │                                    │
          ▼                                    │
     Risk Manager ◄────────────────────────────┘
          │
          ▼
       Execution
          │
          ▼
       Shioaji
          │
          ▼
      Broker / TWSE


TWSE OpenAPI
      │
      └──────── Validation / Fallback
```

---

# 二十、每日資料更新流程

盤後：

```text
14:00+
  │
  ▼
FinMind Fetch
  │
  ▼
TWSE Validation
  │
  ▼
Normalize
  │
  ▼
Update Parquet
  │
  ▼
DuckDB Query
  │
  ▼
Feature Calculation
```

---

# 二十一、盤中流程

```text
09:00
  │
  ▼
Shioaji Subscription
  │
  ├─ Tick
  ├─ BidAsk
  └─ KBar
  │
  ▼
Realtime Strategy
  │
  ▼
Signal
  │
  ▼
Risk Check
  │
  ▼
Order
```

盤中資料可以另外寫入：

```text
data/raw/shioaji/
```

收盤後再轉：

```text
normalized/minute/
normalized/tick/
```

---

# 二十二、資料校驗

重要資料可以進行：

```text
FinMind
   │
   │ compare
   ▼
TWSE
```

驗證：

- close
- volume
- turnover
- trading date

異常則：

```text
data_quality_error
```

例如：

```text
2330
2026-08-27

FinMind close = 1230
TWSE close = 1230

PASS
```

---

# 二十三、第一階段不要做的事情

暫時不要加入：

- Kafka
- Spark
- Flink
- Kubernetes
- ClickHouse
- Elasticsearch
- Redis Cluster
- 全市場 Tick 長期保存

對目前台股研究而言容易過度工程化。

先確定：

```text
Parquet + DuckDB
```

是否真的遇到效能瓶頸。

---

# 二十四、第一階段 MVP

先完成：

## Data

- [ ] 股票清單
- [ ] 歷史日 K
- [ ] Adjusted Price
- [ ] 三大法人
- [ ] 融資融券
- [ ] 大盤指數

## Storage

- [ ] Raw Parquet
- [ ] Normalized Parquet
- [ ] DuckDB Query Layer

## Research

- [ ] 報酬率
- [ ] Moving Average
- [ ] Volume
- [ ] Institutional Flow
- [ ] 基礎策略回測

---

# 二十五、第二階段

加入：

- Shioaji 即時行情
- 1 分 K
- Tick
- 模擬交易
- Portfolio
- Risk Manager

---

# 二十六、第三階段

加入：

- Shioaji 自動下單
- Position Management
- Order Management
- Stop Loss
- Max Position
- Daily Loss Limit
- Kill Switch

---

# 二十七、未來可接的研究框架

資料整理成標準 Schema 後，可以再接：

```text
Qlib
Backtrader
VectorBT
Zipline-reloaded
Polars
LightGBM
XGBoost
PyTorch
```

尤其如果之後研究：

```text
Machine Learning
Factor Investing
Cross-sectional Ranking
Alpha Mining
```

可以考慮把 Parquet 資料轉成 Qlib Data Format。

---

# 二十八、最終選型

## Broker / Execution

```text
Shioaji
```

負責：

- 即時行情
- 帳戶
- 訂單
- 自動交易

---

## Main Data Provider

```text
FinMind
```

負責：

- Historical Data
- Technical Data
- Institutional
- Margin
- Fundamental

---

## Official Data Provider

```text
TWSE OpenAPI
```

負責：

- Validation
- Fallback
- Official Market Data

必要時增加：

```text
TPEx OpenAPI
```

---

## Storage

```text
Parquet
```

負責：

- Historical Storage
- Raw Data
- Normalized Data
- Feature Data

---

## Query Engine

```text
DuckDB
```

負責：

- SQL Query
- Backtest Dataset
- Feature Analysis
- Research

---

# 二十九、核心原則

整套架構維持：

```text
Data Provider
      ↓
Adapter
      ↓
Normalized Data
      ↓
Parquet
      ↓
DuckDB
      ↓
Research / Backtest
      ↓
Strategy
      ↓
Risk
      ↓
Shioaji
```

核心概念：

> **資料、策略、交易三層分離。**

不要讓：

```text
Strategy
```

直接依賴：

```text
FinMind
TWSE
Shioaji API Schema
```

而是讓 Strategy 永遠只讀統一格式資料。

這樣未來即使：

```text
FinMind → TEJ

Shioaji → 其他券商

DuckDB → ClickHouse
```

策略本身仍然可以繼續使用。

---

# 三十、目前最推薦的實作順序

```text
1. FinMind Historical Downloader
        ↓
2. Normalized Parquet Schema
        ↓
3. DuckDB Query Layer
        ↓
4. TWSE Validation
        ↓
5. Basic Backtest
        ↓
6. Shioaji Realtime Feed
        ↓
7. Paper Trading
        ↓
8. Risk Manager
        ↓
9. Shioaji Live Trading
```

第一個真正應該完成的版本是：

```text
FinMind
   ↓
Parquet
   ↓
DuckDB
   ↓
Backtest
```

先證明資料與策略研究流程正常。

再加入：

```text
Shioaji
```

避免研究層與實盤層同時開發造成複雜度快速上升。

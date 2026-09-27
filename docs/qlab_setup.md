# qlab — Qlib + RD-Agent 因子研究子專案（安裝與部署指南）

`src/qlab/` 是 kstock 的 AI 因子研究子專案，與主套件**共用同一個 uv 虛擬環境、同一份 `.env`、同一個 `data/` 目錄**。

```
kstock daily Parquet ──export──▶ Qlib bin 格式 ──▶ Qlib（Alpha158 + LightGBM）──▶ IC 報告
                                        ▲
                                        └── RD-Agent（LLM 自動演化因子/模型，回測跑在 Qlib 上）
```

## 1. 環境需求

| 項目 | 說明 |
|---|---|
| Python | **3.12**（`.python-version` 已鎖定；pyqlib 官方 wheel 最高只到 cp312） |
| 依賴 | 使用 qlab 時需安裝 `ai` 依賴群組（含 `pyqlib` + `rdagent`）；一般 `uv sync` 不安裝此群組 |
| Docker | RD-Agent 的量化情境（`quant`、`fin_factor`）**執行階段**需要 Docker 容器跑 Qlib 回測；安裝階段不需要 |

```bash
uv sync --group ai     # 重建 .venv（Python 3.12）並安裝 qlab AI 依賴
uv run python -c "import qlib, rdagent; print('ok')"
```

## 2. 金鑰設定（共用 .env）

RD-Agent 需要 LLM 金鑰，與 kstock 的 `FINMIND_TOKEN` 放同一份 `.env`。支援 OpenRouter / DeepSeek / OpenAI 擇一：

```bash
# OpenRouter（推薦：一個金鑰可用多種模型）
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-chat-v3-0324

# 或 DeepSeek 官方
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 或 OpenAI 官方
OPENAI_API_KEY=sk-...
```

金鑰由 `kstock.config.settings.Settings` 統一載入，`qlab.rdagent_runner` 執行時自動注入子行程環境（OpenRouter 會映射成 `OPENAI_API_KEY` + `OPENAI_API_BASE` + `CHAT_MODEL`），不需另外 export。

## 3. 資料匯出（kstock → Qlib）

```bash
uv run python -m qlab export
```

把 `data/normalized/daily/year=*/*.parquet` 轉成 Qlib bin 格式，輸出到 `data/qlab/qlib_data/`：

```
calendars/day.txt        # 交易日曆（由資料本身產生，含台股交易日）
instruments/all.txt      # 符號與起訖日期
features/<SYM>/*.day.bin # open/high/low/close/volume/turnover（float32，缺日 NaN）
```

- 欄位對應：`volume_shares`（股）→ `$volume`、`turnover_twd` → `$turnover`。
- 重跑 `export` 是冪等的（直接覆蓋輸出）。
- 位置可用 `QLAB_PROVIDER_DIR` 覆寫。

## 4. 因子分析（初步部署驗證）

```bash
uv run python -m qlab ic --start 2021-01-01 --end 2026-08-26 --test-start 2026-01-01
```

流程：`qlib.init`（指向 bin 資料集）→ Alpha158 handler → DatasetH（train/test 分段）→ LightGBM → 逐日 Rank IC 與 ICIR。

## 5. RD-Agent 使用

```bash
uv run python -m qlab rdagent fin_factor          # 財報因子自動演化（需 Docker）
uv run python -m qlab rdagent quant               # 完整量化研究迴圈 RD-Agent(Q)（需 Docker）
uv run python -m qlab rdagent fin_factor_report   # 產出因子研究報告
```

工作目錄固定在 `data/qlab/rdagent_workspace/`（不入版控）。RD-Agent 的 Qlib 回測容器會自行下載 qlib 鏡像（`micio/rdagent-qlib` 等），首次執行需要一段時間。

## 6. 已知限制

- **Qlib 不支援 Python 3.13** → 專案已透過 `.python-version` 鎖 3.12，勿升回。
- `hourly` 表成交量不可靠（見 AGENTS.md），qlab 因子計算一律以 `daily` 層為準。
- 台股自訂日曆由資料產生，不使用 Qlib 內建的中/美交易日曆。
- RD-Agent 各版本 CLI 參數可能調整，細節以 `rdagent --help` 為準。
- `pydantic-ai-slim` 需鎖定 `<2`（2.x 移除了 `MCPServerStreamableHTTP`，rdagent 0.8.0 會 import 失敗）。

## 7. OpenRouter 模型相容性（實測）

RD-Agent 大量依賴**嚴格 JSON 結構化輸出**（因子規格生成、程式碼生成等）。OpenRouter 模型需能遵守 JSON 指令：

| 模型 | 純文字 | 結構化 JSON | 可跑 RD-Agent |
|---|---|---|---|
| `openai/gpt-4o-mini` | ✅ | ✅ | ✅（建議） |
| `inclusionai/ling-3.0-flash-fin:free` | ✅ 金融問答品質佳 | ⚠️ 原生不守 JSON → 已由 instructor 修正 | ✅（走 instructor backend） |
| `deepseek/deepseek-chat-v3-0324` | ⚠️ content 全為 null（token 被 reasoning 吃掉） | ❌ | ❌ |

### 7.1 `inclusionai/ling-3.0-flash-fin:free` 的結構化輸出修正（instructor）

該模型**不支援 `response_format` / `structured_outputs`**（已由 OpenRouter models API
`supported_parameters` 證實），原生行為是回長篇報告而非 JSON，導致 RD-Agent 無限重試。

解法：`qlab/rdagent_instructor.py` 提供自訂 RD-Agent backend
`InstructorLiteLLMBackend`（繼承 `LiteLLMAPIBackend`），結構化輸出改走
[instructor](https://github.com/567-labs/instructor) 的 `Mode.MD_JSON`：

1. JSON Schema 以 markdown 指令注入 prompt（不依賴伺服端 `response_format`）；
2. 解析回應中的 JSON（容忍 `<think>` 區塊、長篇報告、code block）；
3. pydantic 驗證失敗時，instructor 會把「驗證錯誤 + 原回應」回灌模型自動重試。

`qlab/rdagent_runner.py` 已自動注入 `BACKEND=qlab.rdagent_instructor.InstructorLiteLLMBackend`，
不需手動設定。相關環境變數：

| 變數 | 預設 | 說明 |
|---|---|---|
| `KSTOCK_RDA_INSTRUCTOR_BACKEND` | `1` | 設 `0` 關閉，改回純 LiteLLM backend |
| `KSTOCK_INSTRUCTOR_RETRIES` | `4` | instructor 單次請求內的驗證重試次數 |

單獨實測結構化輸出（不需 Docker / 不跑完整 RD-Agent）：

```bash
uv run python -m qlab rdatest                                   # 用 .env 的 OPENROUTER_MODEL
uv run python -m qlab rdatest --model inclusionai/ling-3.0-flash-fin:free --retries 6
```

> 限制：MD_JSON 是 prompt 約束 + 驗證重試，**不是生成時 logits 約束**
> （後者只有 outlines/vLLM 等本地後端做得到），因此不保證 100% 遵守 schema。

### 7.2 embedding：本地 fallback（OpenRouter / DeepSeek 皆無 embedding 端點）

RD-Agent 的 embedding 只用於知識庫/因子去重的**相似度比較**。實測 OpenRouter 對
`text-embedding-3-small` 回 400（`encoding_format` 錯誤），DeepSeek 官方亦無 embedding API。

`InstructorLiteLLMBackend` 提供 embedding 供應商切換
（`rdagent_runner` 在 OpenRouter 分支自動注入 `KSTOCK_EMBEDDING_PROVIDER=local`）：

| 值 | 行為 |
|---|---|
| `local`（預設） | 純 Python 字元 3-gram hash 向量，零金鑰、離線可用；對因子名/描述的相似度排序足以取代真 embedding |
| `openai` | 走 litellm（需真正的 OpenAI / Azure 金鑰） |

### 7.3 conda 環境（fin_factor 因子執行）

fin_factor 的因子程式碼以 `conda run -n <env>` 在本機 conda 環境執行（quant 走 Docker）。
`rdagent_runner` 會偵測 `~/miniconda3` 並自動注入 `CONDA_DEFAULT_ENV`、`BIN_PATH`。
手動準備（RD-Agent 首次執行也會自動建 `rdagent4qlib` 並安裝 qlib）：

```bash
~/miniconda3/bin/conda create -y -n rdagent python=3.10
~/miniconda3/bin/conda run -n rdagent pip install pandas numpy scipy statsmodels loguru tables
# tables(pytables) 必裝：因子結果以 result.h5 (HDF5) 存取
```

已知陷阱：RD-Agent 的 `CondaConf` 在物件建構時就解析 bin_path，若 conda env
尚未建立會得到空值並快取複用（`qrun`/`python` 找不到、誤用系統 python）。
`rdagent_runner` 以 `BIN_PATH` 環境變數預先注入兩個 env 的 bin 路徑作為 fallback；
若曾卡在此狀態，重啟前順手清 `data/qlab/rdagent_workspace/pickle_cache/`
（pickle cache 會把舊的失敗執行結果快取住，導致修正迴圈不會真正重新執行）。

### 7.4 用台股資料跑 fin_factor（取代內建 A 股）

RD-Agent 的 `fin_factor` 預設從 qlib 下載**中國 A 股**日K（`daily_pv.h5`），
與本專案資料無關。要把因子演化跑在**台股**上，先把台股日K轉成
RD-Agent 期待的 `daily_pv.h5` 格式：

```bash
uv run python -m qlab export-h5              # 正式版（全標的）+ debug 子集（預設 20 檔）
uv run python -m qlab export-h5 --debug-symbols 50
```

輸出（不入版控）：`data/qlab/factor_source_data_tw{,_debug}/daily_pv.h5`。
- index `(datetime, instrument)`，instrument 為 `{market}{symbol}`（如 `TSE2330`）
- 欄位 `$open/$close/$high/$low/$volume/$factor`（$volume=股、$factor=1.0 未復權）
- 只含台股日K（FinMind/TWSE，`data/normalized/daily`），2021/01 ~ 2026/08

產生後 `rdagent_runner` 會自動注入 `FACTOR_COSTEER_DATA_FOLDER`（台股版資料，
RD-Agent 偵測到資料目錄已存在即**跳過 A 股下載**）：

```bash
uv run python -m qlab rdagent fin_factor      # 現在跑的是台股因子演化
```

> 注意：`daily_pv.h5` 未復權（$factor=1.0），跨除權息日算報酬會跳空；
> 且 RD-Agent 首次資料下載後每個工作區是全新 run，先前用 A 股跑出的因子
> （工作區 `RD-Agent_workspace`）不會混入台股 run。

### 7.5 回測宇宙與大盤基準

- **回測宇宙**：`export-tw100` 從正式版 daily_pv.h5 切前 N 檔（不重抓、零誤差），
  固定依**代碼序**（`UniverseSpec.code_first_n`，與 `backtest --top-symbols` 同口徑）。
  舊 `--order turnover`（成交金額排名對照實驗）已於 2026-08-31 實測
  `volume_change_5d` 訊號在流動性前 100 失效後**移除**（結論見 strategy README / AGENTS.md）。
  宇宙語意單一來源為 `qlab/universe.py` 的 `UniverseSpec`。
- **基準**：backtest / OOS 報告的基準為**臺灣大盤 TAIEX 報酬指數（含息）**
  （`qlab/benchmark.py` 經 FinMind `TaiwanStockTotalReturnIndex` 抓取，
  快取 `data/qlab/benchmark_taiex.parquet`，已涵蓋請求區間時離線重跑不再打 API；
  抓不到才 fallback 宇宙等權）。與策略復權含息口徑對齊，超額門檻以大盤為準。

>
> 相關處理已內建於 `qlab/rdagent_runner.py`：空值金鑰覆寫、`openai/` 前綴、
> `ENABLE_RESPONSE_SCHEMA=false`（OpenRouter 免費模型不支援 response_format，走 DeepSeek 式 JSON 降級路徑）、
> 放寬 `MAX_RETRY`/`RETRY_WAIT_SECONDS` 應對免費模型限流、自動偵測 `~/miniconda3`
> 並注入 `CONDA_DEFAULT_ENV=rdagent`（fin_factor 因子程式碼以 conda run 執行）。

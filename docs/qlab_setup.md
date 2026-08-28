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
| 依賴 | `uv sync` 即可（`ai` 依賴群組含 `pyqlib` + `rdagent`，已設為預設群組） |
| Docker | RD-Agent 的量化情境（`quant`、`fin_factor`）**執行階段**需要 Docker 容器跑 Qlib 回測；安裝階段不需要 |

```bash
uv sync                # 重建 .venv（Python 3.12）並安裝全部依賴
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

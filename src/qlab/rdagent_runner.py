"""以共用 .env 驅動 RD-Agent CLI。

RD-Agent 的量化情境（RD-Agent(Q)）會在 Docker 容器中跑 Qlib 回測，
執行前需確認本機 Docker 可用（`docker info`）。金鑰由 kstock Settings
從專案根目錄 .env 讀取後注入子行程環境，不需另外設定。

常用應用：
    uv run python -m qlab rdagent fin_factor   # 財報因子自動演化
    uv run python -m qlab rdagent quant        # 完整量化研究迴圈（需 Docker）
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from kstock.config.settings import Settings, settings


def _force_set(env: dict[str, str], key: str, value: str) -> None:
    """等同 setdefault，但既有的「空字串」也會被覆寫（.env 常見 OPENAI_API_KEY= 空值）。"""
    if not env.get(key):
        env[key] = value


def build_rdagent_env(s: Settings | None = None) -> dict[str, str]:
    """把共用 .env 的金鑰映射成 RD-Agent/LLM 期望的環境變數。

    支援三種後端（擇一即可，優先序：OpenRouter > DeepSeek > OpenAI）：
    OpenRouter 走 OpenAI 相容介面（OPENAI_API_BASE 指向 openrouter.ai/api/v1）。
    """
    st = s or settings
    env = dict(os.environ)
    if st.openai_api_key:
        _force_set(env, "OPENAI_API_KEY", st.openai_api_key)
    if st.deepseek_api_key:
        _force_set(env, "DEEPSEEK_API_KEY", st.deepseek_api_key)
        # RD-Agent 相容 OpenAI 介面，DeepSeek 走 OpenAI 相容端點
        _force_set(env, "OPENAI_API_BASE", st.deepseek_base_url)
        _force_set(env, "CHAT_MODEL", "deepseek-chat")
    if st.openrouter_api_key:
        # OpenRouter：OpenAI 相容，金鑰同時用 OPENROUTER_API_KEY 傳遞（部分模型需要）
        _force_set(env, "OPENROUTER_API_KEY", st.openrouter_api_key)
        _force_set(env, "OPENAI_API_BASE", st.openrouter_base_url)
        _force_set(env, "OPENAI_API_KEY", st.openrouter_api_key)
        # LiteLLM 需 "openai/" 前綴才會用自訂 api_base（OpenRouter 是 OpenAI 相容端點）
        model = st.openrouter_model
        _force_set(env, "CHAT_MODEL", model if model.startswith("openai/") else f"openai/{model}")
        # OpenRouter 模型（尤其 :free）多不支援 response schema（結構化輸出）→ 關閉
        _force_set(env, "ENABLE_RESPONSE_SCHEMA", "false")
        # 部分模型把 <think>...</think> 混進 content，開啟清理
        _force_set(env, "REASONING_THINK_RM", "true")
        # :free 模型有嚴格限流，放寬重試
        _force_set(env, "MAX_RETRY", "20")
        _force_set(env, "RETRY_WAIT_SECONDS", "5")
        # 結構化輸出改走 instructor（MD_JSON：schema 進 prompt + pydantic 驗證失敗帶錯重試），
        # 繞過 OpenRouter 免費模型不支援 response_format 的限制。設 KSTOCK_RDA_INSTRUCTOR_BACKEND=0 可關閉。
        if env.get("KSTOCK_RDA_INSTRUCTOR_BACKEND", "1").lower() not in ("0", "false", "no"):
            _force_set(env, "BACKEND", "qlab.rdagent_instructor.InstructorLiteLLMBackend")
        # fin_factor 的因子程式碼以 `conda run -n <CONDA_DEFAULT_ENV>` 執行（RD-Agent 0.8.0 必填）。
        # 偵測到家目錄的 Miniconda 時自動注入 env 名稱與 PATH（quant 情境走 Docker，不受影響）。
        conda_bin = Path.home() / "miniconda3" / "bin"
        if (conda_bin / "conda").exists():
            _force_set(env, "CONDA_DEFAULT_ENV", env.get("CONDA_DEFAULT_ENV") or "rdagent")
            if str(conda_bin) not in env.get("PATH", ""):
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
        # OpenRouter 無 embedding 端點（DeepSeek 亦無）：embedding 預設改走本地
        # 字元 n-gram 向量（僅用於因子去重相似度）。有真 OpenAI 金鑰時可設 openai。
        _force_set(env, "KSTOCK_EMBEDDING_PROVIDER", env.get("KSTOCK_EMBEDDING_PROVIDER") or "local")
    # 讓 RD-Agent 產物集中在 data/qlab 下（不入版控）
    workdir = Path(settings.data_dir) / "qlab" / "rdagent_workspace"
    workdir.mkdir(parents=True, exist_ok=True)
    env.setdefault("RDA_GIT_HTTP_PROXY", "")
    return env


def run_rdagent(args: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """執行 rdagent CLI，串流輸出（互動式應用需在同一終端觀察進度）。"""
    rdagent = shutil.which("rdagent")
    if rdagent is None:
        raise FileNotFoundError(
            "找不到 rdagent 執行檔；請先 `uv sync`（ai 依賴群組）再試。"
        )
    workdir = cwd or Path(settings.data_dir) / "qlab" / "rdagent_workspace"
    workdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [rdagent, *args],
        env=build_rdagent_env(),
        cwd=str(workdir),
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"rdagent {' '.join(args)} 結束代碼 {proc.returncode}")
    return proc.returncode

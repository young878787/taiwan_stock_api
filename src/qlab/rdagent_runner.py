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


def build_rdagent_env(s: Settings | None = None) -> dict[str, str]:
    """把共用 .env 的金鑰映射成 RD-Agent/LLM 期望的環境變數。

    支援三種後端（擇一即可，優先序：OpenRouter > DeepSeek > OpenAI）：
    OpenRouter 走 OpenAI 相容介面（OPENAI_API_BASE 指向 openrouter.ai/api/v1）。
    """
    st = s or settings
    env = dict(os.environ)
    if st.openai_api_key:
        env.setdefault("OPENAI_API_KEY", st.openai_api_key)
    if st.deepseek_api_key:
        env.setdefault("DEEPSEEK_API_KEY", st.deepseek_api_key)
        # RD-Agent 相容 OpenAI 介面，DeepSeek 走 OpenAI 相容端點
        env.setdefault("OPENAI_API_BASE", st.deepseek_base_url)
        env.setdefault("CHAT_MODEL", "deepseek-chat")
    if st.openrouter_api_key:
        # OpenRouter：OpenAI 相容，金鑰同時用 OPENROUTER_API_KEY 傳遞（部分模型需要）
        env.setdefault("OPENROUTER_API_KEY", st.openrouter_api_key)
        env.setdefault("OPENAI_API_BASE", st.openrouter_base_url)
        env.setdefault("OPENAI_API_KEY", st.openrouter_api_key)
        env.setdefault("CHAT_MODEL", st.openrouter_model)
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

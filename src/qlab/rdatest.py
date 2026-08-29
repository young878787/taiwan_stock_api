"""RD-Agent 結構化輸出實測：用 instructor MD_JSON 打真實模型。

用法（需 .env 已填 OPENROUTER_API_KEY）：

    uv run python -m qlab rdatest
    uv run python -m qlab rdatest --model inclusionai/ling-3.0-flash-fin:free

送出一個小的 pydantic schema（因子規格）請求，驗證模型能否在
instructor 的 prompt 約束 + 驗證重試下產出合法 JSON。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kstock.config.settings import settings as kstock_settings

from qlab.rdagent_instructor import _strip_model_prefix, structured_chat


class FactorSpec(BaseModel):
    """模擬 RD-Agent 因子演化會要求的輸出格式。"""

    name: str = Field(description="因子英文名，snake_case")
    formula: str = Field(description="以收盤價 close 表示的公式，例如 close / close.shift(6) - 1")
    rationale: str = Field(description="選股邏輯說明，繁體中文，50 字以內")


def run_structured_test(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int = 4,
) -> str:
    """對目前設定的模型做一次結構化輸出實測，回傳驗證通過的 JSON 字串。"""
    m = _strip_model_prefix(model or kstock_settings.openrouter_model)
    key = api_key or kstock_settings.openrouter_api_key or kstock_settings.openai_api_key
    base = base_url or (
        kstock_settings.openrouter_base_url
        if kstock_settings.openrouter_api_key
        else "https://openrouter.ai/api/v1"
    )
    if not key:
        raise RuntimeError("缺少金鑰：請在 .env 設定 OPENROUTER_API_KEY（或 OPENAI_API_KEY）")

    messages = [
        {
            "role": "system",
            "content": "你是台股量化研究助理，只能輸出 JSON。",
        },
        {
            "role": "user",
            "content": "設計一個台股短動能因子，依照 schema 輸出。",
        },
    ]
    obj = structured_chat(
        messages,
        FactorSpec,
        model=m,
        api_key=key,
        base_url=base,
        max_retries=max_retries,
    )
    return obj.model_dump_json()

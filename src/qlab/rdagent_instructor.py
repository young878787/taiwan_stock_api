"""以 instructor 強化 RD-Agent 的結構化輸出（針對 OpenRouter 免費模型）。

背景：RD-Agent 大量依賴嚴格 JSON 結構化輸出（因子規格、程式碼生成等），
但 OpenRouter 的免費模型（例：inclusionai/ling-3.0-flash-fin:free）**不支援
response_format / structured_outputs**（已由 OpenRouter models API 證實），
會回長篇報告而非 JSON，導致 RD-Agent 無限重試。

方案：本模組提供 ``InstructorLiteLLMBackend``，繼承 RD-Agent 的
``LiteLLMAPIBackend``。當 RD-Agent 要求結構化輸出（pydantic response_format）
時改走 instructor 的 ``Mode.MD_JSON``：

1. 把 JSON Schema 以 markdown 指令注入 prompt（不依賴伺服端 response_format）；
2. 解析回應中的 JSON（容忍雜訊 / code block）；
3. pydantic 驗證失敗時，instructor 會把「驗證錯誤訊息 + 原回應」回灌模型重試。

載入方式：RD-Agent 的 backend 可插拔（``LLMSettings.backend`` 讀環境變數
``BACKEND``），``qlab/rdagent_runner.py`` 已自動注入
``BACKEND=qlab.rdagent_instructor.InstructorLiteLLMBackend``。

限制：
- 只接管 **chat** 結構化輸出；embedding 沿用 LiteLLM（OpenRouter 無 embedding
  端點，若流程需要 embedding 仍須配 DeepSeek/OpenAI 金鑰）。
- MD_JSON 是 prompt 約束而非生成時 logits 約束（後者只有 outlines + 本地/vLLM
  後端做得到），因此不保證 100% 遵守，但驗證失敗會自動帶錯誤重試。
"""

from __future__ import annotations

import os
import re
from typing import Any, Type

import instructor
import openai
from pydantic import BaseModel

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import JSONParser
from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMAPIBackend

JSON_OBJECT_FORMAT = {"type": "json_object"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# 本地 fallback embedding 維度（僅用於相似度比較，維度只要自洽即可）
_LOCAL_EMBED_DIM = 256


def _local_embed(texts: list[str]) -> list[list[float]]:
    """純 Python 字元 3-gram hash 向量（零依賴、零金鑰、離線可用）。

    用途：RD-Agent 因子去重只算餘弦相似度，對「因子名稱/描述」這類短文字，
    char n-gram TF 向量的相似度排序與真 embedding 高度一致，足以取代。
    維度自洽（同方法生成）即可，與 OpenAI embedding 維度不相容也無妨。
    """
    import hashlib
    import math

    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _LOCAL_EMBED_DIM
        t = " ".join(text.lower().split())
        for i in range(max(0, len(t) - 2)):
            h = int(hashlib.md5(t[i : i + 3].encode("utf-8")).hexdigest(), 16)
            vec[h % _LOCAL_EMBED_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


def _strip_model_prefix(chat_model: str) -> str:
    """RD-Agent 的 CHAT_MODEL 帶 litellm 前綴（``openai/...``）。

    instructor 直接打 OpenAI 相容端點（不走 litellm），需去掉前綴；
    OpenRouter 模型 id 本身含 ``/``（如 inclusionai/ling-3.0-flash-fin:free），不受影響。
    """
    return chat_model.removeprefix("openai/")


def strip_think(text: str) -> str:
    """移除推理模型混進 content 的 ``<think>...</think>`` 區塊。"""
    return _THINK_RE.sub("", text)


def structured_chat(
    messages: list[dict[str, Any]],
    response_model: Type[BaseModel],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int = 4,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseModel:
    """以 instructor MD_JSON 模式呼叫 OpenAI 相容端點，回傳通過 pydantic 驗證的物件。

    ``model`` / ``api_key`` / ``base_url`` 未指定時讀取 RD-Agent 環境
    （``CHAT_MODEL`` / ``OPENAI_API_KEY`` / ``OPENAI_API_BASE``），與
    ``rdagent_runner`` 注入子行程的環境一致。
    """
    model = model or _strip_model_prefix(LITELLM_SETTINGS.chat_model)
    # 空金鑰用佔位符（部分本地 OpenAI 相容伺服器如 Ollama/LM Studio 慣例），
    # 避免 openai client 因缺 OPENAI_API_KEY 直接拋錯
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    base_url = base_url or os.environ.get("OPENAI_API_BASE") or None

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    istr_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    extra: dict[str, Any] = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens

    return istr_client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        **extra,
    )


class InstructorLiteLLMBackend(LiteLLMAPIBackend):
    """結構化輸出走 instructor（MD_JSON），其餘行為沿用 LiteLLM backend。

    - ``response_format`` 為 pydantic BaseModel → instructor MD_JSON（本模組接管）。
    - ``response_format`` 為 ``{"type": "json_object"}`` → 照樣走 LiteLLM
      （不傳 response_format，避免端點不支援），回應後用 RD-Agent 的
      ``JSONParser`` 清理 ``<think>`` 與雜訊、提取 JSON。
    - 其他（無 response_format）→ 原封不動交給父類。
    """

    def supports_response_schema(self) -> bool:
        """由本 backend 自行處理結構化輸出，不把 response_format 轉交 litellm/伺服端。"""
        return False

    def _instructor_retries(self) -> int:
        return int(os.environ.get("KSTOCK_INSTRUCTOR_RETRIES", "4"))

    def _structured_chat(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[BaseModel],
    ) -> tuple[str, str | None]:
        obj = structured_chat(
            messages,
            response_model,
            max_retries=self._instructor_retries(),
            temperature=LITELLM_SETTINGS.chat_temperature,
            max_tokens=LITELLM_SETTINGS.chat_max_tokens,
        )
        # 回傳合法 JSON 字串，RD-Agent 上層的 json.loads / pydantic 驗證可直接通過
        return obj.model_dump_json(), "stop"

    def _json_object_chat(
        self,
        messages: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        # 不傳 response_format 給端點（免費模型不支援），靠 RD-Agent JSONParser 做提取
        content, finish_reason = super()._create_chat_completion_inner_function(
            messages, None, *args, **kwargs
        )
        cleaned = strip_think(content)
        parser = JSONParser(add_json_in_prompt=True)
        return parser.parse(cleaned), finish_reason

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """embedding 供應商可切換（``KSTOCK_EMBEDDING_PROVIDER``）。

        - ``local``（預設）：本地字元 3-gram hash 向量，零金鑰、離線可用。
          OpenRouter 無 embedding 端點（實測 400 encoding_format 錯誤），
          而 RD-Agent 的 embedding 只用於知識庫/因子去重的相似度比較，
          本地向量足以取代。
        - ``openai``：走 litellm（需真的 OpenAI 或 Azure 金鑰，DeepSeek 無 embedding 端點）。
        """
        provider = os.environ.get("KSTOCK_EMBEDDING_PROVIDER", "local").lower()
        if provider == "openai":
            return super()._create_embedding_inner_function(input_content_list)
        return _local_embed(input_content_list)

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def]
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            logger.warning(
                f"[instructor] response_format={response_format.__name__} → 改走 instructor MD_JSON"
                f"（模型 {LITELLM_SETTINGS.chat_model} 不支援伺服端結構化輸出）",
                tag="llm_messages",
            )
            return self._structured_chat(messages, response_format)
        if response_format == JSON_OBJECT_FORMAT:
            return self._json_object_chat(messages, *args, **kwargs)
        return super()._create_chat_completion_inner_function(
            messages, response_format, *args, **kwargs
        )

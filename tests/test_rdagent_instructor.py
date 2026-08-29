"""qlab.rdagent_instructor 離線測試：instructor MD_JSON 結構化輸出 backend。

全部 mock，不打真實 API。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

import qlab.rdagent_instructor as ri
from kstock.config.settings import Settings


class _Factor(BaseModel):
    name: str
    formula: str


def _stub_instructor(monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]) -> None:
    """把 instructor.from_openai 換成 stub client，記錄呼叫參數、回傳合法物件。"""

    class _Completions:
        def create(self, **kwargs: Any) -> _Factor:
            captured.append(kwargs)
            rm = kwargs["response_model"]
            # 依 schema 動態回傳合法物件（rdatest 的 FactorSpec 有三個欄位）
            return rm(**{f: "stub" for f in rm.model_fields})

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    def _fake_from_openai(client: Any, mode: Any) -> _Client:
        assert mode == ri.instructor.Mode.MD_JSON
        return _Client()

    monkeypatch.setattr(ri.instructor, "from_openai", _fake_from_openai)


def test_backend_flags_and_prefix():
    from qlab.rdagent_instructor import InstructorLiteLLMBackend, _strip_model_prefix

    assert InstructorLiteLLMBackend().supports_response_schema() is False
    # litellm 前綴要拆掉，但 OpenRouter 模型 id 本身的 "/" 不能動
    assert _strip_model_prefix("openai/inclusionai/ling-3.0-flash-fin:free") == (
        "inclusionai/ling-3.0-flash-fin:free"
    )
    assert _strip_model_prefix("inclusionai/ling-3.0-flash-fin:free") == (
        "inclusionai/ling-3.0-flash-fin:free"
    )


def test_structured_chat_returns_validated_model(monkeypatch):
    captured: list[dict[str, Any]] = []
    _stub_instructor(monkeypatch, captured)

    obj = ri.structured_chat(
        [{"role": "user", "content": "設計因子"}],
        _Factor,
        model="inclusionai/ling-3.0-flash-fin:free",
        api_key="k-test",
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
    )
    assert obj.name == "stub"
    assert obj.formula == "stub"
    # schema 與訊息有被傳進去
    assert captured[0]["model"] == "inclusionai/ling-3.0-flash-fin:free"
    assert captured[0]["response_model"] is _Factor
    assert captured[0]["max_retries"] == 2


def test_inner_routes_pydantic_to_instructor(monkeypatch):
    captured: list[dict[str, Any]] = []
    _stub_instructor(monkeypatch, captured)

    backend = ri.InstructorLiteLLMBackend()
    content, finish_reason = backend._create_chat_completion_inner_function(
        [{"role": "user", "content": "x"}], _Factor
    )
    # 回傳乾淨 JSON 字串 → RD-Agent 上層 json.loads + pydantic 驗證可直接通過
    assert json.loads(content) == {"name": "stub", "formula": "stub"}
    assert finish_reason == "stop"


def test_json_object_branch_cleans_think_and_extracts(monkeypatch):
    """ling 式髒輸出：<think> 區塊 + 長篇報告 + ```json code block + Python 布林。"""
    dirty = (
        "<think>先推理一下……</think>\n\n"
        "## 因子分析報告\n"
        "以下是您要的結果：\n"
        "```json\n"
        '{"name": "mom6", "formula": "close/close.shift(6)-1", "active": True}\n'
        "```"
    )

    def _fake_super_inner(self: Any, messages: Any, response_format: Any = None, *a: Any, **k: Any):
        # json_object 分支不應把 response_format 轉交端點
        assert response_format is None
        return dirty, "stop"

    monkeypatch.setattr(
        "rdagent.oai.backend.litellm.LiteLLMAPIBackend._create_chat_completion_inner_function",
        _fake_super_inner,
    )

    backend = ri.InstructorLiteLLMBackend()
    content, finish_reason = backend._create_chat_completion_inner_function(
        [{"role": "user", "content": "x"}], {"type": "json_object"}
    )
    assert json.loads(content) == {"name": "mom6", "formula": "close/close.shift(6)-1", "active": True}
    assert finish_reason == "stop"


def test_non_structured_passes_through(monkeypatch):
    def _fake_super_inner(self: Any, messages: Any, response_format: Any = None, *a: Any, **k: Any):
        assert response_format is None
        return "普通文字回應，不是 JSON", "stop"

    monkeypatch.setattr(
        "rdagent.oai.backend.litellm.LiteLLMAPIBackend._create_chat_completion_inner_function",
        _fake_super_inner,
    )
    backend = ri.InstructorLiteLLMBackend()
    content, _ = backend._create_chat_completion_inner_function([{"role": "user", "content": "x"}], None)
    assert content == "普通文字回應，不是 JSON"


def test_runner_injects_instructor_backend(monkeypatch):
    from qlab.rdagent_runner import build_rdagent_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-fin:free")
    monkeypatch.delenv("KSTOCK_RDA_INSTRUCTOR_BACKEND", raising=False)

    env = build_rdagent_env(Settings())
    assert env["BACKEND"] == "qlab.rdagent_instructor.InstructorLiteLLMBackend"
    assert env["ENABLE_RESPONSE_SCHEMA"] == "false"

    # 可用 KSTOCK_RDA_INSTRUCTOR_BACKEND=0 關閉
    monkeypatch.setenv("KSTOCK_RDA_INSTRUCTOR_BACKEND", "0")
    env = build_rdagent_env(Settings())
    assert env.get("BACKEND") != "qlab.rdagent_instructor.InstructorLiteLLMBackend"


def test_rdatest_uses_instructor(monkeypatch):
    """rdatest CLI：mock instructor 後應印出驗證通過的 JSON。"""
    import qlab.rdatest as rt

    captured: list[dict[str, Any]] = []
    _stub_instructor(monkeypatch, captured)

    out = rt.run_structured_test(
        model="openai/inclusionai/ling-3.0-flash-fin:free",
        api_key="k-test",
        base_url="https://openrouter.ai/api/v1",
    )
    parsed = json.loads(out)
    assert set(parsed) == {"name", "formula", "rationale"}
    # 模型 id 前綴已被清理
    assert captured[0]["model"] == "inclusionai/ling-3.0-flash-fin:free"


def test_rdatest_raises_without_key(monkeypatch):
    import qlab.rdatest as rt
    from types import SimpleNamespace

    # 隔離真實 .env：把模組級 settings 換成無金鑰版本
    monkeypatch.setattr(
        rt,
        "kstock_settings",
        SimpleNamespace(openrouter_api_key="", openai_api_key="", openrouter_base_url=""),
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        rt.run_structured_test(model="m", api_key="", base_url=None)

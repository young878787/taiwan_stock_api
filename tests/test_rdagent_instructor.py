"""qlab.rdagent_instructor 離線測試：instructor MD_JSON 結構化輸出 backend。

全部 mock，不打真實 API。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

import qlab.rdagent_instructor as ri
from conftest import make_daily_bars
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


def test_runner_injects_conda_env_when_available(monkeypatch):
    """有家目錄 Miniconda 時應注入 CONDA_DEFAULT_ENV 與 PATH（fin_factor 必要）。"""
    from pathlib import Path

    from qlab.rdagent_runner import build_rdagent_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    env = build_rdagent_env(Settings())
    if (Path.home() / "miniconda3" / "bin" / "conda").exists():
        assert env["CONDA_DEFAULT_ENV"] == "rdagent"
        assert str(Path.home() / "miniconda3" / "bin") in env["PATH"]


def test_local_embed_similarity_and_dims():
    """本地 fallback embedding：維度一致、相似文字比不相關文字接近。"""
    import math

    vecs = ri._local_embed(
        [
            "factor_name: 5-day momentum\nfactor_description: short term price momentum",
            "factor_name: 5-day momentum reversal\nfactor_description: short term momentum signal",
            "完全無關的中文財報品質因子描述，談論現金流量與負債比率。",
        ]
    )
    assert all(len(v) == ri._LOCAL_EMBED_DIM for v in vecs)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0) for v in vecs)  # 已歸一化

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2])  # 相近描述 > 無關描述


def test_backend_embedding_provider(monkeypatch):
    """KSTOCK_EMBEDDING_PROVIDER=local 走本地向量；openai 走父類 litellm。"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    backend = ri.InstructorLiteLLMBackend()

    # local（預設）
    monkeypatch.delenv("KSTOCK_EMBEDDING_PROVIDER", raising=False)
    out = backend._create_embedding_inner_function(["因子A", "因子B"])
    assert len(out) == 2 and len(out[0]) == ri._LOCAL_EMBED_DIM

    # openai → 走父類（mock 掉 litellm embedding 呼叫）
    monkeypatch.setenv("KSTOCK_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(
        "rdagent.oai.backend.litellm.LiteLLMAPIBackend._create_embedding_inner_function",
        lambda self, texts: [[0.1, 0.2] for _ in texts],
    )
    out = backend._create_embedding_inner_function(["因子A"])
    assert out[0] == [0.1, 0.2]


def test_runner_injects_embedding_provider(monkeypatch):
    from qlab.rdagent_runner import build_rdagent_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.delenv("KSTOCK_EMBEDDING_PROVIDER", raising=False)
    env = build_rdagent_env(Settings())
    assert env["KSTOCK_EMBEDDING_PROVIDER"] == "local"


def test_factor_report_collects_and_renders(tmp_path):
    """factor-report：從假工作區收集 result.h5 並產出 Markdown 報告。"""
    import pandas as pd

    from qlab.factor_report import build_report, collect_results, export_report

    ws = tmp_path / "RD-Agent_workspace"
    # 成功因子
    d1 = ws / "aaaa1111"
    d1.mkdir(parents=True)
    idx = pd.MultiIndex.from_tuples(
        [("2024-01-02", "2330"), ("2024-01-03", "2330")], names=["datetime", "instrument"]
    )
    pd.DataFrame({"Mom_5D": [0.01, None]}, index=idx).to_hdf(d1 / "result.h5", key="data")
    # 未執行成功的目錄（無 result.h5）→ 應被忽略
    (ws / "bbbb2222").mkdir()

    results = collect_results(ws)
    assert len(results) == 1
    assert results[0].factor_name == "Mom_5D"
    assert results[0].n_rows == 2
    assert results[0].n_instruments == 1
    assert results[0].stats["na_ratio"] == pytest.approx(0.5)

    report = build_report(results, n_total=10)
    assert "# RD-Agent fin_factor 因子結果報告" in report
    assert "`Mom_5D`" in report
    assert "演化進度 10 個" in report

    out = export_report(workspace=ws, output=tmp_path / "report.md")
    assert "因子摘要" in out.read_text(encoding="utf-8")


def test_export_h5_writes_tw_daily_pv(store, test_settings, monkeypatch, tmp_path):
    """export-h5：台股 parquet → RD-Agent 期待格式 daily_pv.h5。"""
    import pandas as pd

    import qlab.export_h5 as eh
    from qlab.export_h5 import export_daily_pv

    store.write_normalized(
        "daily",
        make_daily_bars("2330", ["2024-01-02", "2024-01-03"], [590.0, 598.0]),
    )
    store.write_normalized(
        "daily", make_daily_bars("0050", ["2024-01-02"], [30.0])
    )

    # 把 kstock settings 指到 test_settings（tmp_path 隔離輸出）
    monkeypatch.setattr(eh, "kstock_settings", test_settings)

    out_dir = tmp_path / "factor_source_data_tw"
    dbg_dir = tmp_path / "factor_source_data_tw_debug"
    full, debug = export_daily_pv(output_dir=out_dir, debug_dir=dbg_dir, debug_symbols=1)

    # 正式版：2 檔
    df = pd.read_hdf(full.path, key="data")
    assert list(df.columns) == ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    assert df.index.names == ["datetime", "instrument"]
    assert df.index.get_level_values("instrument").unique().tolist() == ["TSE0050", "TSE2330"]  # market+symbol
    assert (df["$factor"] == 1.0).all()
    assert df["$volume"].iloc[0] == 1_000_000  # 股
    assert str(df.index.get_level_values("datetime").min().date()) == "2024-01-02"

    # debug 版：只取 1 檔
    d2 = pd.read_hdf(debug.path, key="data")
    assert d2.index.get_level_values("instrument").nunique() == 1

    # README 一併寫出
    assert (out_dir / "README.md").exists()


def test_runner_injects_tw_factor_folder(test_settings, monkeypatch):
    """台股版 daily_pv.h5 存在時才注入 FACTOR_CoSTEER_DATA_FOLDER。"""
    from qlab.rdagent_runner import build_rdagent_env

    _tw = test_settings.data_dir / "qlab" / "factor_source_data_tw"
    _dbg = test_settings.data_dir / "qlab" / "factor_source_data_tw_debug"
    _tw.mkdir(parents=True, exist_ok=True)
    _dbg.mkdir(parents=True, exist_ok=True)
    (_tw / "daily_pv.h5").touch()
    (_dbg / "daily_pv.h5").touch()

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("qlab.rdagent_runner.settings", test_settings)
    env = build_rdagent_env(test_settings)
    assert env["FACTOR_COSTEER_DATA_FOLDER"] == str(_tw)
    assert env["FACTOR_COSTEER_DATA_FOLDER_DEBUG"] == str(_dbg)

    # rdagent_workspace 位置跟著 test_settings（data_dir）走
    assert env["RDA_GIT_HTTP_PROXY"] == ""


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

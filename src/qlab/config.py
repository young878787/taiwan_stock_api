"""qlab 設定：沿用 kstock Settings（同一份 .env、同一個 data/ 根目錄）。

金鑰（OPENAI_API_KEY / DEEPSEEK_API_KEY）由 kstock Settings 統一管理，
本檔只補上 Qlib / RD-Agent 專屬的路径與模型設定。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from kstock.config.settings import Settings, settings as kstock_settings


@dataclass(frozen=True)
class QlabSettings:
    project_root: Path = field(default_factory=lambda: kstock_settings.project_root)
    data_dir: Path = field(default_factory=lambda: kstock_settings.data_dir)
    # Qlib bin 格式資料集輸出位置（預設 <data>/qlab/qlib_data）
    provider_dir: Path = field(default_factory=lambda: kstock_settings.data_dir / "qlab" / "qlib_data")
    # RD-Agent / LLM 金鑰（與 kstock 共用 .env）
    openai_api_key: str = field(default_factory=lambda: kstock_settings.openai_api_key)
    deepseek_api_key: str = field(default_factory=lambda: kstock_settings.deepseek_api_key)
    deepseek_base_url: str = field(default_factory=lambda: kstock_settings.deepseek_base_url)
    openrouter_api_key: str = field(default_factory=lambda: kstock_settings.openrouter_api_key)
    openrouter_base_url: str = field(default_factory=lambda: kstock_settings.openrouter_base_url)
    openrouter_model: str = field(default_factory=lambda: kstock_settings.openrouter_model)
    # Qlib 實驗輸出（回測紀錄、MLflow runs）
    experiment_dir: Path = field(default_factory=lambda: kstock_settings.data_dir / "qlab" / "experiments")

    def __post_init__(self) -> None:
        env = os.environ
        if env.get("QLAB_PROVIDER_DIR"):
            object.__setattr__(self, "provider_dir", Path(env["QLAB_PROVIDER_DIR"]))
        if env.get("QLAB_EXPERIMENT_DIR"):
            object.__setattr__(self, "experiment_dir", Path(env["QLAB_EXPERIMENT_DIR"]))


def qlab_settings(kstock: Settings | None = None) -> QlabSettings:
    """以（可選注入的）kstock Settings 建立 QlabSettings，方便測試時用 tmp_path 隔離。"""
    s = kstock or kstock_settings
    return QlabSettings(
        project_root=s.project_root,
        data_dir=s.data_dir,
        openai_api_key=s.openai_api_key,
        deepseek_api_key=s.deepseek_api_key,
        deepseek_base_url=s.deepseek_base_url,
        openrouter_api_key=s.openrouter_api_key,
        openrouter_base_url=s.openrouter_base_url,
        openrouter_model=s.openrouter_model,
    )

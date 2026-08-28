from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default=PROJECT_ROOT)
    data_dir: Path = field(default=PROJECT_ROOT / "data")
    raw_dir: Path = field(default=PROJECT_ROOT / "data" / "raw")
    normalized_dir: Path = field(default=PROJECT_ROOT / "data" / "normalized")
    finmind_token: str = field(default="")
    finmind_base_url: str = field(default="https://api.finmindtrade.com/api/v4")
    finmind_volume_unit: str = field(default="shares")
    twse_base_url: str = field(default="https://www.twse.com.tw")
    shioaji_key: str = field(default="")
    shioaji_secret: str = field(default="")
    # AI 金鑰（qlab 子專案：Qlib + RD-Agent；與 kstock 共用同一份 .env）
    openai_api_key: str = field(default="")
    deepseek_api_key: str = field(default="")
    deepseek_base_url: str = field(default="https://api.deepseek.com/v1")
    openrouter_api_key: str = field(default="")
    openrouter_base_url: str = field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = field(default="deepseek/deepseek-chat-v3-0324")

    def __post_init__(self) -> None:
        env = os.environ
        if env.get("KSTOCK_DATA_DIR"):
            object.__setattr__(self, "data_dir", Path(env["KSTOCK_DATA_DIR"]))
        object.__setattr__(self, "raw_dir", Path(env.get("KSTOCK_RAW_DIR") or self.data_dir / "raw"))
        object.__setattr__(self, "normalized_dir", Path(env.get("KSTOCK_NORMALIZED_DIR") or self.data_dir / "normalized"))
        if env.get("FINMIND_TOKEN"):
            object.__setattr__(self, "finmind_token", env["FINMIND_TOKEN"])
        if env.get("FINMIND_BASE_URL"):
            object.__setattr__(self, "finmind_base_url", env["FINMIND_BASE_URL"])
        if env.get("FINMIND_VOLUME_UNIT"):
            object.__setattr__(self, "finmind_volume_unit", env["FINMIND_VOLUME_UNIT"])
        if env.get("TWSE_BASE_URL"):
            object.__setattr__(self, "twse_base_url", env["TWSE_BASE_URL"])
        if env.get("SHIOAJI_KEY"):
            object.__setattr__(self, "shioaji_key", env["SHIOAJI_KEY"])
        if env.get("SHIOAJI_SECRET"):
            object.__setattr__(self, "shioaji_secret", env["SHIOAJI_SECRET"])
        if env.get("OPENAI_API_KEY"):
            object.__setattr__(self, "openai_api_key", env["OPENAI_API_KEY"])
        if env.get("DEEPSEEK_API_KEY"):
            object.__setattr__(self, "deepseek_api_key", env["DEEPSEEK_API_KEY"])
        if env.get("DEEPSEEK_BASE_URL"):
            object.__setattr__(self, "deepseek_base_url", env["DEEPSEEK_BASE_URL"])
        if env.get("OPENROUTER_API_KEY"):
            object.__setattr__(self, "openrouter_api_key", env["OPENROUTER_API_KEY"])
        if env.get("OPENROUTER_BASE_URL"):
            object.__setattr__(self, "openrouter_base_url", env["OPENROUTER_BASE_URL"])
        if env.get("OPENROUTER_MODEL"):
            object.__setattr__(self, "openrouter_model", env["OPENROUTER_MODEL"])

    @property
    def finmind_data_url(self) -> str:
        return f"{self.finmind_base_url}/data"


settings = Settings()

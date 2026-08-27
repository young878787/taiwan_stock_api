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

    @property
    def finmind_data_url(self) -> str:
        return f"{self.finmind_base_url}/data"


settings = Settings()

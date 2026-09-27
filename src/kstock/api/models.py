"""日 K 的 HTTP 資料契約，沿用 normalized daily 欄位。"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Symbol = Annotated[str, Field(pattern=r"^[0-9A-Z]{4,12}$")]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]


class DailyValues(BaseModel):
    """PUT 完整替換的內容；自然主鍵由 URL 決定。"""

    model_config = ConfigDict(extra="forbid")

    market: Literal["TSE"] = "TSE"
    open: NonnegativeFloat | None
    high: NonnegativeFloat | None
    low: NonnegativeFloat | None
    close: NonnegativeFloat | None
    volume_shares: NonnegativeInt | None
    turnover_twd: NonnegativeFloat | None
    trade_count: NonnegativeInt | None
    source: Literal["twse", "manual"] = "manual"

    @model_validator(mode="after")
    def valid_prices(self):
        prices = [value for value in (self.open, self.high, self.low, self.close) if value is not None]
        if self.high is not None and any(value > self.high for value in prices):
            raise ValueError("high 必須大於或等於其他價格")
        if self.low is not None and any(value < self.low for value in prices):
            raise ValueError("low 必須小於或等於其他價格")
        return self


class DailyBar(DailyValues):
    symbol: Symbol
    date: date


class DailyPage(BaseModel):
    items: list[DailyBar]
    total: int
    limit: int
    offset: int


class APIError(BaseModel):
    detail: str

"""研究層宇宙 spec：回測宇宙的單一可重現定義（WP2/WP5）。

核心命題：資料層只放資料，回測宇宙由研究層的 spec 定義且可重現。
此模組把原本散落三處的宇宙語意收斂到單一 :class:`UniverseSpec`：

- ``export-h5.slice_top_symbols``（export-tw100）＝ ``code_first_n`` 的切片實作；
- 回測 CLI ``--top-symbols N``（``qlab backtest`` / ``exec-timing``）＝
  CLI int 向後相容別名，解析成 ``UniverseSpec.code_first_n(n)``；
- RD-Agent ``--universe tw100`` ＝ ``UniverseSpec.code_first_n(100)`` 的別名
  （與回測口徑一致，見 :mod:`qlab.rdagent_runner`）。

``pit_liquidity``（WP5）為逐日 point-in-time 資格矩陣：
每個再平衡日以「當時」的滾動流動性排名取前 N 檔 + 上市滿 N 日 + 下市自動退場，
同時治好幸存者偏差（下市股進入歷史橫截面）與宇宙前視（排名不用未來資訊）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class UniverseSpec:
    """回測宇宙的宣告式定義（可序列化、可重現）。

    - ``kind="all"``：資料集全部標的。
    - ``kind="code_first_n"``：instrument 代碼序（``sorted(instruments)[:n]``，
      OTC < TSE）取前 ``n`` 檔——台股研究的正式口徑（tw100 即此）。
    - ``kind="pit_liquidity"``：point-in-time 流動性過濾（見 :func:`pit_eligibility_mask`）。
    """

    kind: str  # "all" | "code_first_n" | "pit_liquidity"
    n: int | None = None
    lookback_days: int = 60
    min_listed_days: int = 120

    @staticmethod
    def all() -> "UniverseSpec":
        """資料集全部標的。"""
        return UniverseSpec(kind="all")

    @staticmethod
    def code_first_n(n: int) -> "UniverseSpec":
        """instrument 代碼序前 ``n`` 檔（正式口徑）。"""
        return UniverseSpec(kind="code_first_n", n=n)

    @staticmethod
    def pit_liquidity(
        top_n: int, lookback_days: int = 60, min_listed_days: int = 120
    ) -> "UniverseSpec":
        """PIT 流動性過濾（近 ``lookback_days`` 日均成交金額前 ``top_n`` 檔、上市滿 ``min_listed_days`` 日）。

        資格逐日判定（見 :func:`pit_eligibility_mask`）：下市股序列結束自動退場，
        新上市股滿 ``min_listed_days`` 個有價交易日才入宇宙。
        """
        return UniverseSpec(
            kind="pit_liquidity", n=top_n, lookback_days=lookback_days, min_listed_days=min_listed_days
        )

    def describe(self) -> str:
        """繁體中文可重現描述，供回測報告 params 記錄宇宙定義。"""
        if self.kind == "all":
            return "all（資料集全部標的）"
        if self.kind == "code_first_n":
            return f"code_first_n(n={self.n})（代碼序前 {self.n} 檔）"
        if self.kind == "pit_liquidity":
            return (
                f"pit_liquidity(top_n={self.n}, lookback_days={self.lookback_days}, "
                f"min_listed_days={self.min_listed_days})"
            )
        raise ValueError(f"未知的宇宙 kind：{self.kind}")

    def slice_instruments(self, instruments: list[str]) -> list[str]:
        """對資料集標的清單套用宇宙切片，回傳最終回測宇宙。

        ``pit_liquidity`` 不支援靜態切片（需逐日 eligible mask，見
        :func:`pit_eligibility_mask`）。
        """
        if self.kind == "all":
            return list(instruments)
        if self.kind == "code_first_n":
            return sorted(instruments)[: self.n]
        raise NotImplementedError(
            "pit_liquidity 的切片尚未支援：PIT 橫截面過濾為逐日資格矩陣"
            "（用 qlab.universe.pit_eligibility_mask），靜態切片語意不適用"
        )


def pit_eligibility_mask(
    price_df: "pd.DataFrame",
    top_n: int,
    lookback_days: int = 60,
    min_listed_days: int = 120,
) -> tuple["pd.DataFrame", "pd.Series"]:
    """point-in-time 流動性資格矩陣（WP5 / 階段 C3）。

    以「當時可得」的資料逐日判定資格，無前視：

    - 流動性：``min(lookback_days, ...)`` 日的滾動平均成交金額（proxy = close × volume；
      h5 內無 turnover 欄位，兩者排名等價——比例同尺度）橫截面排名前 ``top_n``。
    - 上市天數：累積有價資料日 ≥ ``min_listed_days``（新上市股先出場後才入宇宙）。
    - 下市：序列結束（close/volume 缺值）→ 成交金額 proxy 為 NaN → 自動退場。

    回傳 ``(eligible_mask, universe_size)``：mask 對齊 ``price_df`` 的日期×標的
    （值=bool）；universe_size 為逐日合格檔數序列。

    ``price_df``：daily_pv.h5 格式（MultiIndex (datetime, instrument)、欄位 $close/$volume）。
    """
    close = price_df["$close"].unstack("instrument")
    vol = price_df["$volume"].unstack("instrument")
    # 停牌 0 收盤/0 量 → NaN（不計入流動性與資料日數）
    tradable = (close > 0) & (vol > 0)
    amt = (close * vol).where(tradable)
    amt_mean = amt.rolling(lookback_days, min_periods=1).mean()
    rank = amt_mean.rank(axis=1, ascending=False, na_option="keep")
    liquidity_ok = rank <= top_n
    listed_ok = tradable.cumsum() >= min_listed_days
    # 當日可交易（下市/停牌當日即退場；否則滾動均值會殘留 lookback_days 日）
    eligible = (liquidity_ok & listed_ok & tradable).reindex(close.index).fillna(False)
    eligible = eligible.astype(bool)
    universe_size = eligible.sum(axis=1)
    return eligible, universe_size


# pd 僅供型別標註（qlab 子專案可用 pandas）
try:  # pragma: no cover
    import pandas as pd  # noqa: F401
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

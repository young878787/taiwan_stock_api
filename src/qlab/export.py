"""把 kstock normalized daily（Parquet）轉成 Qlib bin 格式資料集。

輸出結構（Qlib 官方 provider_uri 佈局）：
    <provider_dir>/
        calendars/day.txt        # 交易日曆（YYYY-MM-DD，全部交易日聯集）
        instruments/all.txt      # <SYMBOL>\t<start>\t<end>（日曆起始/結束）
        features/<SYMBOL>/open.day.bin ...  # float32：[start_idx, end_idx, 值...]

與 Qlib 0.9.x FileFeatureStorage 相同格式：小端 float32，第一個值是日曆索引起點，
後面依日曆對齊；停牌/缺日補 NaN。

成交量沿襲 kstock 不變量：一律為「股」（volume_shares → $volume）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from qlab.config import QlabSettings, qlab_settings

# kstock normalized 欄位 → Qlib $field 名稱
FIELD_MAP: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume_shares": "volume",
    "turnover_twd": "turnover",
}


@dataclass(frozen=True)
class ExportReport:
    provider_dir: Path
    n_symbols: int
    n_calendar_days: int
    start: str
    end: str

    def __str__(self) -> str:  # pragma: no cover - 顯示用
        return (
            f"Qlib 資料集已輸出：{self.provider_dir}\n"
            f"  symbols={self.n_symbols}, 交易日={self.n_calendar_days}, "
            f"區間={self.start} ~ {self.end}"
        )


class QlibDataExporter:
    def __init__(self, settings_: QlabSettings | None = None) -> None:
        self.s = settings_ or qlab_settings()

    def _load_daily(self) -> pl.DataFrame:
        glob = str(self.s.data_dir / "normalized" / "daily" / "year=*" / "*.parquet")
        files = sorted(self.s.data_dir.joinpath("normalized", "daily").glob("year=*/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"找不到 daily parquet（{glob}）。請先跑 kstock 的 DailyUpdatePipeline。"
            )
        df = pl.read_parquet(files)
        # 寫入邏輯本身已冪等去重，這裡再做一次保險：同 (symbol, date) 取最後一筆
        df = (
            df.sort(["symbol", "date"])
            .unique(subset=["symbol", "date"], keep="last")
            .with_columns(pl.col("symbol").str.to_uppercase())
        )
        return df

    def export(self) -> ExportReport:
        df = self._load_daily()
        calendar = sorted(df["date"].unique().to_list())
        n_days = len(calendar)
        start_s, end_s = str(calendar[0]), str(calendar[-1])

        provider = self.s.provider_dir
        (provider / "calendars").mkdir(parents=True, exist_ok=True)
        (provider / "instruments").mkdir(parents=True, exist_ok=True)
        features_dir = provider / "features"

        # 1) 交易日曆
        (provider / "calendars" / "day.txt").write_text(
            "\n".join(str(d) for d in calendar) + "\n", encoding="utf-8"
        )

        # 2) instruments/all.txt（Qlib 依此檔符號大小排序）
        inst_lines: list[str] = []
        for sym in sorted(df["symbol"].unique().to_list()):
            sub = df.filter(pl.col("symbol") == sym)
            inst_lines.append(f"{sym}\t{sub['date'].min()}\t{sub['date'].max()}")
        (provider / "instruments" / "all.txt").write_text(
            "\n".join(inst_lines) + "\n", encoding="utf-8"
        )

        # 3) features/<SYM>/<field>.day.bin（值對齊完整日曆，缺日 NaN）
        cal_df = pl.DataFrame({"date": calendar}).with_columns(
            pl.arange(0, n_days).alias("_idx")
        )
        for sym in df["symbol"].unique().to_list():
            sym_dir = features_dir / sym
            sym_dir.mkdir(parents=True, exist_ok=True)
            sub = df.filter(pl.col("symbol") == sym)
            aligned = cal_df.join(
                sub.select(["date", *FIELD_MAP.keys()]), on="date", how="left"
            ).sort("_idx")
            for kstock_col, qlib_field in FIELD_MAP.items():
                # null/NaN 對齊後即為缺日，float64 → float32 時保留 NaN
                values = aligned[kstock_col].cast(pl.Float64).to_numpy(allow_copy=True)
                # qlib 0.9.x FileFeatureStorage 格式：[start_index, 值...]（單一 header，
                # 資料長度 = size/4 - 1，資料對應 calendar[start : start+len]）
                arr = np.empty(n_days + 1, dtype="<f4")
                arr[0] = 0  # start index in calendar
                arr[1:] = values.astype("<f4", copy=False)
                arr.astype("<f4", copy=False).tofile(sym_dir / f"{qlib_field}.day.bin")

        return ExportReport(
            provider_dir=provider,
            n_symbols=df["symbol"].n_unique(),
            n_calendar_days=n_days,
            start=start_s,
            end=end_s,
        )

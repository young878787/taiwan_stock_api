"""資料校驗（FinMind vs TWSE 交叉比較）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

VALID_FIELDS = ("close", "volume_shares", "turnover_twd")


@dataclass
class ValidationReport:
    checked: int = 0
    mismatches: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.checked > 0 and not self.mismatches

    @property
    def summary(self) -> str:
        return f"checked={self.checked}, mismatches={len(self.mismatches)}"


def compare_sources(
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    fields: tuple[str, ...] = ("close", "volume_shares"),
    tolerance: float = 0.005,
) -> ValidationReport:
    """以 primary 為基準、secondary 為對照，比較指定欄位（比例誤差 ≤ tolerance）。"""
    report = ValidationReport()
    if primary.is_empty() or secondary.is_empty():
        return report

    for field_name in fields:
        if field_name not in primary.columns or field_name not in secondary.columns:
            continue
        joined = primary.select(["symbol", "date", field_name]).rename({field_name: f"{field_name}_a"})
        other = secondary.select(["symbol", "date", field_name]).rename({field_name: f"{field_name}_b"})
        compared = joined.join(other, on=["symbol", "date"], how="inner")
        report.checked += compared.height
        if compared.height == 0:
            continue
        a = compared[f"{field_name}_a"].to_list()
        b = compared[f"{field_name}_b"].to_list()
        for row in compared.iter_rows(named=True):
            av = row[f"{field_name}_a"]
            bv = row[f"{field_name}_b"]
            if av is None or bv is None or av == bv:
                continue
            base = abs(av)
            error = abs(av - bv)
            relative = error / base if base > 0 else error
            if relative > tolerance:
                report.mismatches.append(
                    {
                        "symbol": row["symbol"],
                        "date": str(row["date"]),
                        "field": field_name,
                        "primary": av,
                        "secondary": bv,
                        "error": error,
                    }
                )
    return report
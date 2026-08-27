from __future__ import annotations

VOLUME_UNIT_MULTIPLIER = {
    "share": 1,
    "shares": 1,
    "股": 1,
    "lot": 1000,
    "lots": 1000,
    "張": 1000,
}


def _to_number(value) -> int | float:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value
    text = str(value).replace(",", "").strip()
    if not text:
        return 0
    return float(text)


def volume_to_shares(value, unit: str = "lots") -> int:
    """將任一來源的成交量單位轉成股數（volume_shares）。

    unit: shares/股 → ×1；lots/張 → ×1000。
    """
    key = str(unit).strip().lower()
    if key not in VOLUME_UNIT_MULTIPLIER:
        raise ValueError(f"未知成交量單位: {unit!r}（支援: {sorted(VOLUME_UNIT_MULTIPLIER)}）")
    return int(round(_to_number(value) * VOLUME_UNIT_MULTIPLIER[key]))


def normalize_volume_shares(value, unit: str = "lots") -> int:
    return volume_to_shares(value, unit=unit)


def shares_to_lots(shares) -> float:
    return _to_number(shares) / 1000.0
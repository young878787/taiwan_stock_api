"""UniverseSpec：研究層宇宙 spec 的語意測試（WP2）。"""

import pandas as pd
import pytest


def test_describe_all():
    from qlab.universe import UniverseSpec

    assert UniverseSpec.all().describe() == "all（資料集全部標的）"


def test_describe_code_first_n():
    from qlab.universe import UniverseSpec

    assert UniverseSpec.code_first_n(100).describe() == "code_first_n(n=100)（代碼序前 100 檔）"


def test_describe_pit_liquidity():
    from qlab.universe import UniverseSpec

    spec = UniverseSpec.pit_liquidity(top_n=50, lookback_days=30, min_listed_days=180)
    assert spec.describe() == "pit_liquidity(top_n=50, lookback_days=30, min_listed_days=180)"
    # 預設值
    assert UniverseSpec.pit_liquidity(10).describe() == (
        "pit_liquidity(top_n=10, lookback_days=60, min_listed_days=120)"
    )


def test_slice_all_returns_original_list():
    from qlab.universe import UniverseSpec

    insts = ["OTC1234", "TSE2330", "TSE0050"]
    assert UniverseSpec.all().slice_instruments(insts) == insts


def test_slice_code_first_n_sorted_prefix():
    from qlab.universe import UniverseSpec

    insts = ["TSE2330", "OTC1234", "TSE0050", "OTC5678"]
    # 排序後 OTC1234 < OTC5678 < TSE0050 < TSE2330 → 前 2 檔
    assert UniverseSpec.code_first_n(2).slice_instruments(insts) == ["OTC1234", "OTC5678"]


def test_slice_code_first_n_leading_zero_order():
    from qlab.universe import UniverseSpec

    # 前導零：字串排序 TSE1101 < TSE2330 < TSE2454（非數值序 1101 < 2330 < 2454 同樣成立，
    # 但字串序保證 0050 類前導零不會被當數字吃掉）
    insts = ["TSE1101", "TSE2330", "TSE2454"]
    assert UniverseSpec.code_first_n(2).slice_instruments(insts) == ["TSE1101", "TSE2330"]
    assert UniverseSpec.code_first_n(2).slice_instruments(["TSE0050", "TSE1101", "TSE2330"]) == [
        "TSE0050",
        "TSE1101",
    ]


def test_slice_pit_liquidity_not_implemented():
    from qlab.universe import UniverseSpec

    spec = UniverseSpec.pit_liquidity(10)
    # WP5 已實作：PIT 不支援靜態切片，導向逐日資格矩陣 pit_eligibility_mask
    with pytest.raises(NotImplementedError, match="pit_eligibility_mask"):
        spec.slice_instruments(["TSE2330"])


def test_resolve_universe_spec_compat_alias():
    """CLI --top-symbols（int）→ code_first_n 向後相容別名；顯式 spec 優先。"""
    from qlab.factor_backtest import _resolve_universe_spec
    from qlab.universe import UniverseSpec

    assert _resolve_universe_spec(100, None) == UniverseSpec.code_first_n(100)
    assert _resolve_universe_spec(None, None) == UniverseSpec.all()
    explicit = UniverseSpec.pit_liquidity(20)
    assert _resolve_universe_spec(100, explicit) is explicit


# ---- WP5：pit_eligibility_mask ----


def _pit_price_df(closes: dict[str, list[float]], vols: dict[str, list[float]], dates) -> pd.DataFrame:
    """closes/vols: {instrument: 逐日值} → daily_pv.h5 形狀的長表。"""
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), sorted(closes)], names=["datetime", "instrument"])
    data = {}
    for k in ("$close", "$volume"):
        src = closes if k == "$close" else vols
        data[k] = [src[inst][i] for i in range(len(dates)) for inst in sorted(closes)]
    return pd.DataFrame(data, index=idx)


def test_pit_eligibility_mask_rules():
    import numpy as np

    from qlab.universe import pit_eligibility_mask

    dates = pd.bdate_range("2024-01-01", periods=250).date
    # A：全程高流動；B：中途下市（130 日後序列結束）；C：新上市（80 日才有資料）、流動性低
    a_close = [10.0] * 250
    a_vol = [1_000_000] * 250
    b_close = [10.0] * 130 + [np.nan] * 120
    b_vol = [900_000] * 130 + [np.nan] * 120
    c_close = [np.nan] * 80 + [5.0] * 170
    c_vol = [np.nan] * 80 + [10_000] * 170
    pdf = _pit_price_df(
        {"A": a_close, "B": b_close, "C": c_close},
        {"A": a_vol, "B": b_vol, "C": c_vol},
        dates,
    )
    mask, size = pit_eligibility_mask(pdf, top_n=2, lookback_days=60, min_listed_days=120)

    # A：上市滿 120 日（第 120 日）起全程有資格
    assert mask["A"].iloc[:119].sum() == 0
    assert mask["A"].iloc[119:].all()
    # B：上市滿 120 日（第 120 日）到序列結束（第 130 日）→ 資格在第 131 日消失（下市退場）
    assert bool(mask["B"].iloc[119]) is True
    assert bool(mask["B"].iloc[130]) is False
    # C：A、B 都有資格時（前 130 日）C 排名第 3 → 不入場；B 下市（第 131 日起）遞補流動性位，
    # 但 C 的上市天數累積到第 120 個有價日（iloc 199）才滿 120 → 遞補日 = iloc 199
    assert not mask["C"].iloc[:199].any()
    assert mask["C"].iloc[199:].all()
    # top_n=2 但只有 A/B 有資格 → 每日合格檔數 ≤ 2
    assert size.max() <= 2


def test_pit_eligibility_mask_no_lookahead():
    from qlab.universe import pit_eligibility_mask

    dates = pd.bdate_range("2024-01-01", periods=160).date
    # 兩檔流動性互換：前段 A 大（1-90 日）、後段 B 大（91-160 日）；資格必須只反映「過去」
    a_vol = [2_000_000] * 90 + [10_000] * 70
    b_vol = [10_000] * 90 + [2_000_000] * 70
    close = [10.0] * 160
    pdf = _pit_price_df({"A": close, "B": close}, {"A": a_vol, "B": b_vol}, dates)
    mask, _ = pit_eligibility_mask(pdf, top_n=1, lookback_days=40, min_listed_days=20)
    # 無前視核心斷言：互換發生在第 91 日，第 100 日（滾動 40 日 = 30 高量 A + 10 低量 A）
    # 仍是 A 佔優 → mask 只反映「過去」的流動性，不會因成交量變化提前變動
    assert bool(mask["A"].iloc[100]) is True and bool(mask["B"].iloc[100]) is False
    # 互換已完全進入滾動視窗（第 131 日起，視窗全為互換後）→ B 佔優
    assert bool(mask["B"].iloc[131]) is True and bool(mask["A"].iloc[131]) is False
    assert bool(mask["B"].iloc[159]) is True and bool(mask["A"].iloc[159]) is False

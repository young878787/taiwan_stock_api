"""做空導向的假設性引導（RD-Agent ``fin_factor`` 假設生成注入）。

背景：RD-Agent 的 ``QlibFactorHypothesisGen.prepare_context`` 會把
``factor_hypothesis_specification`` prompt 組進假設生成的 system prompt。
本模組以子類在該 context 附加「做空為主」的研究引導，使 LLM 演化迴圈
專注尋找「因子值高 → 未來報酬偏低（負 IC）」的做空候選因子，並硬性排除：

1. **既有策略的反向**：``volume_change_5d`` 已被做多反轉策略採用
   （``strategy/volume_change_5d``，買量能萎縮）。提案「放空量能放大」
   （＝原策略取反向）只是同一訊號換邊，明文禁止。
2. **既有因子的翻版**：變號、rescale、z-score、rank 反轉等
   reparametrization 一律禁止，要求真正新的資訊或構造。

注入方式：RD-Agent 支援以環境變數 ``QLIB_FACTOR_HYPOTHESIS_GEN`` 覆寫
假設生成類別（``rdagent_runner.build_rdagent_env`` 在 ``--guidance short``
時自動注入 ``qlab.short_factor_proposal.GuidedShortFactorHypothesisGen``）。
引導本文以英文撰寫（LLM 遵從度），docstring／註解依專案慣例用繁體中文。
"""

from __future__ import annotations

from typing import Any

from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesisGen

# 既有因子清單（data/qlab/factor_report.md + 做空導向已產出者）：禁止以任何單調變換重提
EXISTING_FACTORS = (
    "daily_return",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "risk_adjusted_momentum_20d",
    "ma_deviation_20d",
    "obv_10d_change",
    "realized_vol_10d",
    "volatility_20d",
    "downside_deviation_20d",
    "volume_change_5d",
    # 2026-08-31 做空導向演化已產出且「實測無做空預測力」（tw100 宇宙、5 日前瞻 IC）：
    # upper_shadow_ratio / down_volume_share / Down_Volume_Dominance_Ratio ≈ 0；
    # gap_down_frequency 為顯著正 IC（t=3.76，反向）。
    "upper_shadow_ratio",
    "down_volume_share",
    "gap_down_frequency",
    "Down_Volume_Dominance_Ratio",
)

_SHORT_GUIDANCE = """
SHORT-SIDE RESEARCH GOAL (MANDATORY, overrides generic guidance below if they conflict):
- This run searches for SHORT-SELLING strategies. Every hypothesis must propose factors whose
  HIGH values predict LOWER future returns (expected negative Rank IC against forward returns),
  i.e., stocks worth shorting (sell first, buy back later). State the expected negative
  relationship and the economic rationale (sell pressure, distribution, exhaustion, crowding)
  explicitly in the hypothesis.
- We evaluate by shorting the TOP bucket (highest factor values). A factor with positive IC
  is NOT what we want here; "reversal" framings that simply buy losers are out of scope too.

HARD EXCLUSIONS (do NOT propose):
1. CRITICAL: the existing factor `volume_change_5d` (5-day volume change) is already exploited
   by a LONG strategy that buys LOW volume change. Proposing to short HIGH volume change - or
   ANY sign-flip, negation, rescaling, z-score, rank inversion, or trivial transform of it
   (e.g. -1*x, 1/x, (x-mean)/std, rolling-window variants that keep the same signal) - is
   FORBIDDEN: it merely reverses the existing strategy instead of finding new alpha.
2. Do NOT propose any transform of the existing factor library either:
   daily_return, momentum_5d, momentum_10d, momentum_20d, risk_adjusted_momentum_20d,
   ma_deviation_20d, obv_10d_change, realized_vol_10d, volatility_20d,
   downside_deviation_20d, volume_change_5d, upper_shadow_ratio, down_volume_share,
   gap_down_frequency, Down_Volume_Dominance_Ratio.
   The last four were already attempted as short-side factors in this exact universe and
   showed NO predictive power (IC≈0) or the OPPOSITE sign (gap_down_frequency: significant
   positive IC - high gap-down frequency predicts RISES here). Moving to different
   information sources or construction principles is required, not re-rolling these.
3. Do NOT re-propose generic short-term price reversal ("negative N-day return") - that is
   just the sign of the existing momentum factors.
4. A new factor must introduce genuinely NEW information or a NEW construction (new
   interaction of price path, volume pattern, range/gap structure, or duration/intensity
   conditioning), not a reparametrization of anything above.

FAVORED SHORT-SIDE DIRECTIONS (all computable from $open/$close/$high/$low/$volume/$factor):
- Distribution & sell pressure: share of volume on down days, ratio of high-volume down days,
  close location within the day range on down days, upper-shadow rejection after run-ups.
- Deterioration structure: consecutive lower-high/lower-close streaks, drawdown depth versus
  rebound strength asymmetry, counts of new N-day lows.
- Volume-price divergence on declines: falling price with expanding volume (failed support),
  volume-weighted downside momentum (must be clearly distinct from obv_10d_change and
  volume_change_5d constructions).
- Exhaustion / crowding: distance from recent high combined with fading volume, range
  expansion near local tops, gap-down behavior after extended advances.
Practical notes: use $close * $factor (dividend-adjusted) when computing returns; prefer
horizons of 5-20 trading days; guard against division by zero and suspended trading days.
""".strip()

# 使用者 prompt 的 RAG 提示：保留「先簡單後複雜」的節奏，再補做空方向提醒
_SHORT_RAG_SUFFIX = (
    " Remember: this run targets SHORT-selling factors (high value => lower future returns, "
    "negative IC). Never propose sign-flips or trivial transforms of the existing factors "
    "listed in the specification, especially volume_change_5d."
)


class GuidedShortFactorHypothesisGen(QlibFactorHypothesisGen):
    """在 ``prepare_context`` 附加做空引導的假設生成器。

    - ``hypothesis_specification``：附上做空目標與硬性排除清單（system prompt）。
    - ``RAG``：附加做空方向提醒（user prompt），保留原「先簡單後複雜」節奏。
    """

    def prepare_context(self, trace: Any) -> tuple[dict, bool]:
        context, ok = super().prepare_context(trace)
        if ok and isinstance(context, dict):
            base_spec = context.get("hypothesis_specification") or ""
            context["hypothesis_specification"] = f"{base_spec}\n\n{_SHORT_GUIDANCE}".strip()
            base_rag = context.get("RAG") or ""
            context["RAG"] = f"{base_rag}{_SHORT_RAG_SUFFIX}".strip()
        return context, ok

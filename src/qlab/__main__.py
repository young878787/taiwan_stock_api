"""qlab CLI：uv run python -m qlab <export|ic|rdagent|factor-report|backtest>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qlab",
        description="Qlib + RD-Agent 台股因子研究子專案（共用 kstock venv / .env / data/）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("export", help="把 data/normalized/daily 轉成 Qlib bin 格式資料集")

    p_ic = sub.add_parser("ic", help="Alpha158 + LightGBM 因子 IC 分析")
    p_ic.add_argument("--start", required=True, help="資料起點 YYYY-MM-DD")
    p_ic.add_argument("--end", required=True, help="資料終點 YYYY-MM-DD")
    p_ic.add_argument("--test-start", default=None, help="測試區間起點（預設最後 20%% 交易日）")

    p_rd = sub.add_parser("rdagent", help="執行 RD-Agent CLI（參數原樣轉傳，金鑰自動注入）")
    p_rd.add_argument("args", nargs="*", help='例如：fin_factor / quant / fin_factor_report')

    p_rdtest = sub.add_parser(
        "rdatest", help="以 instructor MD_JSON 對目前模型做結構化輸出實測（需真實金鑰）"
    )
    p_rdtest.add_argument("--model", default=None, help="覆寫模型（預設 OPENROUTER_MODEL / CHAT_MODEL）")
    p_rdtest.add_argument("--retries", type=int, default=4, help="instructor 驗證重試次數（預設 4）")

    p_rdt = sub.add_parser(
        "factor-report", help="匯出 fin_factor 因子結果報告（data/qlab/factor_report.md）"
    )
    p_rdt.add_argument("--output", default=None, help="報告輸出路徑（預設 <data>/qlab/factor_report.md）")
    p_rdt.add_argument("--fwd-days", type=int, default=5, help="IC 評估的前瞻報酬天數（預設 5）")

    p_h5 = sub.add_parser(
        "export-h5", help="台股日K → RD-Agent fin_factor 資料格式（daily_pv.h5，取代內建 A 股資料）"
    )
    p_h5.add_argument("--debug-symbols", type=int, default=20, help="debug 子集標的數（預設 20）")
    p_h5.add_argument("--no-adjust", action="store_true", help="不做復權（$factor=1.0，跳過 yfinance 抓取）")

    p_bt = sub.add_parser(
        "backtest", help="fin_factor 因子 → 台股橫截面投組回測（data/qlab/backtest_report.md）"
    )
    p_bt.add_argument(
        "--factors", default=None, help="逗號分隔的因子名稱（預設 ma_deviation_20d,volume_change_5d）"
    )
    p_bt.add_argument("--top-n", type=int, default=5, help="每次再平衡持有的標的數（預設 5）")
    p_bt.add_argument(
        "--rebalance-days", type=int, default=20, help="再平衡間隔（交易日，預設 20；愈短成本侵蝕愈重）"
    )
    p_bt.add_argument(
        "--direction",
        default="auto",
        choices=["auto", "top", "bottom"],
        help="買方方向：top=買因子大者、bottom=買小者、auto=以全樣本 IC 判斷（in-sample，預設）",
    )
    p_bt.add_argument("--commission", type=float, default=0.001425, help="單邊手續費率（預設 0.1425%%）")
    p_bt.add_argument("--tax", type=float, default=0.003, help="賣出證交稅率（預設 0.3%%）")
    p_bt.add_argument("--slippage", type=float, default=0.0, help="單邊滑價比例（預設 0）")
    p_bt.add_argument(
        "--gross", action="store_true", help="免成本模式（commission/tax/slippage 全 0），僅供訊號驗證"
    )
    p_bt.add_argument("--fwd-days", type=int, default=5, help="auto 方向判斷的 IC 前瞻天數（預設 5）")
    p_bt.add_argument("--output", default=None, help="報告輸出路徑（預設 <data>/qlab/backtest_report.md）")

    ns = parser.parse_args(argv)

    if ns.command == "export":
        from qlab.export import QlibDataExporter

        print(QlibDataExporter().export())
        return 0

    if ns.command == "ic":
        from qlab.factor_analysis import run_alpha158_ic

        print(run_alpha158_ic(start=ns.start, end=ns.end, test_start=ns.test_start))
        return 0

    if ns.command == "rdagent":
        from qlab.rdagent_runner import run_rdagent

        return run_rdagent(ns.args)

    if ns.command == "export-h5":
        from qlab.export_h5 import export_daily_pv

        full, debug = export_daily_pv(debug_symbols=ns.debug_symbols, adjust=not ns.no_adjust)
        print(f"正式版：{full.path}（{full.n_rows:,} 列 / {full.n_symbols} 標的 / {full.start}~{full.end} / 復權 {full.n_adjusted} 檔）")
        print(f"debug 版：{debug.path}（{debug.n_rows:,} 列 / {debug.n_symbols} 標的 / 復權 {debug.n_adjusted} 檔）")
        return 0

    if ns.command == "factor-report":
        from qlab.factor_report import export_report

        out = export_report(output=Path(ns.output) if ns.output else None, fwd_days=ns.fwd_days)
        print(f"報告已輸出：{out}")
        return 0

    if ns.command == "backtest":
        from qlab.factor_backtest import run_backtest_report

        factor_names = tuple(f.strip() for f in ns.factors.split(",")) if ns.factors else None
        if ns.gross:
            commission = tax = slippage = 0.0
        else:
            commission, tax, slippage = ns.commission, ns.tax, ns.slippage
        out = run_backtest_report(
            factor_names=factor_names,
            top_n=ns.top_n,
            rebalance_days=ns.rebalance_days,
            direction=ns.direction,
            commission=commission,
            tax=tax,
            slippage_rate=slippage,
            fwd_days=ns.fwd_days,
            output=Path(ns.output) if ns.output else None,
        )
        print(f"回測報告已輸出：{out}")
        return 0

    if ns.command == "rdatest":
        from qlab.rdatest import run_structured_test

        print(run_structured_test(model=ns.model, max_retries=ns.retries))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

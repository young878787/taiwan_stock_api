"""qlab CLI：uv run python -m qlab <export|ic|rdagent|factor-report|backtest>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_backtest_args(p: argparse.ArgumentParser) -> None:
    """backtest / exec-timing 共用的參數。"""
    p.add_argument(
        "--factors", default=None, help="逗號分隔的因子名稱（預設 ma_deviation_20d,volume_change_5d）"
    )
    p.add_argument("--top-n", type=int, default=5, help="每次再平衡持有的標的數（預設 5）")
    p.add_argument(
        "--rebalance-days", type=int, default=20, help="再平衡間隔（交易日，預設 20；愈短成本侵蝕愈重）"
    )
    p.add_argument(
        "--direction",
        default="auto",
        choices=["auto", "top", "bottom", "short"],
        help=(
            "買方方向：top=買因子大者、bottom=買小者、auto=以全樣本 IC 判斷（in-sample，預設）、"
            "short=放空因子大者（做空：top 訊號、報酬取負，淨值含成本；借券費不計；exec-timing 不支援）"
        ),
    )
    p.add_argument("--commission", type=float, default=0.001425, help="單邊手續費率（預設 0.1425%%）")
    p.add_argument("--tax", type=float, default=0.003, help="賣出證交稅率（預設 0.3%%）")
    p.add_argument("--slippage", type=float, default=0.0, help="單邊滑價比例（預設 0）")
    p.add_argument(
        "--buffer",
        type=int,
        default=None,
        metavar="N",
        help="緩衝帶：持倉跌出前 N 名才換股（N > top-n），降低換倉與成本（例：--buffer 10）",
    )
    p.add_argument("--fwd-days", type=int, default=5, help="auto 方向判斷的 IC 前瞻天數（預設 5）")
    p.add_argument("--output", default=None, help="報告輸出路徑（預設 <data>/qlab/ 下對應報告檔）")
    p.add_argument(
        "--data", default=None, help="價格 h5 路徑（預設 factor_report 候選：debug 20 檔優先）"
    )
    p.add_argument(
        "--top-symbols",
        type=int,
        default=None,
        help="只取前 N 檔（依代碼排序）當宇宙，因子值會在該子集上重跑（例：--top-symbols 100）",
    )


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
    p_rd.add_argument(
        "--universe",
        default="full",
        choices=["full", "tw100", "debug"],
        help="因子演化資料宇宙：full=台股全市場（預設）、tw100=代碼前 100 檔（與回測口徑一致）、debug=20 檔",
    )
    p_rd.add_argument(
        "--guidance",
        default="none",
        choices=["none", "short"],
        help="假設性引導：short=做空導向（高值→未來跌；排除 volume_change_5d 反向與既有因子翻版）",
    )
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
    p_h5.add_argument("--no-adjust", action="store_true", help="不做復權（$factor=1.0，跳過抓取）")
    p_h5.add_argument(
        "--adjust-source",
        default="finmind",
        choices=["finmind", "yfinance"],
        help="復權因子來源：finmind=除權息事件自算（預設、事件覆蓋完整）；yfinance=Adj Close/Close",
    )

    p_tw100 = sub.add_parser(
        "export-tw100",
        help="從既有正式版 daily_pv.h5 切出前 N 檔（依 instrument 排序）→ factor_source_data_twN/（不重抓、零誤差）",
    )
    p_tw100.add_argument("--top", type=int, default=100, help="宇宙檔數（預設 100）")
    p_tw100.add_argument(
        "--order",
        default="code",
        choices=["code", "turnover"],
        help="排序口徑：code=代碼序（預設、正式口徑）；turnover=成交金額排名（對照實驗用）",
    )
    p_tw100.add_argument("--src", default=None, help="來源 h5（預設 factor_source_data_tw/daily_pv.h5）")

    p_bt = sub.add_parser(
        "backtest", help="fin_factor 因子 → 台股橫截面投組回測（data/qlab/backtest_report.md）"
    )
    _add_backtest_args(p_bt)
    p_bt.add_argument(
        "--gross", action="store_true", help="免成本模式（commission/tax/slippage 全 0），僅供訊號驗證"
    )

    p_vd = sub.add_parser(
        "verify-data", help="驗證 daily_pv.h5 與 kstock 台股 parquet 資料庫一致（來源/排列/數值）"
    )
    p_vd.add_argument("--h5", default=None, help="h5 路徑（預設 factor_source_data_tw/daily_pv.h5）")
    p_bt.add_argument(
        "--oos",
        type=float,
        default=None,
        metavar="RATIO",
        help="樣本外驗證：前 RATIO 期間定方向（如 0.7），其餘為 OOS 回測（輸出 oos_backtest_report.md）",
    )
    p_bt.add_argument(
        "--trades",
        action="store_true",
        help="額外輸出實際進出倉明細 CSV（trades_<因子>.csv，與報告同目錄；OOS 模式為 oos_trades_<因子>.csv）",
    )

    p_et = sub.add_parser(
        "exec-timing",
        help="執行時點對照：同一訊號以 t收盤/t+1開盤/t+1收盤 成交回測（診斷 alpha 對隔夜段的依賴）",
    )
    _add_backtest_args(p_et)

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

        return run_rdagent(ns.args, universe=ns.universe, guidance=ns.guidance)

    if ns.command == "export-tw100":
        from qlab.export_h5 import slice_top_symbols

        rep = slice_top_symbols(n=ns.top, src=Path(ns.src) if ns.src else None, order=ns.order)
        print(f"已切出：{rep.path}（{rep.n_rows:,} 列 / {rep.n_symbols} 標的 / {rep.start}~{rep.end}）")
        return 0

    if ns.command == "export-h5":
        from qlab.export_h5 import export_daily_pv

        full, debug = export_daily_pv(
            debug_symbols=ns.debug_symbols, adjust=not ns.no_adjust, adjust_source=ns.adjust_source
        )
        print(f"正式版：{full.path}（{full.n_rows:,} 列 / {full.n_symbols} 標的 / {full.start}~{full.end} / 復權 {full.n_adjusted} 檔）")
        print(f"debug 版：{debug.path}（{debug.n_rows:,} 列 / {debug.n_symbols} 標的 / 復權 {debug.n_adjusted} 檔）")
        return 0

    if ns.command == "factor-report":
        from qlab.factor_report import export_report

        out = export_report(output=Path(ns.output) if ns.output else None, fwd_days=ns.fwd_days)
        print(f"報告已輸出：{out}")
        return 0

    if ns.command == "backtest":
        from qlab.factor_backtest import run_backtest_report, run_oos_report

        factor_names = tuple(f.strip() for f in ns.factors.split(",")) if ns.factors else None
        common = dict(
            factor_names=factor_names,
            top_n=ns.top_n,
            rebalance_days=ns.rebalance_days,
            buffer_n=ns.buffer,
            commission=0.0 if ns.gross else ns.commission,
            tax=0.0 if ns.gross else ns.tax,
            slippage_rate=0.0 if ns.gross else ns.slippage,
            data=Path(ns.data) if ns.data else None,
            top_symbols=ns.top_symbols,
            trades=ns.trades,
        )
        if ns.oos is not None:
            if not 0.1 < ns.oos < 0.9:
                parser.error("--oos RATIO 需在 0.1~0.9 之間")
            out = run_oos_report(
                fwd_days=ns.fwd_days,
                is_ratio=ns.oos,
                output=Path(ns.output) if ns.output else None,
                **common,
            )
        else:
            out = run_backtest_report(
                fwd_days=ns.fwd_days,
                direction=ns.direction,
                output=Path(ns.output) if ns.output else None,
                **common,
            )
        print(f"回測報告已輸出：{out}")
        return 0

    if ns.command == "exec-timing":
        from qlab.factor_backtest import run_execution_timing_report

        factor_names = tuple(f.strip() for f in ns.factors.split(",")) if ns.factors else None
        out = run_execution_timing_report(
            factor_names=factor_names,
            top_n=ns.top_n,
            rebalance_days=ns.rebalance_days,
            direction=ns.direction,
            buffer_n=ns.buffer,
            commission=ns.commission,
            tax=ns.tax,
            slippage_rate=ns.slippage,
            fwd_days=ns.fwd_days,
            output=Path(ns.output) if ns.output else None,
            data=Path(ns.data) if ns.data else None,
            top_symbols=ns.top_symbols,
        )
        print(f"執行時點對照報告已輸出：{out}")
        return 0

    if ns.command == "verify-data":
        from qlab.data_check import verify_daily_pv

        print(verify_daily_pv(Path(ns.h5) if ns.h5 else None))
        return 0

    if ns.command == "rdatest":
        from qlab.rdatest import run_structured_test

        print(run_structured_test(model=ns.model, max_retries=ns.retries))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

"""qlab CLI：uv run python -m qlab <export|ic|rdagent>"""

from __future__ import annotations

import argparse
import sys


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

    if ns.command == "factor-report":
        from pathlib import Path

        from qlab.factor_report import export_report

        out = export_report(output=Path(ns.output) if ns.output else None)
        print(f"報告已輸出：{out}")
        return 0

    if ns.command == "rdatest":
        from qlab.rdatest import run_structured_test

        print(run_structured_test(model=ns.model, max_retries=ns.retries))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

"""instrument 參考表入庫：FinMind TaiwanStockInfo → normalized/instrument。

單一請求、低頻參考表（股票清單異動頻率低），故為獨立 CLI、不掛日更 pipeline：

    uv run python -m kstock.instruments

行為：
- 抓全量現活躍清單（2026 改版後 TaiwanStockInfo 僅 5 欄，欄位映射見
  FinMindAdapter.get_instruments：無 delist_date、date≠list_date）。
- 以 write_reference_table 單檔 upsert（冪等鍵 symbol，keep="last"），重跑即更新。
- 輸出列數與庫內 market 分布（TSE/OTC/EMG）。
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from kstock.adapters.finmind import FinMindAdapter
from kstock.storage.parquet import ParquetStore


def run_instruments(
    adapter: FinMindAdapter | None = None, store: ParquetStore | None = None
) -> int:
    """抓全量 instrument 清單並單檔 upsert 入庫；回傳本次抓取列數（空抓回 0）。"""
    adapter = adapter or FinMindAdapter()
    store = store or ParquetStore()
    df = adapter.get_instruments()
    if df.height == 0:
        return 0
    store.write_reference_table("instrument", df, subset=["symbol"])
    return df.height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kstock.instruments",
        description="TaiwanStockInfo → instrument 參考表入庫（單檔 upsert，冪等鍵 symbol）",
    )
    parser.parse_args(argv)
    store = ParquetStore()
    try:
        n = run_instruments(store=store)
    except Exception as e:  # 遮罩 token，避免 FinMind URL 外洩到 log
        text = str(e)
        if "token=" in text:
            text = text.split("token=")[0].rstrip("&?") + " token=***"
        print(f"instrument 入庫失敗：{text}", file=sys.stderr)
        return 1
    print(f"instrument 入庫完成：本次抓取 {n} 列")
    if n == 0:
        print(
            "警告：抓到 0 列（dataset 可能改版，用 field_names(\"TaiwanStockInfo\") 對照欄位）",
            file=sys.stderr,
        )
        return 0
    target = store.normalized_dir / "instrument" / "data.parquet"
    df = pl.read_parquet(target)
    print(f"庫內合計 {df.height} 檔，market 分布：")
    dist = df.group_by("market").len().sort("market")
    for market, count in dist.iter_rows():
        print(f"  {market}: {count} 檔")
    return 0


if __name__ == "__main__":
    sys.exit(main())

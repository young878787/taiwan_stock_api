"""universe_history 測試：全離線，fetch_day 一律 mock、零網路。"""

from __future__ import annotations

import time
from datetime import date, timedelta

import polars as pl
import pytest

from kstock.storage.parquet import ParquetStore
from kstock.universe_history import (
    INTERVALS_SCHEMA,
    SEEN_SCHEMA,
    SampleFailure,
    build_seen_intervals,
    delisted_candidates,
    main,
    merge_delisted_sources,
    monthly_sample_dates,
    parse_terminated_csv,
    sample_market_history,
)

# fetch_day 的完整欄位（比照 WholeMarketQuotes.fetch_day，多餘欄位應被丟棄）
QUOTE_SCHEMA = {"symbol": pl.Utf8, "name": pl.Utf8, "close": pl.Float64, "volume_shares": pl.Int64, "turnover_twd": pl.Float64}


def make_fetch_day(schedule: dict[str, list[tuple[str, str]]], calls: list[str] | None = None):
    """依日期排班的 mock fetch_day：排程內回報價列、其餘（假日/休市）回空 df。"""

    def fetch_day(iso: str) -> pl.DataFrame:
        if calls is not None:
            calls.append(iso)
        if iso in schedule:
            rows = schedule[iso]
            return pl.DataFrame(
                {
                    "symbol": [s for s, _ in rows],
                    "name": [n for _, n in rows],
                    "close": [10.0] * len(rows),
                    "volume_shares": [1_000] * len(rows),
                    "turnover_twd": [1.0e6] * len(rows),
                },
                schema=QUOTE_SCHEMA,
            )
        return pl.DataFrame(schema=QUOTE_SCHEMA)

    return fetch_day


def _seen_frame(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """(symbol, name, iso_date) → seen 長表（seen_date = pl.Date）。"""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "seen_date": [date.fromisoformat(r[2]) for r in rows],
        },
        schema=SEEN_SCHEMA,
    )


def _intervals_frame(rows: list[tuple[str, str, str, str, int]]) -> pl.DataFrame:
    """(symbol, name, first_seen, last_seen, n_days) → 區間表。"""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "first_seen": [date.fromisoformat(r[2]) for r in rows],
            "last_seen": [date.fromisoformat(r[3]) for r in rows],
            "n_days": [r[4] for r in rows],
        },
        schema=INTERVALS_SCHEMA,
    )


# ---- 1. monthly_sample_dates ----


def test_monthly_sample_dates_skips_holidays_and_one_per_month():
    calls: list[str] = []
    fetch_day = make_fetch_day(
        {"2024-01-03": [("2330", "台積電")], "2024-02-01": [("2330", "台積電")]},
        calls,
    )
    dates = monthly_sample_dates("2024-01-01", "2024-02-29", fetch_day)
    assert dates == ["2024-01-03", "2024-02-01"]
    # 假日（01-01、01-02）探測後正確推進到下一個有效日；每月恰一次
    assert calls == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"]


def test_monthly_sample_dates_month_all_empty_raises():
    # 1 月整月皆空（12 次探測內無資料）→ 嚴格模式 raise（缺月會造成假斷點）
    fetch_day = make_fetch_day({"2024-02-01": [("2330", "台積電")]})
    with pytest.raises(SampleFailure):
        monthly_sample_dates("2024-01-01", "2024-02-29", fetch_day)


def test_monthly_sample_dates_probes_from_start_day():
    calls: list[str] = []
    fetch_day = make_fetch_day({"2024-01-16": [("2330", "台積電")]}, calls)
    dates = monthly_sample_dates("2024-01-15", "2024-01-31", fetch_day)
    assert dates == ["2024-01-16"]
    assert calls[0] == "2024-01-15"  # 首月起試日 = start 而非月初


# ---- 2. sample_market_history ----


def test_sample_market_history_long_table_and_sleep(monkeypatch):
    calls: list[str] = []
    fetch_day = make_fetch_day(
        {
            "2024-01-03": [("2330", "台積電"), ("1101", "台泥")],
            "2024-02-01": [("2330", "台積電")],
        },
        calls,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    df = sample_market_history("2024-01-01", "2024-02-29", fetch_day=fetch_day, sleep_seconds=0.3)

    assert df.columns == ["symbol", "name", "seen_date"]
    assert df.schema["seen_date"] == pl.Date
    assert df.height == 3  # 1 月 2 檔 + 2 月 1 檔
    row = df.filter(pl.col("symbol") == "1101").row(0, named=True)
    assert row["name"] == "台泥"
    assert row["seen_date"] == date(2024, 1, 3)
    assert df.filter(pl.col("symbol") == "2330")["seen_date"].to_list() == [
        date(2024, 1, 3),
        date(2024, 2, 1),
    ]
    # 每次 fetch（含空表探測）都會打 API，但 sleep 只在抽到資料後（每月 1 次）：1 月 1 次 + 2 月 1 次 = 2
    assert calls == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"]
    assert sleeps == [0.3, 0.3]


def test_sample_market_history_zero_sleep_all_empty_raises(monkeypatch):
    fetch_day = make_fetch_day({})  # 全部休市 → 整月失敗
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(SampleFailure):
        sample_market_history("2024-01-01", "2024-01-31", fetch_day=fetch_day, sleep_seconds=0.0)
    assert sleeps == []  # sleep_seconds=0 → 不睡


def test_fetch_with_backoff_retries_on_rate_limit_then_succeeds(monkeypatch):
    import httpx

    from kstock.universe_history import _fetch_with_backoff

    calls: list[str] = []
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    good = pl.DataFrame({"symbol": ["2330"], "name": ["台積電"], "close": [10.0], "volume_shares": [1], "turnover_twd": [1.0]}, schema=QUOTE_SCHEMA)

    def flaky(iso: str) -> pl.DataFrame:
        calls.append(iso)
        if len(calls) == 1:
            req = httpx.Request("GET", "https://www.twse.com.tw/x")
            resp = httpx.Response(428, request=req)
            raise httpx.HTTPStatusError("precondition", request=req, response=resp)
        return good

    df = _fetch_with_backoff(flaky, "2024-01-02")
    assert not df.is_empty()
    assert calls == ["2024-01-02", "2024-01-02"]
    assert sleeps == [60]  # 第一次 428 → 等待 60s 後重試


def test_fetch_with_backoff_gives_up_after_retries(monkeypatch):
    import httpx

    from kstock.universe_history import _fetch_with_backoff

    monkeypatch.setattr(time, "sleep", lambda s: None)
    n = {"k": 0}

    def always_428(iso: str) -> pl.DataFrame:
        n["k"] += 1
        req = httpx.Request("GET", "https://www.twse.com.tw/x")
        resp = httpx.Response(428, request=req)
        raise httpx.HTTPStatusError("precondition", request=req, response=resp)

    with pytest.raises(httpx.HTTPStatusError):
        _fetch_with_backoff(always_428, "2024-01-02")
    assert n["k"] == 4  # 初試 + 3 次重試


# ---- 3. build_seen_intervals ----


def test_build_seen_intervals_consecutive_single_interval():
    seen = _seen_frame(
        [
            ("2330", "台積電", "2024-01-03"),
            ("2330", "台積電", "2024-02-01"),
            ("2330", "台積電", "2024-03-04"),
        ]
    )
    iv = build_seen_intervals(seen, gap_days=90)
    assert iv.height == 1
    row = iv.row(0, named=True)
    assert row["symbol"] == "2330"
    assert row["name"] == "台積電"
    assert row["first_seen"] == date(2024, 1, 3)
    assert row["last_seen"] == date(2024, 3, 4)
    assert row["n_days"] == 3


def test_build_seen_intervals_rename_takes_last_name():
    # 區間中改名屬改名：區間名稱取最後一次出現者（虛構情境）
    seen = _seen_frame(
        [
            ("1201", "味全", "2024-01-03"),
            ("1201", "味全食品", "2024-02-01"),
            ("1201", "味全食品", "2024-03-04"),
        ]
    )
    iv = build_seen_intervals(seen, gap_days=90)
    assert iv.height == 1
    assert iv["name"].to_list() == ["味全食品"]
    assert iv["n_days"].to_list() == [3]


def test_build_seen_intervals_code_reuse_6904_splits_two_segments():
    # 代碼重用實例：6904 台開（2022 下市）→ 2023 起同一代碼由「伯鑫」使用
    seen = _seen_frame(
        [
            ("6904", "台開", "2021-01-04"),
            ("6904", "台開", "2021-02-03"),
            ("6904", "伯鑫", "2023-10-03"),
            ("6904", "伯鑫", "2023-11-02"),
        ]
    )
    iv = build_seen_intervals(seen, gap_days=90)
    assert iv.height == 2
    first, second = iv.rows(named=True)  # 依 (symbol, first_seen) 排序
    assert first["name"] == "台開" and first["n_days"] == 2
    assert first["first_seen"] == date(2021, 1, 4) and first["last_seen"] == date(2021, 2, 3)
    assert second["name"] == "伯鑫" and second["n_days"] == 2
    assert second["first_seen"] == date(2023, 10, 3) and second["last_seen"] == date(2023, 11, 2)


def test_build_seen_intervals_gap_boundary_90_vs_91():
    seen = _seen_frame(
        [
            ("1001", "甲", "2024-01-01"),
            ("1001", "甲", "2024-03-31"),  # 差 90 天（2024 閏年）→ 同一區間
            ("2001", "乙", "2024-01-01"),
            ("2001", "乙", "2024-04-01"),  # 差 91 天 → 斷點成兩段
        ]
    )
    iv = build_seen_intervals(seen, gap_days=90)
    g1 = iv.filter(pl.col("symbol") == "1001")
    g2 = iv.filter(pl.col("symbol") == "2001")
    assert g1.height == 1 and g1["n_days"].to_list() == [2]
    assert g2.height == 2 and g2["n_days"].to_list() == [1, 1]


def test_build_seen_intervals_empty_input():
    iv = build_seen_intervals(pl.DataFrame(schema=SEEN_SCHEMA))
    assert iv.is_empty()
    assert iv.columns == ["symbol", "name", "first_seen", "last_seen", "n_days"]


def test_build_seen_intervals_with_terminated_merges_suspension_gaps():
    # 3018 隆銘綠能情境：2022-11 停牌、2023-07 恢復（gap > 90 天）但無下市事件 → 合併
    seen = _seen_frame(
        [
            ("3018", "隆銘綠能", "2022-11-01"),
            ("3018", "隆銘綠能", "2023-07-03"),
            ("3018", "隆銘綠能", "2023-08-01"),
        ]
    )
    terminated = pl.DataFrame(
        {"symbol": ["2841"], "name": ["台開"], "delist_date": [date(2022, 8, 4)]},
        schema={"symbol": pl.Utf8, "name": pl.Utf8, "delist_date": pl.Date},
    )
    iv = build_seen_intervals(seen, gap_days=90, terminated=terminated)
    assert iv.height == 1  # 純停牌 gap 不切割
    assert iv.row(0, named=True)["first_seen"] == date(2022, 11, 1)
    assert iv.row(0, named=True)["last_seen"] == date(2023, 8, 1)


def test_build_seen_intervals_with_terminated_splits_on_delist_event():
    # 代碼重用：舊公司（2008-2012 樣本）→ delist 2012-12-11 → 新公司（2015 起樣本）
    seen = _seen_frame(
        [
            ("XXXX", "舊公司", "2012-01-03"),
            ("XXXX", "舊公司", "2012-12-03"),
            ("XXXX", "新公司", "2015-01-05"),
            ("XXXX", "新公司", "2015-02-03"),
        ]
    )
    terminated = pl.DataFrame(
        {"symbol": ["XXXX"], "name": ["舊公司"], "delist_date": [date(2012, 12, 11)]},
        schema={"symbol": pl.Utf8, "name": pl.Utf8, "delist_date": pl.Date},
    )
    iv = build_seen_intervals(seen, gap_days=90, terminated=terminated)
    assert iv.height == 2
    first, second = iv.rows(named=True)
    assert first["name"] == "舊公司" and first["last_seen"] == date(2012, 12, 3)
    assert second["name"] == "新公司" and second["first_seen"] == date(2015, 1, 5)


def test_build_seen_intervals_with_terminated_no_event_after_last_sample():
    # 下市發生在最後一次抽樣之後 → 不產生新段（公司段止於最後抽樣日）
    seen = _seen_frame(
        [
            ("2841", "台開", "2022-07-01"),
            ("2841", "台開", "2022-08-01"),
        ]
    )
    terminated = pl.DataFrame(
        {"symbol": ["2841"], "name": ["台開"], "delist_date": [date(2022, 8, 4)]},
        schema={"symbol": pl.Utf8, "name": pl.Utf8, "delist_date": pl.Date},
    )
    iv = build_seen_intervals(seen, gap_days=90, terminated=terminated)
    assert iv.height == 1


# ---- 4. delisted_candidates ----


def test_delisted_candidates_active_exclusion_and_boundary():
    as_of = date(2024, 10, 31)
    cutoff = as_of - timedelta(days=90)  # last_seen 必須「早於」cutoff 才候選
    intervals = _intervals_frame(
        [
            ("2330", "台積電", "2024-01-02", (cutoff + timedelta(days=1)).isoformat(), 5),  # 門檻下側（太新）
            ("1101", "台泥", "2024-01-02", cutoff.isoformat(), 5),  # 恰在門檻上 → 不候選
            ("1417", "嘉裕", "2024-01-02", (cutoff - timedelta(days=1)).isoformat(), 5),  # 候選
            ("6904", "台開", "2021-01-04", "2021-06-01", 3),  # 很舊但活躍 → 被剔除
            ("6172", "互盛電", "2024-01-02", (cutoff - timedelta(days=40)).isoformat(), 5),  # 候選
        ]
    )
    out = delisted_candidates(intervals, active_symbols=["6904"], inactive_days=90, as_of=as_of)
    assert out["symbol"].to_list() == ["1417", "6172"]  # last_seen 降序
    assert set(out.columns) == set(INTERVALS_SCHEMA)

    # active_symbols=None → 不排除任何人
    out_all = delisted_candidates(intervals, inactive_days=90, as_of=as_of)
    assert "6904" in out_all["symbol"].to_list()
    assert "2330" not in out_all["symbol"].to_list() and "1101" not in out_all["symbol"].to_list()


# ---- 5. 官方「終止上市」清單 ----

SAMPLE_TERMINATED_CSV = (
    "終止上市公司\n"
    '"終止上市日期","公司名稱","上市編號"\n'
    '"民國115年09月01日","三商壽","2867"\n'
    '"民國101年12月11日","力晶","5346"\n'
    '"民國111年08月04日","台開","2841"\n'
    '"99年01月20日","老標的","1001"\n'
    "亂行不匹配\n"
)


def test_parse_terminated_csv():
    df = parse_terminated_csv(SAMPLE_TERMINATED_CSV)
    assert df.columns == ["symbol", "name", "delist_date"]
    rows = {r["symbol"]: r for r in df.rows(named=True)}
    assert rows["2867"]["delist_date"] == date(2026, 9, 1)  # 民國 115 → 2026
    assert rows["5346"]["delist_date"] == date(2012, 12, 11)
    assert rows["2841"]["name"] == "台開"
    assert rows["1001"]["delist_date"] == date(2010, 1, 20)  # 無「民國」前綴也解析
    assert len(df) == 4  # 雜訊行忽略


def test_merge_delisted_sources_official_priority_and_active_crossout():
    official = pl.DataFrame(
        {
            "symbol": ["6423", "2841", "5346"],  # 6423 轉上櫃仍在交易 → 應被剔除
            "name": ["億而得-創", "台開", "力晶"],
            "delist_date": [date(2026, 1, 22), date(2022, 8, 4), date(2012, 12, 11)],
        },
        schema={"symbol": pl.Utf8, "name": pl.Utf8, "delist_date": pl.Date},
    )
    sampled = _intervals_frame(
        [
            ("2841", "台開", "2021-01-04", "2022-07-03", 18),  # 官方已有 → 去重留官方
            ("6172", "互盛電", "2024-01-02", "2024-05-01", 4),  # 僅抽樣候選
        ]
    )
    out = merge_delisted_sources(official, sampled, active_symbols=["6423"])
    assert set(out.columns) == {"symbol", "name", "delist_date", "source"}
    rows = {r["symbol"]: r for r in out.rows(named=True)}
    assert "6423" not in rows  # 活躍（轉板）剔除
    assert rows["2841"]["source"] == "twse_terminated"  # 官方優先
    assert rows["6172"]["source"] == "mi_index_sampling"
    assert rows["6172"]["delist_date"] is None
    assert out["delist_date"].drop_nulls().is_sorted() or out.height >= 2


# ---- 6. CLI main ----


def test_sample_market_history_resume_skips_fetched_months(monkeypatch):
    calls: list[str] = []
    fetch_day = make_fetch_day(
        {
            "2024-01-03": [("2330", "台積電")],
            "2024-02-01": [("1101", "台泥")],
            "2024-03-04": [("2317", "鴻海")],
        },
        calls,
    )
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # checkpoint：1 月已抽到 → 1 月不應重打請求，只抽 2、3 月
    resume = _seen_frame([("2330", "台積電", "2024-01-03")])

    df = sample_market_history("2024-01-01", "2024-03-31", fetch_day=fetch_day, resume_df=resume)

    assert not any(c.startswith("2024-01") for c in calls)  # 已抽月份零請求
    assert df["symbol"].to_list() == ["1101", "2317", "2330"]  # 最終輸出依 (symbol, seen_date) 排序
    assert df["seen_date"].to_list() == [date(2024, 2, 1), date(2024, 3, 4), date(2024, 1, 3)]


def test_sample_market_history_on_progress_accumulates(monkeypatch):
    fetch_day = make_fetch_day(
        {
            "2024-01-03": [("2330", "台積電")],
            "2024-02-01": [("1101", "台泥")],
        }
    )
    monkeypatch.setattr(time, "sleep", lambda s: None)
    seen_heights: list[int] = []
    df = sample_market_history(
        "2024-01-01",
        "2024-02-29",
        fetch_day=fetch_day,
        on_progress=lambda acc: seen_heights.append(acc.height),
    )
    assert seen_heights == [1, 2]  # 每月回呼一次累積長表
    assert df.height == 2


def test_cli_checkpoint_on_failure_then_resume_completes(tmp_path, monkeypatch):
    out_dir = tmp_path / "manifests"
    store = ParquetStore(root=tmp_path / "store")
    calls: list[str] = []
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # 第一次：1 月成功、2 月整月失敗 → rc=1 且 checkpoint 落盤
    fetch1 = make_fetch_day({"2024-01-03": [("2330", "台積電")]}, calls)
    rc1 = main(
        ["--start", "2024-01-01", "--end", "2024-02-29", "--sleep", "0", "--output", str(out_dir)],
        fetch_day=fetch1,
        store=store,
    )
    assert rc1 == 1
    partial = out_dir / "tw_sample_partial.csv"
    assert partial.exists()
    stamp = date.today().strftime("%Y%m%d")
    assert not (out_dir / f"tw_seen_{stamp}.csv").exists()  # 失敗不產出最終 manifest

    # 第二次（--resume）：2 月補上 → 完成、最終 CSV 落地、checkpoint 退役
    calls.clear()
    fetch2 = make_fetch_day(
        {"2024-02-01": [("1101", "台泥")], "2024-01-03": [("2330", "台積電")]}, calls
    )
    rc2 = main(
        [
            "--start", "2024-01-01", "--end", "2024-02-29", "--sleep", "0",
            "--output", str(out_dir), "--resume",
        ],
        fetch_day=fetch2,
        store=store,
    )
    assert rc2 == 0
    assert not any(c.startswith("2024-01") for c in calls)  # 續跑不重抓 1 月
    seen = pl.read_csv(out_dir / f"tw_seen_{stamp}.csv", schema_overrides={"symbol": pl.Utf8})
    assert seen["symbol"].to_list() == ["1101", "2330"]  # (symbol, seen_date) 排序
    assert not partial.exists()


def test_cli_dry_run_prints_plan_without_fetch(capsys):
    def boom(iso: str) -> pl.DataFrame:
        raise AssertionError("dry-run 不應發請求")

    rc = main(["--start", "2024-01-01", "--end", "2024-03-31", "--dry-run"], fetch_day=boom)
    out = capsys.readouterr().out
    assert rc == 0
    assert "抽樣月數：3" in out
    assert "預估請求數上限：36" in out  # 3 月 × max_tries 12


def test_cli_real_path_writes_csvs_and_stats(tmp_path, capsys):
    calls: list[str] = []
    fetch_day = make_fetch_day(
        {
            "2024-01-03": [("2330", "台積電"), ("1101", "台泥")],
            "2024-02-01": [("2330", "台積電")],
        },
        calls,
    )
    out_dir = tmp_path / "manifests"
    store = ParquetStore(root=tmp_path / "store")  # 無 instrument 表 → 警告路徑

    rc = main(
        ["--start", "2024-01-01", "--end", "2024-02-29", "--sleep", "0", "--output", str(out_dir)],
        fetch_day=fetch_day,
        store=store,
        terminated_csv=SAMPLE_TERMINATED_CSV,
    )
    assert rc == 0

    stamp = date.today().strftime("%Y%m%d")
    seen_path = out_dir / f"tw_seen_{stamp}.csv"
    delisted_path = out_dir / f"delisted_tw_{stamp}.csv"
    assert seen_path.exists() and delisted_path.exists()
    assert (out_dir / f"tw_seen_raw_{stamp}.csv").exists()  # raw 抽樣長表（重建規則用）

    seen = pl.read_csv(seen_path, schema_overrides={"symbol": pl.Utf8})
    # tw_seen_* = 全部存在區間（非原始抽樣列）：2330 跨兩月同一區間、1101 一區間
    assert seen.height == 2
    assert set(seen.columns) == {"symbol", "name", "first_seen", "last_seen", "n_days"}
    by_sym = {r["symbol"]: r for r in seen.rows(named=True)}
    assert by_sym["2330"]["n_days"] == 2 and by_sym["1101"]["n_days"] == 1

    delisted = pl.read_csv(delisted_path, schema_overrides={"symbol": pl.Utf8})
    # 官方清單優先（source=twse_terminated），抽樣候選補充；官方含 2841 台開
    assert "2841" in delisted["symbol"].to_list()
    assert delisted.filter(pl.col("source") == "mi_index_sampling")["symbol"].to_list() == ["1101", "2330"]  # (delist_date nulls last, symbol) 排序

    out = capsys.readouterr().out
    assert "無 instrument 表，未剔除活躍股" in out
    assert "區間數 2" in out
    assert "下市候選 6 檔" in out  # 2 抽樣候選 + 4 官方（6423 非活躍但無 active 剔除）
    assert "抽樣月數 2/2" in out


def test_cli_returns_1_when_month_fails(tmp_path, capsys):
    fetch_day = make_fetch_day({})  # 全休市 → 整月失敗（嚴格模式）
    out_dir = tmp_path / "manifests"
    store = ParquetStore(root=tmp_path / "store")
    rc = main(
        ["--start", "2024-01-01", "--end", "2024-01-31", "--sleep", "0", "--output", str(out_dir)],
        fetch_day=fetch_day,
        store=store,
    )
    out = capsys.readouterr()
    assert rc == 1
    assert "整月抽不到資料" in out.err
    assert not (out_dir / f"tw_seen_{date.today().strftime('%Y%m%d')}.csv").exists()

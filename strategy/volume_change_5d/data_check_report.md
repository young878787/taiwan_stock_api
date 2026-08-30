# daily_pv.h5 資料來源與品質驗證

- h5：`/home/owo/taiwan_stock_api/data/qlab/factor_source_data_tw/daily_pv.h5`（400,923 列）
- parquet 資料庫：`/home/owo/taiwan_stock_api/data/normalized/daily`（kstock 台股日K，入庫來源 FinMind/TWSE/yfinance）

| # | 檢查項 | 結果 |
|---|---|---|
| 1 | MultiIndex 單調遞增（datetime→instrument 排列） | ✅  |
| 2 | 無重複 (datetime, instrument) | ✅ 重複 0 組 |
| 3 | 無重複 (instrument, datetime) | ✅ 重複 0 組 |
| 4 | 日期範圍 | ✅ 2021-01-04 ~ 2026-08-26 |
| 5 | h5 標的 ⊆ parquet 資料庫（來源一致） | ✅ h5 300 檔；資料庫 300 檔；缺失 0 |
| 6 | h5 交易日 ⊆ parquet 交易日 | ✅ h5 1370 個交易日；多出 0 日 |
| 7 | 每列都能對應到 parquet 一列（不缺列） | ✅ 未對應 0 列 |
| 8 | $close ≡ parquet close（全量 400,923 列） | ✅ 不一致 0 列（最大差 0） |
| 9 | $volume ≡ parquet volume_shares（股） | ✅ 不一致 0 列（最大差 0） |
| 10 | close 無 0/負值 | ⚠️ ≤0 共 112 列（來源 parquet 同位置亦為 0 → 上游停牌資料；回測端視為缺價防禦） |
| 11 | close 無 NaN/inf | ✅ NaN/inf 共 0 列 |
| 12 | 無 factor 缺日洞（孤立 1.0） | ✅ 0 列（前後日因子都 <0.9 卻單日 =1.0）——yfinance 缺日 fallback 所致，會讓復權價單日假跳 |
| 13 | 復權價日跳動 >60%（yfinance 除息事件覆蓋限制） | ⚠️ 214 次 / 35 檔（佔 0.053%） |
| 14 | $factor 範圍合理 | ✅ min=0.3433, max=1（1.0=無除權息；跳變=復權調整） |

**結論：通過，2 項已知限制**（14 項檢查；⚠️ = 已知限制、不影響回測防禦）


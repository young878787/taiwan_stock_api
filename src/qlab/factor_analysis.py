"""Qlib 因子分析初步入口：Alpha158 + LightGBM，輸出 IC（Rank IC）評估。

使用前先執行 `uv run python -m qlab export` 把 kstock daily 轉成 Qlib bin 格式。
台股資料自帶交易日曆（由資料本身產生），因此以 REG_CN 設定即可運作。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.utils import init_instance_by_config

from qlab.config import QlabSettings, qlab_settings


def init_qlib(provider_dir: Path | str | None = None) -> None:
    """初始化 Qlib（指向 qlab 匯出的 bin 資料集）。"""
    s = qlab_settings()
    uri = Path(provider_dir) if provider_dir else s.provider_dir
    if not (uri / "calendars" / "day.txt").exists():
        raise FileNotFoundError(
            f"{uri} 不是有效的 Qlib 資料集，請先執行 `uv run python -m qlab export`。"
        )
    # 新版 MLflow 預設拒絕 file store backend（qlib workflow 依賴它記錄實驗）
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    s.experiment_dir.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=str(uri),
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": str(s.experiment_dir / "mlruns"), "default_exp_name": "qlab_factor"},
        },
    )


@dataclass(frozen=True)
class ICReport:
    ic_mean: float
    icir: float
    n_test_days: int
    model: str = "LGBModel"
    handler: str = "Alpha158"

    def __str__(self) -> str:  # pragma: no cover - 顯示用
        return (
            f"[{self.handler} + {self.model}] Rank IC={self.ic_mean:.4f} "
            f"ICIR={self.icir:.4f}（test {self.n_test_days} 天）"
        )


def run_alpha158_ic(
    start: str,
    end: str,
    test_start: str | None = None,
    top_k: int = 50,
    settings_: QlabSettings | None = None,
) -> ICReport:
    """訓練 Alpha158 + LightGBM 並在測試區間計算逐日 Rank IC / ICIR。

    Args:
        start / end: 資料完整區間（YYYY-MM-DD）。
        test_start:  測試區間起點（預設最後 20% 交易日）。
        top_k:       Alpha158 參數（label 相關設定），沿用預設即可。
    """
    s = settings_ or qlab_settings()
    init_qlib(s.provider_dir)

    instruments = D.instruments(market="all")
    handler = Alpha158(
        instruments=instruments,
        start_time=start,
        end_time=end,
        fit_start_time=start,
        fit_end_time=test_start or end,
    )

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": {
                "train": (start, test_start or end),
                "test": (test_start or end, end),
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)

    model = LGBModel()
    model.fit(dataset)
    pred = model.predict(dataset)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    pred.columns = ["score"]

    # 測試區間標籤：Alpha158 預設 label = Ref($close, -2)/Ref($close, -1) - 1（隔日報酬）
    label = dataset.prepare("test", col_set="label")
    label.columns = ["label"]
    joined = pred.join(label, how="inner").dropna()

    ic_by_day = (
        joined.groupby(level="datetime")
        .apply(lambda g: g["score"].rank().corr(g["label"].rank()) if len(g) > 1 else np.nan)
        .dropna()
    )
    ic_mean = float(ic_by_day.mean())
    icir = float(ic_by_day.mean() / ic_by_day.std()) if ic_by_day.std() > 0 else 0.0
    return ICReport(ic_mean=ic_mean, icir=icir, n_test_days=int(len(ic_by_day)))

"""qlab 子專案：Qlib + RD-Agent 因子研究（共用 kstock 的 .env 與 data/）。

匯入本套件前 kstock.config.settings 已完成專案根目錄 .env 的載入，
因此 AI 金鑰（OPENAI_API_KEY / DEEPSEEK_API_KEY）與 kstock 共用同一份 .env。
"""

from qlab.config import QlabSettings, qlab_settings

__all__ = ["QlabSettings", "qlab_settings"]

#!/usr/bin/env bash
# 等 fin_factor 主程序結束後自動產出最終因子報告。
# 用法：
#   scripts/wait_fin_factor_report.sh            # 背景監控，完成後寫 data/qlab/factor_report.md
#   scripts/wait_fin_factor_report.sh --now      # 立即產報告（不等）
#
# 判準：fin_factor 主程序（rdagent fin_factor）消失 = 演化結束 → 產報告。
set -euo pipefail
cd "$(dirname "$0")/.."

PROC_MATCH="python -m qlab rdagent fin_factor"
LOG="data/qlab/rdagent_workspace/factor_monitor.log"

if [[ "${1:-}" == "--now" ]]; then
  echo "[$(date '+%H:%M:%S')] 立即產出報告" >> "$LOG"
  uv run python -m qlab factor-report
  echo "[$(date '+%H:%M:%S')] 報告完成" >> "$LOG"
  exit 0
fi

echo "[$(date '+%H:%M:%S')] 開始監控 fin_factor 主程序..." >> "$LOG"
# 等待主程序消失
rc=1
while true; do
  if ! pgrep -f "$PROC_MATCH" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] fin_factor 主程序已結束，共 [$(
      find data/qlab/rdagent_workspace/git_ignore_folder/RD-Agent_workspace -name 'result.h5' 2>/dev/null | wc -l
    )] 個因子" >> "$LOG"
    if uv run python -m qlab factor-report; then
      echo "[$(date '+%H:%M:%S')] 已產出最終報告 data/qlab/factor_report.md" >> "$LOG"
      cat data/qlab/factor_report.md
      exit 0
    fi
    exit 1
  fi
  sleep 120
done

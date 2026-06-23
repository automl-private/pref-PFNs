#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

RESULTS="${RESULTS:-results/gp_log_regret/results.pt}"
OUT_DIR="${OUT_DIR:-figures/gp_log_regret}"
SUMMARY_OUT="${SUMMARY_OUT:-results/gp_log_regret/summary.csv}"
DPI="${DPI:-200}"
TITLE_PREFIX="${TITLE_PREFIX:-GP preference BO}"

cmd=(
  "${PYTHON_BIN}" evaluation/plot_gp_log_regret.py
  --results "${RESULTS}"
  --out-dir "${OUT_DIR}"
  --summary-out "${SUMMARY_OUT}"
  --dpi "${DPI}"
  --title-prefix "${TITLE_PREFIX}"
)

# Optional filter:
#   METHODS="random qeubo qts qei qnei pfn" bash scripts/plot_gp_log_regret.sh
if [[ -n "${METHODS:-}" ]]; then
  cmd+=(--methods ${METHODS})
fi

echo "[plot gp log regret] results=${RESULTS}"
echo "[plot gp log regret] out_dir=${OUT_DIR}"
"${cmd[@]}"

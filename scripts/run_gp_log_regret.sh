#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

PFN_CHECKPOINT="${PFN_CHECKPOINT:-}"
PFN_CONFIG="${PFN_CONFIG:-}"
INPUT_DIM="${INPUT_DIM:-1}"
OUT="${OUT:-results/gp_log_regret/results.pt}"

BUDGET="${BUDGET:-60}"
N_INIT="${N_INIT:-5}"
N_GP_FUNCTIONS="${N_GP_FUNCTIONS:-5}"
N_BO_SEEDS="${N_BO_SEEDS:-10}"
N_GRID="${N_GRID:-500}"
EPS="${EPS:-1e-12}"
DEVICE="${DEVICE:-cpu}"

PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE:-4096}"
QEUBO_NUM_ACQF_SAMPLES="${QEUBO_NUM_ACQF_SAMPLES:-64}"
QEUBO_MAX_FIT_ITER="${QEUBO_MAX_FIT_ITER:-100}"
QEUBO_MAX_FIT_ATTEMPTS="${QEUBO_MAX_FIT_ATTEMPTS:-20}"
QEUBO_FIT_HYPERPARAMS="${QEUBO_FIT_HYPERPARAMS:-1}"
QEUBO_CONTINUOUS_NUM_RESTARTS="${QEUBO_CONTINUOUS_NUM_RESTARTS:-20}"
QEUBO_CONTINUOUS_RAW_SAMPLES="${QEUBO_CONTINUOUS_RAW_SAMPLES:-1024}"
QEUBO_CONTINUOUS_MAXITER="${QEUBO_CONTINUOUS_MAXITER:-100}"

cmd=(
  "${PYTHON_BIN}" evaluation/run_gp_log_regret.py
  --input-dim "${INPUT_DIM}"
  --out "${OUT}"
  --budget "${BUDGET}"
  --n-init "${N_INIT}"
  --n-gp-functions "${N_GP_FUNCTIONS}"
  --n-bo-seeds "${N_BO_SEEDS}"
  --n-grid "${N_GRID}"
  --eps "${EPS}"
  --device "${DEVICE}"
  --pfn-pair-batch-size "${PFN_PAIR_BATCH_SIZE}"
  --qeubo-num-acqf-samples "${QEUBO_NUM_ACQF_SAMPLES}"
  --qeubo-max-fit-iter "${QEUBO_MAX_FIT_ITER}"
  --qeubo-max-fit-attempts "${QEUBO_MAX_FIT_ATTEMPTS}"
  --qeubo-continuous-num-restarts "${QEUBO_CONTINUOUS_NUM_RESTARTS}"
  --qeubo-continuous-raw-samples "${QEUBO_CONTINUOUS_RAW_SAMPLES}"
  --qeubo-continuous-maxiter "${QEUBO_CONTINUOUS_MAXITER}"
)

if [[ -n "${PFN_CHECKPOINT}" ]]; then
  cmd+=(--pfn-checkpoint "${PFN_CHECKPOINT}")
fi
if [[ -n "${PFN_CONFIG}" ]]; then
  cmd+=(--pfn-config "${PFN_CONFIG}")
fi

if [[ "${QEUBO_FIT_HYPERPARAMS}" == "1" ]]; then
  cmd+=(--qeubo-fit-hyperparams)
else
  cmd+=(--no-qeubo-fit-hyperparams)
fi

# Optional filters:
#   METHODS="random qeubo qts qei qnei pfn" bash scripts/run_gp_log_regret.sh
#   EXCLUDE_METHODS="qeubo" bash ...
if [[ -n "${METHODS:-}" ]]; then
  cmd+=(--methods ${METHODS})
fi
if [[ -n "${EXCLUDE_METHODS:-}" ]]; then
  cmd+=(--exclude-methods ${EXCLUDE_METHODS})
fi
if [[ -n "${VERBOSE:-}" ]]; then
  cmd+=(--verbose)
fi

echo "[run gp log regret] out=${OUT}"
echo "[run gp log regret] budget=${BUDGET} n_gp_functions=${N_GP_FUNCTIONS} n_bo_seeds=${N_BO_SEEDS} n_grid=${N_GRID}"
echo "[run gp log regret] input_dim=${INPUT_DIM} gp_support=continuous_rff"
echo "[run gp log regret] pfn_checkpoint=${PFN_CHECKPOINT:-none} pfn_config=${PFN_CONFIG:-none}"
"${cmd[@]}"

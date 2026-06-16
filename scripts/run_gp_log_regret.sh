#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_DIRS="${CHECKPOINT_DIRS:-checkpoints checkpoints2 checkpoints3 checkpoints4}"
CONFIG_DIRS="${CONFIG_DIRS:-my_configs my_configs2 my_configs3}"
OUT="${OUT:-results/gp_log_regret/results.pt}"

BUDGET="${BUDGET:-60}"
N_INIT="${N_INIT:-5}"
N_GP_FUNCTIONS="${N_GP_FUNCTIONS:-5}"
N_BO_SEEDS="${N_BO_SEEDS:-10}"
N_GRID="${N_GRID:-500}"
GRID_DESIGN="${GRID_DESIGN:-uniform}"
GRID_SEED_OFFSET="${GRID_SEED_OFFSET:-20000}"
GP_SUPPORT="${GP_SUPPORT:-grid}"
EPS="${EPS:-1e-12}"
DEVICE="${DEVICE:-cpu}"

PFN_TS_SAMPLES="${PFN_TS_SAMPLES:-2}"
PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE:-4096}"
QEUBO_NUM_ACQF_SAMPLES="${QEUBO_NUM_ACQF_SAMPLES:-512}"
QEUBO_MAX_FIT_ITER="${QEUBO_MAX_FIT_ITER:-100}"
QEUBO_FIT_HYPERPARAMS="${QEUBO_FIT_HYPERPARAMS:-0}"

FIXED_QEUBO_XTOL="${FIXED_QEUBO_XTOL:-1e-6}"
FIXED_QEUBO_MAXFEV="${FIXED_QEUBO_MAXFEV:-100}"
FIXED_QEUBO_MC_SAMPLES="${FIXED_QEUBO_MC_SAMPLES:-512}"
FIXED_QEUBO_BATCH_EVAL_SIZE="${FIXED_QEUBO_BATCH_EVAL_SIZE:-2048}"
FIXED_QEUBO_JITTER="${FIXED_QEUBO_JITTER:-1e-6}"
FIXED_QEUBO_MEAN_CONSTANT="${FIXED_QEUBO_MEAN_CONSTANT:-0.0}"

BENCHMARK_MODE="${BENCHMARK_MODE:-gp_only}"
DETERMINISTIC_BENCHMARKS="${DETERMINISTIC_BENCHMARKS:-}"
DETERMINISTIC_NORMALIZATIONS="${DETERMINISTIC_NORMALIZATIONS:-raw std1}"
DETERMINISTIC_NOISE_STD="${DETERMINISTIC_NOISE_STD:-0.05}"

cmd=(
  "${PYTHON_BIN}" evaluation/run_gp_log_regret.py
  --checkpoint-dirs ${CHECKPOINT_DIRS}
  --config-dirs ${CONFIG_DIRS}
  --out "${OUT}"
  --budget "${BUDGET}"
  --n-init "${N_INIT}"
  --n-gp-functions "${N_GP_FUNCTIONS}"
  --n-bo-seeds "${N_BO_SEEDS}"
  --n-grid "${N_GRID}"
  --grid-design "${GRID_DESIGN}"
  --grid-seed-offset "${GRID_SEED_OFFSET}"
  --gp-support "${GP_SUPPORT}"
  --eps "${EPS}"
  --device "${DEVICE}"
  --pfn-ts-samples "${PFN_TS_SAMPLES}"
  --pfn-pair-batch-size "${PFN_PAIR_BATCH_SIZE}"
  --qeubo-num-acqf-samples "${QEUBO_NUM_ACQF_SAMPLES}"
  --qeubo-max-fit-iter "${QEUBO_MAX_FIT_ITER}"
  --fixed-qeubo-xtol "${FIXED_QEUBO_XTOL}"
  --fixed-qeubo-maxfev "${FIXED_QEUBO_MAXFEV}"
  --fixed-qeubo-mc-samples "${FIXED_QEUBO_MC_SAMPLES}"
  --fixed-qeubo-batch-eval-size "${FIXED_QEUBO_BATCH_EVAL_SIZE}"
  --fixed-qeubo-jitter "${FIXED_QEUBO_JITTER}"
  --fixed-qeubo-mean-constant "${FIXED_QEUBO_MEAN_CONSTANT}"
  --benchmark-mode "${BENCHMARK_MODE}"
  --deterministic-normalizations ${DETERMINISTIC_NORMALIZATIONS}
  --deterministic-noise-std "${DETERMINISTIC_NOISE_STD}"
)

if [[ -n "${DETERMINISTIC_BENCHMARKS}" ]]; then
  cmd+=(--deterministic-benchmarks ${DETERMINISTIC_BENCHMARKS})
fi

if [[ "${QEUBO_FIT_HYPERPARAMS}" == "1" ]]; then
  cmd+=(--qeubo-fit-hyperparams)
fi

# Optional filters:
#   METHODS="random gp_pbo qeubo fixed_qeubo pref_gp_1d_10M" bash scripts/run_gp_log_regret.sh
#   ONLY_CHECKPOINTS="pfn_pref_gp_1d_10M.pt pfn_pref_gp_1d_qeubo_10M.pt" bash ...
#   EXCLUDE_METHODS="qeubo" bash ...
if [[ -n "${METHODS:-}" ]]; then
  cmd+=(--methods ${METHODS})
fi
if [[ -n "${ONLY_CHECKPOINTS:-}" ]]; then
  cmd+=(--only-checkpoints ${ONLY_CHECKPOINTS})
fi
if [[ -n "${EXCLUDE_METHODS:-}" ]]; then
  cmd+=(--exclude-methods ${EXCLUDE_METHODS})
fi
if [[ -n "${SKIP_CHECKPOINTS:-}" ]]; then
  cmd+=(--skip-checkpoints ${SKIP_CHECKPOINTS})
fi
if [[ -n "${VERBOSE:-}" ]]; then
  cmd+=(--verbose)
fi

echo "[run gp log regret] out=${OUT}"
echo "[run gp log regret] budget=${BUDGET} n_gp_functions=${N_GP_FUNCTIONS} n_bo_seeds=${N_BO_SEEDS} n_grid=${N_GRID}"
echo "[run gp log regret] grid_design=${GRID_DESIGN} grid_seed_offset=${GRID_SEED_OFFSET} gp_support=${GP_SUPPORT}"
echo "[run gp log regret] benchmark_mode=${BENCHMARK_MODE} deterministic_benchmarks=${DETERMINISTIC_BENCHMARKS:-default} deterministic_normalizations=${DETERMINISTIC_NORMALIZATIONS}"
"${cmd[@]}"

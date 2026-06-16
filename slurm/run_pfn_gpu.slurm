#!/bin/bash
#SBATCH --job-name=pbo-pfn-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=nils
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
 
set -euo pipefail
 
export HF_HOME="${HF_HOME:-/shared/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/shared/models/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/shared/models/huggingface/hub}"
 
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda activate pfns
 
cd /shared/home/$(whoami)/pref-PFNs
 
EXPERIMENT_NAME="${EXPERIMENT_NAME:-final_pfn_gpu}"
CHECKPOINT_DIRS="${CHECKPOINT_DIRS:-checkpoints checkpoints2 checkpoints3 checkpoints4}"
CONFIG_DIRS="${CONFIG_DIRS:-my_configs my_configs2 my_configs3}"
DEVICE="${DEVICE:-cuda}"
BUDGET="${BUDGET:-100}"
N_INIT="${N_INIT:-5}"
N_BO_SEEDS="${N_BO_SEEDS:-10}"
N_GP_FUNCTIONS="${N_GP_FUNCTIONS:-50}"
N_GRID="${N_GRID:-500}"
GRID_DESIGN="${GRID_DESIGN:-uniform}"
GRID_SEED_OFFSET="${GRID_SEED_OFFSET:-20000}"
GP_SUPPORT="${GP_SUPPORT:-grid}"
EPS="${EPS:-1e-12}"
PFN_TS_SAMPLES="${PFN_TS_SAMPLES:-2}"
PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE:-2048}"
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
OUT="${OUT:-results/gp_log_regret/${EXPERIMENT_NAME}/results.pt}"
METHODS="${METHODS:-pref_gp_1d_10M pref_gp_1d_compare_10M_copeland pref_gp_1d_qeubo_10M pref_gp_1d_qeubo_exp_10M pref_gp_1d_qeubo_regret_10M pref_gp_1d_regret_10M}"
ONLY_CHECKPOINTS="${ONLY_CHECKPOINTS:-}"
EXCLUDE_METHODS="${EXCLUDE_METHODS:-}"
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-}"
VERBOSE="${VERBOSE:-}"
 
mkdir -p logs/slurm
mkdir -p "$(dirname "${OUT}")"
 
echo "=== Job info ==="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURMD_NODENAME}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Checkpoint dirs:  ${CHECKPOINT_DIRS}"
echo "Config dirs:      ${CONFIG_DIRS}"
echo "Device:           ${DEVICE}"
echo "Budget:           ${BUDGET}"
echo "N_INIT:           ${N_INIT}"
echo "N_BO_SEEDS:       ${N_BO_SEEDS}"
echo "N_GP_FUNCTIONS:   ${N_GP_FUNCTIONS}"
echo "N_GRID:           ${N_GRID}"
echo "Grid design:      ${GRID_DESIGN}"
echo "Grid seed offset: ${GRID_SEED_OFFSET}"
echo "GP support:       ${GP_SUPPORT}"
echo "EPS:              ${EPS}"
echo "PFN TS samples:   ${PFN_TS_SAMPLES}"
echo "PFN batch size:   ${PFN_PAIR_BATCH_SIZE}"
echo "qEUBO samples:    ${QEUBO_NUM_ACQF_SAMPLES}"
echo "qEUBO max fit:    ${QEUBO_MAX_FIT_ITER}"
echo "qEUBO fit hpars:  ${QEUBO_FIT_HYPERPARAMS}"
echo "Fixed qEUBO MC:   ${FIXED_QEUBO_MC_SAMPLES}"
echo "Fixed qEUBO batch:${FIXED_QEUBO_BATCH_EVAL_SIZE}"
echo "Benchmark mode:   ${BENCHMARK_MODE}"
echo "Det benchmarks:   ${DETERMINISTIC_BENCHMARKS:-default}"
echo "Det normalizers:  ${DETERMINISTIC_NORMALIZATIONS}"
echo "Det noise std:    ${DETERMINISTIC_NOISE_STD}"
echo "Methods:          ${METHODS}"
echo "Only checkpoints: ${ONLY_CHECKPOINTS:-none}"
echo "Exclude methods:  ${EXCLUDE_METHODS:-none}"
echo "Skip checkpoints: ${SKIP_CHECKPOINTS:-none}"
echo "Output:           ${OUT}"
echo "Started:          $(date)"
 
echo "=== CUDA info ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
 
PYTHON_BIN="${PYTHON_BIN:-/shared/home/kseniia.kuvshinova/miniconda3/envs/pfns/bin/python}"
echo "Python:           ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
 
PYTHON_BIN="${PYTHON_BIN}" \
CHECKPOINT_DIRS="${CHECKPOINT_DIRS}" \
CONFIG_DIRS="${CONFIG_DIRS}" \
DEVICE="${DEVICE}" \
BUDGET="${BUDGET}" \
N_INIT="${N_INIT}" \
N_BO_SEEDS="${N_BO_SEEDS}" \
N_GP_FUNCTIONS="${N_GP_FUNCTIONS}" \
N_GRID="${N_GRID}" \
GRID_DESIGN="${GRID_DESIGN}" \
GRID_SEED_OFFSET="${GRID_SEED_OFFSET}" \
GP_SUPPORT="${GP_SUPPORT}" \
EPS="${EPS}" \
PFN_TS_SAMPLES="${PFN_TS_SAMPLES}" \
PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE}" \
QEUBO_NUM_ACQF_SAMPLES="${QEUBO_NUM_ACQF_SAMPLES}" \
QEUBO_MAX_FIT_ITER="${QEUBO_MAX_FIT_ITER}" \
QEUBO_FIT_HYPERPARAMS="${QEUBO_FIT_HYPERPARAMS}" \
FIXED_QEUBO_XTOL="${FIXED_QEUBO_XTOL}" \
FIXED_QEUBO_MAXFEV="${FIXED_QEUBO_MAXFEV}" \
FIXED_QEUBO_MC_SAMPLES="${FIXED_QEUBO_MC_SAMPLES}" \
FIXED_QEUBO_BATCH_EVAL_SIZE="${FIXED_QEUBO_BATCH_EVAL_SIZE}" \
FIXED_QEUBO_JITTER="${FIXED_QEUBO_JITTER}" \
FIXED_QEUBO_MEAN_CONSTANT="${FIXED_QEUBO_MEAN_CONSTANT}" \
BENCHMARK_MODE="${BENCHMARK_MODE}" \
DETERMINISTIC_BENCHMARKS="${DETERMINISTIC_BENCHMARKS}" \
DETERMINISTIC_NORMALIZATIONS="${DETERMINISTIC_NORMALIZATIONS}" \
DETERMINISTIC_NOISE_STD="${DETERMINISTIC_NOISE_STD}" \
METHODS="${METHODS}" \
ONLY_CHECKPOINTS="${ONLY_CHECKPOINTS}" \
EXCLUDE_METHODS="${EXCLUDE_METHODS}" \
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS}" \
VERBOSE="${VERBOSE}" \
OUT="${OUT}" \
bash scripts/run_gp_log_regret.sh
 
echo "Finished: $(date)"

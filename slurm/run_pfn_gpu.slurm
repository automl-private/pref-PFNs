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
PFN_CHECKPOINT="${PFN_CHECKPOINT:-}"
PFN_CONFIG="${PFN_CONFIG:-}"
INPUT_DIM="${INPUT_DIM:-1}"
DEVICE="${DEVICE:-cuda}"
BUDGET="${BUDGET:-100}"
N_INIT="${N_INIT:-5}"
N_BO_SEEDS="${N_BO_SEEDS:-10}"
N_GP_FUNCTIONS="${N_GP_FUNCTIONS:-50}"
N_GRID="${N_GRID:-500}"
EPS="${EPS:-1e-12}"
PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE:-2048}"
QEUBO_NUM_ACQF_SAMPLES="${QEUBO_NUM_ACQF_SAMPLES:-64}"
QEUBO_MAX_FIT_ITER="${QEUBO_MAX_FIT_ITER:-100}"
QEUBO_MAX_FIT_ATTEMPTS="${QEUBO_MAX_FIT_ATTEMPTS:-20}"
QEUBO_FIT_HYPERPARAMS="${QEUBO_FIT_HYPERPARAMS:-0}"
QEUBO_CONTINUOUS_NUM_RESTARTS="${QEUBO_CONTINUOUS_NUM_RESTARTS:-20}"
QEUBO_CONTINUOUS_RAW_SAMPLES="${QEUBO_CONTINUOUS_RAW_SAMPLES:-1024}"
QEUBO_CONTINUOUS_MAXITER="${QEUBO_CONTINUOUS_MAXITER:-100}"
OUT="${OUT:-results/gp_log_regret/${EXPERIMENT_NAME}/results.pt}"
METHODS="${METHODS:-pfn}"
EXCLUDE_METHODS="${EXCLUDE_METHODS:-}"
VERBOSE="${VERBOSE:-}"
 
mkdir -p logs/slurm
mkdir -p "$(dirname "${OUT}")"
 
echo "=== Job info ==="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURMD_NODENAME}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "PFN checkpoint:   ${PFN_CHECKPOINT:-none}"
echo "PFN config:       ${PFN_CONFIG:-none}"
echo "Input dim:        ${INPUT_DIM}"
echo "Device:           ${DEVICE}"
echo "Budget:           ${BUDGET}"
echo "N_INIT:           ${N_INIT}"
echo "N_BO_SEEDS:       ${N_BO_SEEDS}"
echo "N_GP_FUNCTIONS:   ${N_GP_FUNCTIONS}"
echo "N_GRID:           ${N_GRID}"
echo "GP support:       continuous_rff"
echo "EPS:              ${EPS}"
echo "PFN batch size:   ${PFN_PAIR_BATCH_SIZE}"
echo "qEUBO samples:    ${QEUBO_NUM_ACQF_SAMPLES}"
echo "qEUBO max fit:    ${QEUBO_MAX_FIT_ITER}"
echo "qEUBO fit attempts: ${QEUBO_MAX_FIT_ATTEMPTS}"
echo "qEUBO fit hpars:  ${QEUBO_FIT_HYPERPARAMS}"
echo "qEUBO restarts:   ${QEUBO_CONTINUOUS_NUM_RESTARTS}"
echo "qEUBO raw samples:${QEUBO_CONTINUOUS_RAW_SAMPLES}"
echo "qEUBO maxiter:    ${QEUBO_CONTINUOUS_MAXITER}"
echo "Methods:          ${METHODS}"
echo "Exclude methods:  ${EXCLUDE_METHODS:-none}"
echo "Output:           ${OUT}"
echo "Started:          $(date)"
 
echo "=== CUDA info ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
 
PYTHON_BIN="${PYTHON_BIN:-/shared/home/kseniia.kuvshinova/miniconda3/envs/pfns/bin/python}"
echo "Python:           ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
 
PYTHON_BIN="${PYTHON_BIN}" \
PFN_CHECKPOINT="${PFN_CHECKPOINT}" \
PFN_CONFIG="${PFN_CONFIG}" \
INPUT_DIM="${INPUT_DIM}" \
DEVICE="${DEVICE}" \
BUDGET="${BUDGET}" \
N_INIT="${N_INIT}" \
N_BO_SEEDS="${N_BO_SEEDS}" \
N_GP_FUNCTIONS="${N_GP_FUNCTIONS}" \
N_GRID="${N_GRID}" \
EPS="${EPS}" \
PFN_PAIR_BATCH_SIZE="${PFN_PAIR_BATCH_SIZE}" \
QEUBO_NUM_ACQF_SAMPLES="${QEUBO_NUM_ACQF_SAMPLES}" \
QEUBO_MAX_FIT_ITER="${QEUBO_MAX_FIT_ITER}" \
QEUBO_MAX_FIT_ATTEMPTS="${QEUBO_MAX_FIT_ATTEMPTS}" \
QEUBO_FIT_HYPERPARAMS="${QEUBO_FIT_HYPERPARAMS}" \
QEUBO_CONTINUOUS_NUM_RESTARTS="${QEUBO_CONTINUOUS_NUM_RESTARTS}" \
QEUBO_CONTINUOUS_RAW_SAMPLES="${QEUBO_CONTINUOUS_RAW_SAMPLES}" \
QEUBO_CONTINUOUS_MAXITER="${QEUBO_CONTINUOUS_MAXITER}" \
METHODS="${METHODS}" \
EXCLUDE_METHODS="${EXCLUDE_METHODS}" \
VERBOSE="${VERBOSE}" \
OUT="${OUT}" \
bash scripts/run_gp_log_regret.sh
 
echo "Finished: $(date)"

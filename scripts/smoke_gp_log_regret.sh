#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/Users/kseniakuvshinova/miniforge3/envs/pfns/bin/python}"
SMOKE_ROOT="${SMOKE_ROOT:-/private/tmp/pref_pfns_gp_log_regret_smoke_test}"

RESULTS="${SMOKE_ROOT}/results.pt"
FIGURES="${SMOKE_ROOT}/figures"
SUMMARY="${SMOKE_ROOT}/summary.csv"

rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}"

echo "[smoke] python=${PYTHON_BIN}"
echo "[smoke] root=${SMOKE_ROOT}"

METHODS="random fixed_qeubo pfn" \
PFN_CHECKPOINT="checkpoints2/pfn_pref_gp_1d_qeubo_10M.pt" \
PFN_CONFIG="my_configs2/train_pref_gp_1d_qeubo_10M.py" \
INPUT_DIM=1 \
N_GP_FUNCTIONS=1 \
N_BO_SEEDS=1 \
BUDGET=3 \
N_GRID=12 \
PFN_PAIR_BATCH_SIZE=64 \
FIXED_QEUBO_MC_SAMPLES=16 \
FIXED_QEUBO_MAXFEV=20 \
FIXED_QEUBO_BATCH_EVAL_SIZE=64 \
OUT="${RESULTS}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_gp_log_regret.sh

RESULTS="${RESULTS}" \
OUT_DIR="${FIGURES}" \
SUMMARY_OUT="${SUMMARY}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/plot_gp_log_regret.sh

"${PYTHON_BIN}" - "${RESULTS}" "${FIGURES}" "${SUMMARY}" <<'PY'
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

results_path = Path(sys.argv[1])
figures_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

assert results_path.is_file(), f"missing results: {results_path}"
assert summary_path.is_file(), f"missing summary: {summary_path}"
pngs = sorted(figures_dir.glob("*.png"))
assert pngs, f"no PNG files in {figures_dir}"
assert all(p.stat().st_size > 0 for p in pngs), "empty PNG output"
assert summary_path.stat().st_size > 0, "empty CSV summary"

results = torch.load(results_path, map_location="cpu")
assert "suites" in results and results["suites"], "missing suites"

expected_methods = {
    "random",
    "fixed_qeubo",
    "pfn",
}

for suite_name, suite in results["suites"].items():
    methods = set(suite["methods"])
    missing = expected_methods - methods
    assert not missing, f"{suite_name}: missing methods {missing}"
    for method, payload in suite["methods"].items():
        log10_regret = payload["log10_regret"]
        simple_regret = payload["simple_regret"]
        utility = payload["utility_at_recommendation"]
        assert tuple(log10_regret.shape) == (1, 1, 3), (method, log10_regret.shape)
        assert simple_regret.shape == log10_regret.shape
        assert utility.shape == log10_regret.shape
        assert torch.isfinite(log10_regret).all(), f"{method}: non-finite log regret"
        assert (simple_regret >= -1e-7).all(), f"{method}: negative regret"

print("[smoke] OK")
print(f"[smoke] results={results_path}")
print(f"[smoke] figures={figures_dir}")
print(f"[smoke] summary={summary_path}")
PY

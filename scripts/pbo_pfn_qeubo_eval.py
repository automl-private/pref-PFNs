#!/usr/bin/env python3
"""
Compute PFN qEUBO predictions for all stored ground-truth iterations of one seed.

For each file in:
    runs/<seed>/<it>_<effective_sample_size>.pt

this script computes the PFN qEUBO prediction and stores:
    runs_pfn/<model_name>/<seed>/<it>_<effective_sample_size>_<spearman>_<softargmax>.pt

The saved file mirrors the runs directory and includes summary metrics so aggregate
statistics can be computed later without loading the full qEUBO tensor.

Key features
------------
- resume-safe / preemption-friendly
- skips iterations that already have a stored PFN result
- stores metrics in filename and payload
- optional storage of full predicted qEUBO matrix
- atomic writes

Assumptions
-----------
- `load_model` is importable in your environment
- PFN API:
      model = load_model(model_name)
      logits = model(x_ctx, y_ctx, test_x=x_query_pair)
      qeubo_pred = model.criterion.mean(logits)[0].reshape(M, M)
- stored runs files contain:
      x_plot
      F_post
      comparisons_before_update   (preferred)
  where comparisons are [winner_idx, loser_idx]
- PFN expects:
      y_ctx = 0
  regardless of comparison label semantics

Example
-------
python pbo_pfn_qeubo_eval.py \
    --model pref_gp_1d_qeubo_exp_10M \
    --seed 17 \
    --runs-root runs \
    --out-root runs_pfn

"""

import argparse
import math
import re
import signal
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

from pfns.run_training_cli import load_config_from_python

def load_model(name, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    config_path = f"/work/dlclarge2/adriaens-pref-pfn/my_configs/train_{name}.py"
    checkpoint_path = f"/work/dlclarge2/adriaens-pref-pfn/checkpoints/pfn_{name}.pt"
    # Load config
    config = load_config_from_python(config_path, 0)
    
    # Build model exactly as in training
    model = config.model.create_model().to(device)
    model.eval()
    
    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    
    # Some PFNs checkpoints are plain state_dict, others wrap it
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    print(f"Loading {name} completed!")
    return model


_STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received signal {signum}; will stop after current safe point", flush=True)


signal.signal(signal.SIGTERM, _handle_stop_signal)
signal.signal(signal.SIGUSR1, _handle_stop_signal)


ITER_FILE_RE = re.compile(r"^(\d{3})_(\d+)\.pt$")
PFN_FILE_RE = re.compile(r"^(\d{3})_(\d+)_([-+0-9.eE]+)_([-+0-9.eE]+)\.pt$")


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def compute_qeubo(F: torch.Tensor) -> torch.Tensor:
    # F: (N, M)
    Fi = F[:, :, None]
    Fj = F[:, None, :]
    return torch.maximum(Fi, Fj).mean(dim=0)


def offdiag_mask(M: int, device=None) -> torch.Tensor:
    return ~torch.eye(M, dtype=torch.bool, device=device)


def flatten_offdiag(mat: torch.Tensor) -> torch.Tensor:
    return mat[offdiag_mask(mat.shape[0], device=mat.device)]


def spearman_offdiag(q1: torch.Tensor, q2: torch.Tensor) -> float:
    v1 = flatten_offdiag(q1).detach().cpu().numpy()
    v2 = flatten_offdiag(q2).detach().cpu().numpy()
    return float(spearmanr(v1, v2).correlation)


def soft_argmax(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    1.0 if prediction argmax equals reference argmax
    0.0 if prediction argmax is the reference minimizer
    computed over off-diagonal entries only
    """
    M = reference.shape[0]
    mask = offdiag_mask(M, device=reference.device)

    ref_vec = reference[mask]
    pred_vec = prediction[mask]

    K = ref_vec.numel()
    if K <= 1:
        return float("nan")

    order = torch.argsort(ref_vec, descending=False)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(K, device=order.device, dtype=order.dtype)

    pred_argmax = torch.argmax(pred_vec)
    rank = inv[pred_argmax].item()
    return float(rank / (K - 1))


def soft_argmax_sym(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (soft_argmax(a, b) + soft_argmax(b, a))


def find_run_files(seed_dir: Path):
    out = []
    for p in seed_dir.glob("*.pt"):
        m = ITER_FILE_RE.match(p.name)
        if m:
            it = int(m.group(1))
            ess = int(m.group(2))
            out.append((it, ess, p))
    return sorted(out, key=lambda t: t[0])


def find_existing_pfn_results(seed_out_dir: Path):
    """
    Returns dict:
        it -> {path, ess, spearman, softargmax}
    """
    out = {}
    for p in seed_out_dir.glob("*.pt"):
        m = PFN_FILE_RE.match(p.name)
        if m:
            it = int(m.group(1))
            ess = int(m.group(2))
            spearman = float(m.group(3))
            soft = float(m.group(4))
            out[it] = {
                "path": p,
                "effective_sample_size": ess,
                "spearman": spearman,
                "softargmax": soft,
            }
    return out


def build_pfn_context(x_plot: torch.Tensor, comparisons: torch.Tensor):
    """
    comparisons: (n, 2), [winner_idx, loser_idx]
    returns:
        x_ctx: (1, n, 2)
        y_ctx: (1, n), all zeros
    """
    comparisons = torch.as_tensor(comparisons).reshape(-1, 2).long()

    if comparisons.shape[0] == 0:
        x_ctx = x_plot.new_empty((1, 0, 2))
        y_ctx = x_plot.new_empty((1, 0))
    else:
        winner_x = x_plot[comparisons[:, 0]]
        loser_x = x_plot[comparisons[:, 1]]
        x_ctx = torch.stack([winner_x, loser_x], dim=-1).unsqueeze(0)
        y_ctx = torch.zeros((1, comparisons.shape[0]), dtype=x_plot.dtype)

    return x_ctx, y_ctx


def build_pair_query(x_plot: torch.Tensor):
    X1, X2 = torch.meshgrid(x_plot, x_plot, indexing="ij")
    return torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1).unsqueeze(0)


def get_comparisons_before_update(data):
    if "comparisons_before_update" in data:
        comps = data["comparisons_before_update"]
    elif "comparisons_after_update" in data:
        comps = data["comparisons_after_update"]
    else:
        comps = torch.empty((0, 2), dtype=torch.long)
    return torch.as_tensor(comps).reshape(-1, 2).long()


def safe_float_for_filename(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.6f}"


def evaluate_single_iteration(
    run_path: Path,
    model,
    store_full_qeubo: bool = True,
):
    data = torch.load(run_path, map_location="cpu")

    x_plot = data["x_plot"].reshape(-1).cpu().float()
    F_post = data["F_post"].cpu().float()
    comparisons = get_comparisons_before_update(data)

    M = x_plot.numel()
    qeubo_ref = compute_qeubo(F_post).cpu().float()

    x_ctx, y_ctx = build_pfn_context(x_plot, comparisons)
    x_query_pair = build_pair_query(x_plot)

    param0 = next(model.parameters())
    model_device = param0.device
    model_dtype = param0.dtype

    x_ctx_model = x_ctx.to(device=model_device, dtype=model_dtype)
    y_ctx_model = y_ctx.to(device=model_device, dtype=model_dtype)
    x_query_pair_model = x_query_pair.to(device=model_device, dtype=model_dtype)

    with torch.no_grad():
        logits = model(x_ctx_model, y_ctx_model, test_x=x_query_pair_model)
        qeubo_pred = (
            model.criterion.mean(logits)[0]
            .reshape(M, M)
            .detach()
            .cpu()
            .float()
        )

    spearman = spearman_offdiag(qeubo_ref, qeubo_pred)
    soft = soft_argmax_sym(qeubo_ref, qeubo_pred)

    payload = {
        "seed": data.get("seed", None),
        "iteration": data.get("iteration", None),
        "x_plot": x_plot,
        "effective_sample_size": int(F_post.shape[0]),
        "comparisons_before_update": comparisons.cpu(),
        "num_comparisons": int(comparisons.shape[0]),
        "model_name": getattr(model, "name", None),
        "spearman": float(spearman),
        "softargmax": float(soft),
    }

    if "qeubo_argmax_pair" in data:
        payload["reference_argmax_pair_saved"] = data["qeubo_argmax_pair"].cpu()
    if "observed_winner_loser" in data:
        payload["observed_winner_loser"] = data["observed_winner_loser"].cpu()

    if store_full_qeubo:
        payload["qeubo_pred"] = qeubo_pred
    payload["qeubo_ref_shape"] = torch.tensor(qeubo_ref.shape, dtype=torch.long)

    return payload


def run(
    model_name: str,
    seed: int,
    runs_root: str,
    out_root: str,
    store_full_qeubo: bool = True,
):
    global _STOP_REQUESTED

    in_seed_dir = Path(runs_root) / str(seed)
    if not in_seed_dir.exists():
        raise FileNotFoundError(f"Missing input seed directory: {in_seed_dir}")

    out_seed_dir = Path(out_root) / model_name / str(seed)
    out_seed_dir.mkdir(parents=True, exist_ok=True)

    run_files = find_run_files(in_seed_dir)
    if not run_files:
        print(f"[seed={seed}] no run files found in {in_seed_dir}", flush=True)
        return 0

    existing = find_existing_pfn_results(out_seed_dir)
    missing = [t for t in run_files if t[0] not in existing]

    if not missing:
        print(f"[seed={seed}] all {len(run_files)} iterations already processed", flush=True)
        return 0

    print(
        f"[seed={seed}] found {len(run_files)} GT iterations, "
        f"{len(existing)} already processed, {len(missing)} remaining",
        flush=True,
    )

    print(f"[seed={seed}] loading model {model_name}", flush=True)
    model = load_model(model_name)
    model.eval()

    for it, ess, run_path in run_files:
        if _STOP_REQUESTED:
            print(f"[seed={seed}] stop requested before iteration {it:03d}; exiting cleanly", flush=True)
            return 3

        existing = find_existing_pfn_results(out_seed_dir)
        if it in existing:
            print(f"[seed={seed}] it={it:03d} already present, skipping", flush=True)
            continue

        payload = evaluate_single_iteration(
            run_path=run_path,
            model=model,
            store_full_qeubo=store_full_qeubo,
        )

        spearman = float(payload["spearman"])
        soft = float(payload["softargmax"])

        out_name = (
            f"{it:03d}_{ess}_"
            f"{safe_float_for_filename(spearman)}_"
            f"{safe_float_for_filename(soft)}.pt"
        )
        out_path = out_seed_dir / out_name

        atomic_torch_save(payload, out_path)

        print(
            f"[seed={seed}] it={it:03d} ess={ess:5d} "
            f"spearman={spearman:.4f} softargmax={soft:.4f} "
            f"saved={out_name}",
            flush=True,
        )

    print(f"[seed={seed}] done", flush=True)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="PFN model name passed to load_model")
    p.add_argument("--seed", type=int, required=True, help="Seed directory to process")
    p.add_argument("--runs-root", type=str, default="../slurm/runs", help="Ground-truth runs root")
    p.add_argument("--out-root", type=str, default="../slurm/runs_pfn", help="PFN output root")
    p.add_argument(
        "--no-store-full-qeubo",
        action="store_true",
        help="Do not store full qeubo_pred matrix, only summary metadata",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rc = run(
        model_name=args.model,
        seed=args.seed,
        runs_root=args.runs_root,
        out_root=args.out_root,
        store_full_qeubo=not args.no_store_full_qeubo,
    )
    sys.exit(rc)
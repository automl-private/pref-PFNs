#!/usr/bin/env python3
"""
Compute BoTorch / fixed-GP qEUBO estimates for all stored ground-truth iterations of one seed.

Input:
    ../slurm/runs/<seed>/<it>_<effective_sample_size>.pt

Output:
    runs_botorch/<run_name>/<seed>/<it>_<effective_sample_size>_<spearman>_<softargmax>.pt

Key behavior
------------
- resume-safe / preemption-friendly
- skips iterations already processed
- stores summary metrics in filename and payload
- optional storage of full estimated qEUBO matrix
- uses fixed known hyperparameters
- uses the SAME MC sample count in:
    * prior fallback (no comparisons)
    * posterior qEUBO estimate (BoTorch sampler)

Rank-only convention
--------------------
- We evaluate rank-based metrics only.
- Ground-truth qEUBO is computed from stored F_post in original latent f scale.
- BoTorch estimate is computed in the internally scaled latent utility space:
      u(x) = (f(x) - mean_constant) / noise_std
  and is NOT mapped back to f scale, since only rank matters.

CLI examples
------------
python ../scripts/pbo_botorch_qeubo_eval.py \
    --run-name botorch_default \
    --seed 17

python ../scripts/pbo_botorch_qeubo_eval.py \
    --run-name botorch_conservative \
    --seed 17 \
    --xtol 1e-8 \
    --maxfev 1000 \
    --qeubo-mc-samples 10000
"""

import argparse
import math
import re
import signal
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.models.pairwise_gp import PairwiseGP
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import RBFKernel, ScaleKernel


_STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received signal {signum}; will stop after current safe point", flush=True)


signal.signal(signal.SIGTERM, _handle_stop_signal)
signal.signal(signal.SIGUSR1, _handle_stop_signal)


ITER_FILE_RE = re.compile(r"^(\d{3})_(\d+)\.pt$")
OUT_FILE_RE = re.compile(r"^(\d{3})_(\d+)_([-+0-9.eE]+)_([-+0-9.eE]+)\.pt$")


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def safe_float_for_filename(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.6f}"


def find_run_files(seed_dir: Path):
    out = []
    for p in seed_dir.glob("*.pt"):
        m = ITER_FILE_RE.match(p.name)
        if m:
            it = int(m.group(1))
            ess = int(m.group(2))
            out.append((it, ess, p))
    return sorted(out, key=lambda t: t[0])


def find_existing_results(seed_out_dir: Path):
    """
    Returns dict:
        iteration -> metadata
    """
    out = {}
    for p in seed_out_dir.glob("*.pt"):
        m = OUT_FILE_RE.match(p.name)
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


def compute_qeubo_from_samples(F: torch.Tensor) -> torch.Tensor:
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
    1.0 if prediction argmax is the reference argmax
    0.0 if prediction argmax is the reference minimizer
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


def get_comparisons_before_update(data):
    if "comparisons_before_update" in data:
        comps = data["comparisons_before_update"]
    elif "comparisons_after_update" in data:
        comps = data["comparisons_after_update"]
    else:
        comps = torch.empty((0, 2), dtype=torch.long)
    return torch.as_tensor(comps).reshape(-1, 2).long()


def rbf_kernel_1d(x: torch.Tensor, lengthscale: float, outputscale: float) -> torch.Tensor:
    x = x.reshape(-1, 1)
    d2 = (x - x.T) ** 2
    return outputscale * torch.exp(-0.5 * d2 / (lengthscale ** 2))


def qeubo_from_known_prior_mc(
    x_plot: torch.Tensor,
    *,
    lengthscale: float,
    outputscale: float,
    mean_constant: float,
    jitter: float,
    n_samples: int,
) -> torch.Tensor:
    """
    Prior fallback for empty context.
    Same MC budget as the posterior qEUBO estimator.
    """
    K = rbf_kernel_1d(x_plot, lengthscale=lengthscale, outputscale=outputscale)
    K = K + jitter * torch.eye(x_plot.numel(), dtype=x_plot.dtype, device=x_plot.device)

    L = torch.linalg.cholesky(K)
    z = torch.randn(n_samples, x_plot.numel(), dtype=x_plot.dtype, device=x_plot.device)
    F = mean_constant + z @ L.T
    return compute_qeubo_from_samples(F).float()


def evaluate_single_iteration(
    run_path: Path,
    *,
    lengthscale: float,
    outputscale: float,
    mean_constant: float,
    noise_std: float,
    jitter: float,
    xtol: float,
    maxfev: int,
    qeubo_mc_samples: int,
    batch_eval_size: int,
    store_full_qeubo: bool,
):
    data = torch.load(run_path, map_location="cpu")

    x_plot = data["x_plot"].reshape(-1).cpu().double()
    F_post = data["F_post"].cpu().double()
    comparisons = get_comparisons_before_update(data).cpu().long()

    M = x_plot.numel()
    qeubo_ref = compute_qeubo_from_samples(F_post).cpu().float()

    if comparisons.shape[0] == 0:
        qeubo_est = qeubo_from_known_prior_mc(
            x_plot=x_plot,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            jitter=jitter,
            n_samples=qeubo_mc_samples,
        )
        estimator_name = "known_prior_mc"
    else:
        train_X = x_plot.unsqueeze(-1)
        train_comp = comparisons

        # Internal scaling to match comparison noise in a unit-noise probit model:
        # u(x) = (f(x) - mean_constant) / noise_std
        base_kernel = RBFKernel()
        base_kernel.lengthscale = lengthscale

        covar_module = ScaleKernel(base_kernel)
        covar_module.outputscale = outputscale / (noise_std ** 2)

        # fixed hypers
        base_kernel.raw_lengthscale.requires_grad_(False)
        covar_module.raw_outputscale.requires_grad_(False)

        model = PairwiseGP(
            train_X,
            train_comp,
            covar_module=covar_module,
            jitter=jitter,
            xtol=xtol,
            maxfev=maxfev,
        ).eval()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([qeubo_mc_samples]))
        acqf = qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

        X1, X2 = torch.meshgrid(x_plot, x_plot, indexing="ij")
        pair_grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1).unsqueeze(-1)  # (M*M, 2, 1)

        values = []
        with torch.no_grad():
            for start in range(0, pair_grid.shape[0], batch_eval_size):
                X_batch = pair_grid[start:start + batch_eval_size]
                v = acqf(X_batch)  # latent utility scale
                values.append(v.cpu())

        qeubo_est = torch.cat(values).reshape(M, M).float()
        estimator_name = "botorch_pairwisegp"

    spearman = spearman_offdiag(qeubo_ref, qeubo_est)
    soft = soft_argmax_sym(qeubo_ref, qeubo_est)

    payload = {
        "seed": data.get("seed", None),
        "iteration": data.get("iteration", None),
        "effective_sample_size": int(F_post.shape[0]),
        "x_plot": x_plot.float(),
        "comparisons_before_update": comparisons.cpu(),
        "num_comparisons": int(comparisons.shape[0]),
        "spearman": float(spearman),
        "softargmax": float(soft),
        "estimator_name": estimator_name,
        "rank_only": True,
        "lengthscale": float(lengthscale),
        "outputscale": float(outputscale),
        "mean_constant": float(mean_constant),
        "noise_std": float(noise_std),
        "jitter": float(jitter),
        "xtol": float(xtol),
        "maxfev": int(maxfev),
        "qeubo_mc_samples": int(qeubo_mc_samples),
        "batch_eval_size": int(batch_eval_size),
    }

    if "qeubo_argmax_pair" in data:
        payload["reference_argmax_pair_saved"] = data["qeubo_argmax_pair"].cpu()
    if "observed_winner_loser" in data:
        payload["observed_winner_loser"] = data["observed_winner_loser"].cpu()

    if store_full_qeubo:
        payload["qeubo_est"] = qeubo_est.cpu()

    return payload


def run(
    *,
    run_name: str,
    seed: int,
    runs_root: str,
    out_root: str,
    lengthscale: float,
    outputscale: float,
    mean_constant: float,
    noise_std: float,
    jitter: float,
    xtol: float,
    maxfev: int,
    qeubo_mc_samples: int,
    batch_eval_size: int,
    store_full_qeubo: bool,
):
    global _STOP_REQUESTED

    in_seed_dir = Path(runs_root) / str(seed)
    if not in_seed_dir.exists():
        raise FileNotFoundError(f"Missing input seed directory: {in_seed_dir}")

    out_seed_dir = Path(out_root) / run_name / str(seed)
    out_seed_dir.mkdir(parents=True, exist_ok=True)

    run_files = find_run_files(in_seed_dir)
    if not run_files:
        print(f"[seed={seed}] no run files found in {in_seed_dir}", flush=True)
        return 0

    existing = find_existing_results(out_seed_dir)
    missing = [t for t in run_files if t[0] not in existing]

    if not missing:
        print(f"[seed={seed}] all {len(run_files)} iterations already processed", flush=True)
        return 0

    print(
        f"[seed={seed}] found {len(run_files)} GT iterations, "
        f"{len(existing)} already processed, {len(missing)} remaining",
        flush=True,
    )
    print(
        f"[seed={seed}] run_name={run_name} "
        f"xtol={xtol} maxfev={maxfev} qeubo_mc_samples={qeubo_mc_samples}",
        flush=True,
    )

    for it, ess, run_path in run_files:
        if _STOP_REQUESTED:
            print(f"[seed={seed}] stop requested before iteration {it:03d}; exiting cleanly", flush=True)
            return 3

        existing = find_existing_results(out_seed_dir)
        if it in existing:
            print(f"[seed={seed}] it={it:03d} already present, skipping", flush=True)
            continue

        payload = evaluate_single_iteration(
            run_path=run_path,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            noise_std=noise_std,
            jitter=jitter,
            xtol=xtol,
            maxfev=maxfev,
            qeubo_mc_samples=qeubo_mc_samples,
            batch_eval_size=batch_eval_size,
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
    p.add_argument("--run-name", type=str, required=True, help="Output subdirectory name, e.g. botorch_default")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs-root", type=str, default="../slurm/runs")
    p.add_argument("--out-root", type=str, default="runs_botorch")

    p.add_argument("--lengthscale", type=float, default=0.2)
    p.add_argument("--outputscale", type=float, default=1.0)
    p.add_argument("--mean-constant", type=float, default=0.0)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--jitter", type=float, default=1e-6)

    p.add_argument("--xtol", type=float, default=1e-6)
    p.add_argument("--maxfev", type=int, default=100)
    p.add_argument("--qeubo-mc-samples", type=int, default=512)
    p.add_argument("--batch-eval-size", type=int, default=2048)

    p.add_argument(
        "--no-store-full-qeubo",
        action="store_true",
        help="Do not store full estimated qEUBO matrix, only summary metadata",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rc = run(
        run_name=args.run_name,
        seed=args.seed,
        runs_root=args.runs_root,
        out_root=args.out_root,
        lengthscale=args.lengthscale,
        outputscale=args.outputscale,
        mean_constant=args.mean_constant,
        noise_std=args.noise_std,
        jitter=args.jitter,
        xtol=args.xtol,
        maxfev=args.maxfev,
        qeubo_mc_samples=args.qeubo_mc_samples,
        batch_eval_size=args.batch_eval_size,
        store_full_qeubo=not args.no_store_full_qeubo,
    )
    sys.exit(rc)
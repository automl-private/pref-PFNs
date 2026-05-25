#!/usr/bin/env python3
"""
Slurm-friendly preferential BO simulator with rejection-sampling qEUBO ground truth.

What it does
------------
1. Samples one target GP path f_true on a fixed 1D grid using the provided seed.
2. Runs 50 preferential BO iterations.
3. At each iteration:
   - samples up to N=10_000 accepted posterior draws by rejection sampling
     (or stops after max_draws=1_000_000 proposals),
   - stores the accepted posterior samples (F, Y) to:
         <out_root>/<seed>/<it>_<effective_sample_size>.pt
   - approximates qEUBO(x_i, x_j) = E[max(F_i, F_j)] from the accepted F samples,
   - picks the argmax pair,
   - queries the target function with fresh comparison noise,
   - appends the resulting winner/loser pair to the comparison context.

Saved files
-----------
- target.pt
- 000_1234.pt, 001_10000.pt, ...

Each iteration file contains accepted posterior samples and metadata.

Usage
-----
python prefbo_ground_truth_run.py 123

Example Slurm
-------------
sbatch --wrap="python prefbo_ground_truth_run.py 123 --out-root runs"

Notes
-----
- Comparisons are stored as grid indices [winner_idx, loser_idx].
- The posterior sampler conditions on comparisons through hard rejection using Y.
- qEUBO is approximated from accepted latent F samples, not Y.
"""

import argparse
import math
import os
from pathlib import Path

import gpytorch
import torch


# ============================================================
# GP utilities
# ============================================================

def make_gp_prior(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    jitter=1e-6,
):
    mean_module = gpytorch.means.ConstantMean()
    mean_module.initialize(constant=mean_constant)

    base_kernel = gpytorch.kernels.RBFKernel()
    base_kernel.lengthscale = lengthscale

    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    covar_module.outputscale = outputscale

    return gpytorch.distributions.MultivariateNormal(
        mean_module(X),
        covar_module(X),
    ).add_jitter(jitter)


def sample_gp_batch(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=None,
    jitter=1e-6,
):
    """
    Single GP path sample.
    Returns:
        f: (..., M) here actually (M,)
        y: (..., M) same shape
    """
    with torch.no_grad():
        f_dist = make_gp_prior(
            X,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            jitter=jitter,
        )

        f = f_dist.rsample()

        if noise_std is None:
            y = f
        else:
            y = f + noise_std * torch.randn_like(f)

    return f.detach(), y.detach()


# ============================================================
# Rejection posterior sampler
# ============================================================

def rejection_sample_pref_gp_posterior(
    x_plot,
    comparisons,
    N,
    *,
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    jitter=1e-6,
    max_draws=1_000_000,
):
    """
    Rejection sampler for iid posterior samples on the fixed grid.

    comparisons:
        tensor of shape (n, 2), rows are [winner_idx, loser_idx]
    """
    if x_plot.ndim == 1:
        x_plot = x_plot.unsqueeze(-1)

    device = x_plot.device
    M = x_plot.shape[0]

    comparisons = torch.as_tensor(comparisons, device=device)

    if comparisons.numel() == 0:
        comparisons = torch.empty((0, 2), device=device, dtype=torch.long)
    else:
        comparisons = comparisons.reshape(-1, 2).long()

    if comparisons.shape[0] == 0:
        F_list = []
        Y_list = []
        for _ in range(N):
            f, y = sample_gp_batch(
                x_plot,
                lengthscale=lengthscale,
                outputscale=outputscale,
                mean_constant=mean_constant,
                noise_std=noise_std,
                jitter=jitter,
            )
            F_list.append(f)
            Y_list.append(y)

        F_post = torch.stack(F_list, dim=0)
        Y_post = torch.stack(Y_list, dim=0)
        return F_post, Y_post, 1.0, N, N

    winner_idx = comparisons[:, 0]
    loser_idx = comparisons[:, 1]

    if (winner_idx < 0).any() or (winner_idx >= M).any() or (loser_idx < 0).any() or (loser_idx >= M).any():
        raise ValueError("Comparison indices out of bounds.")
    if (winner_idx == loser_idx).any():
        raise ValueError("Self-comparisons are not allowed.")

    accepted_F = []
    accepted_Y = []
    n_accept = 0
    n_total = 0

    while n_accept < N and n_total < max_draws:
        f, y = sample_gp_batch(
            x_plot,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            noise_std=noise_std,
            jitter=jitter,
        )

        if torch.all(y[winner_idx] > y[loser_idx]):
            accepted_F.append(f)
            accepted_Y.append(y)
            n_accept += 1

        n_total += 1

    if n_accept == 0:
        return (
            torch.empty((0, M), device=device, dtype=x_plot.dtype),
            torch.empty((0, M), device=device, dtype=x_plot.dtype),
            0.0,
            0,
            n_total,
        )

    F_post = torch.stack(accepted_F, dim=0)
    Y_post = torch.stack(accepted_Y, dim=0)
    accept_rate = n_accept / max(n_total, 1)
    return F_post, Y_post, accept_rate, n_accept, n_total


# ============================================================
# qEUBO approximation and BO step
# ============================================================

def approximate_qeubo_matrix(F_samples):
    """
    F_samples: (N_eff, M)
    returns:
        qeubo: (M, M), where qeubo[i, j] = E[max(F_i, F_j)]
    """
    if F_samples.ndim != 2:
        raise ValueError(f"Expected F_samples of shape (N, M), got {tuple(F_samples.shape)}")
    Fi = F_samples[:, :, None]
    Fj = F_samples[:, None, :]
    return torch.maximum(Fi, Fj).mean(dim=0)


def pick_argmax_pair(qeubo, allow_diagonal=False):
    """
    returns:
        i_star, j_star
    """
    q = qeubo.clone()
    if not allow_diagonal:
        idx = torch.arange(q.shape[0], device=q.device)
        q[idx, idx] = -torch.inf

    flat_idx = torch.argmax(q)
    i_star = (flat_idx // q.shape[1]).item()
    j_star = (flat_idx % q.shape[1]).item()
    return i_star, j_star


def query_pairwise_comparison_from_target(f_true, i, j, noise_std):
    """
    Uses fresh comparison noise from the target path.
    Returns winner_idx, loser_idx and the noisy utilities used.
    """
    yi = f_true[i] + noise_std * torch.randn_like(f_true[i])
    yj = f_true[j] + noise_std * torch.randn_like(f_true[j])

    if yi >= yj:
        return i, j, yi.detach(), yj.detach()
    else:
        return j, i, yj.detach(), yi.detach()


# ============================================================
# Main simulation
# ============================================================

def run(
    seed,
    out_root,
    T=50,
    M=101,
    N_accept=10_000,
    max_draws=1_000_000,
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    jitter=1e-6,
    device="cpu",
):
    torch.manual_seed(seed)

    device = torch.device(device)
    seed_dir = Path(out_root) / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Fixed 1D grid
    x_plot = torch.linspace(0.0, 1.0, M, device=device)

    # Sample target function once
    x_plot_col = x_plot.unsqueeze(-1)
    f_true, y_true = sample_gp_batch(
        x_plot_col,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    torch.save(
        {
            "seed": seed,
            "x_plot": x_plot.cpu(),
            "f_true": f_true.cpu(),
            "y_true_single_draw": y_true.cpu(),
            "lengthscale": lengthscale,
            "outputscale": outputscale,
            "mean_constant": mean_constant,
            "noise_std": noise_std,
            "jitter": jitter,
            "M": M,
            "T": T,
            "N_accept_target": N_accept,
            "max_draws": max_draws,
        },
        seed_dir / "target.pt",
    )

    # Context of observed comparisons, stored as winner/loser grid indices
    comparisons = torch.empty((0, 2), dtype=torch.long, device=device)

    for it in range(T):
        F_post, Y_post, accept_rate, n_eff, n_total = rejection_sample_pref_gp_posterior(
            x_plot,
            comparisons,
            N_accept,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            noise_std=noise_std,
            jitter=jitter,
            max_draws=max_draws,
        )

        if n_eff == 0:
            raise RuntimeError(
                f"Iteration {it}: rejection sampler returned zero accepted samples "
                f"after {n_total} proposals."
            )

        qeubo = approximate_qeubo_matrix(F_post)
        i_star, j_star = pick_argmax_pair(qeubo, allow_diagonal=False)

        winner, loser, y_winner, y_loser = query_pairwise_comparison_from_target(
            f_true, i_star, j_star, noise_std=noise_std
        )

        comparisons = torch.cat(
            [comparisons, torch.tensor([[winner, loser]], device=device, dtype=torch.long)],
            dim=0,
        )

        out_path = seed_dir / f"{it:03d}_{n_eff}.pt"
        torch.save(
            {
                "seed": seed,
                "iteration": it,
                "x_plot": x_plot.cpu(),
                "comparisons_before_update": comparisons[:-1].cpu() if comparisons.shape[0] > 1 else torch.empty((0, 2), dtype=torch.long),
                "comparisons_after_update": comparisons.cpu(),
                "F_post": F_post.cpu(),
                "Y_post": Y_post.cpu(),
                "effective_sample_size": n_eff,
                "num_proposals": n_total,
                "accept_rate": accept_rate,
                "qeubo_argmax_pair": torch.tensor([i_star, j_star], dtype=torch.long),
                "queried_x_pair": x_plot[torch.tensor([i_star, j_star], device=device)].cpu(),
                "observed_winner_loser": torch.tensor([winner, loser], dtype=torch.long),
                "observed_noisy_utilities_winner_loser": torch.stack([y_winner, y_loser]).cpu(),
            },
            out_path,
        )

        print(
            f"[seed={seed}] it={it:03d} "
            f"n_eff={n_eff:5d} "
            f"proposals={n_total:7d} "
            f"accept_rate={accept_rate:.6f} "
            f"argmax=({i_star:3d},{j_star:3d}) "
            f"obs=({winner:3d}>{loser:3d})"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("--out-root", type=str, default="runs", help="Output root directory")
    parser.add_argument("--T", type=int, default=50, help="Number of BO iterations")
    parser.add_argument("--M", type=int, default=101, help="Number of 1D grid points")
    parser.add_argument("--N-accept", type=int, default=10_000, help="Target accepted posterior samples per iteration")
    parser.add_argument("--max-draws", type=int, default=1_000_000, help="Max rejection proposals per iteration")
    parser.add_argument("--lengthscale", type=float, default=0.2)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--mean-constant", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        seed=args.seed,
        out_root=args.out_root,
        T=args.T,
        M=args.M,
        N_accept=args.N_accept,
        max_draws=args.max_draws,
        lengthscale=args.lengthscale,
        outputscale=args.outputscale,
        mean_constant=args.mean_constant,
        noise_std=args.noise_std,
        jitter=args.jitter,
        device=args.device,
    )
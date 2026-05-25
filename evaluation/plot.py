"""
Plotting utilities for evaluation results.

Usage:
    python evaluation/plot.py --results results/eval.pt --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {
    "random": "gray",
    "gp_pbo": "steelblue",
    "pfn":    "crimson",
}

LABELS = {
    "random": "Random",
    "gp_pbo": "GP-PBO (Laplace)",
    "pfn":    "PFN (ours)",
}


def _mean_ci(regret_tensor: torch.Tensor, ci: float = 0.95):
    """
    regret_tensor: (n_seeds, budget)
    Returns (mean, lo, hi) each of shape (budget,).
    """
    n = regret_tensor.shape[0]
    mean = regret_tensor.mean(0).numpy()
    std  = regret_tensor.std(0).numpy()
    z = 1.96 if ci == 0.95 else 1.0
    se = z * std / np.sqrt(n)
    return mean, mean - se, mean + se


# ---------------------------------------------------------------------------
# Per-benchmark plot
# ---------------------------------------------------------------------------

def plot_benchmark(
    results: dict,
    benchmark_name: str,
    ax: plt.Axes,
    log_scale: bool = False,
):
    for method, bench_results in results.items():
        if benchmark_name not in bench_results:
            continue
        sr = bench_results[benchmark_name]["simple_regret"]  # (n_seeds, budget)
        mean, lo, hi = _mean_ci(sr)
        steps = np.arange(len(mean))

        color = COLORS.get(method, "black")
        label = LABELS.get(method, method)

        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

    ax.set_title(benchmark_name)
    ax.set_xlabel("# comparisons")
    ax.set_ylabel("Simple Regret")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_all(
    results: dict,
    out_dir: str = "figures",
    log_scale: bool = False,
    aggregate: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_names = list(next(iter(results.values())).keys())
    n = len(benchmark_names)

    # --- individual plots ---
    for bname in benchmark_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_benchmark(results, bname, ax, log_scale=log_scale)
        fig.tight_layout()
        fig.savefig(out_dir / f"{bname}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / bname}.png")

    # --- grid plot ---
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, bname in enumerate(benchmark_names):
        plot_benchmark(results, bname, axes[i], log_scale=log_scale)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "all_benchmarks.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'all_benchmarks.png'}")

    # --- aggregate (mean over benchmarks) ---
    if aggregate:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in results:
            all_sr = []
            for bname in benchmark_names:
                if bname in results[method]:
                    all_sr.append(results[method][bname]["simple_regret"])
            if not all_sr:
                continue
            stacked = torch.cat(all_sr, dim=0)  # (n_seeds * n_benchmarks, budget)
            mean, lo, hi = _mean_ci(stacked)
            steps = np.arange(len(mean))
            color = COLORS.get(method, "black")
            label = LABELS.get(method, method)
            ax.plot(steps, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps, lo, hi, alpha=0.2, color=color)

        ax.set_title("Aggregate (all benchmarks)")
        ax.set_xlabel("# comparisons")
        ax.set_ylabel("Simple Regret")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "aggregate.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'aggregate.png'}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/eval.pt")
    parser.add_argument("--out", type=str, default="figures")
    parser.add_argument("--log", action="store_true", help="Log-scale y axis")
    args = parser.parse_args()

    results = torch.load(args.results, map_location="cpu")
    plot_all(results, out_dir=args.out, log_scale=args.log)


if __name__ == "__main__":
    main()

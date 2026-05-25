"""
Main evaluation script.

Usage:
    python evaluation/run_eval.py \
        --checkpoint /path/to/pfn.pt \
        --config     /path/to/train_pref_gp_1d_10M.py \
        --budget     60 \
        --n_init     5 \
        --n_seeds    10 \
        --out        results/eval.pt

Results are saved as a dict:
    {
      "method_name": {
        "benchmark_name": {
          "simple_regret": Tensor(n_seeds, budget),
        }
      }
    }
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

# make sure the package is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle import make_benchmarks
from evaluation.loop import run_bo_loop
from evaluation.agents import RandomAgent, PFNAgent, GPPBOAgent, QEUBOAgent


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pfn_model(config_path: str, checkpoint_path: str, device: str) -> object:
    from pfns.run_training_cli import load_config_from_python
    config = load_config_from_python(config_path, 0)
    model = config.model.create_model().to(device)
    model.eval()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(f"Loaded PFN from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _plot_results(results: dict, out_path: str) -> None:
    """Save one subplot per benchmark with median ± IQR regret curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench_names = sorted({b for method in results.values() for b in method})
    n_bench = len(bench_names)
    fig, axes = plt.subplots(1, n_bench, figsize=(5 * n_bench, 4), squeeze=False)

    for col, bname in enumerate(bench_names):
        ax = axes[0][col]
        for method, bench_data in results.items():
            if bname not in bench_data:
                continue
            sr = bench_data[bname]["simple_regret"].numpy()  # (n_seeds, budget)
            steps = np.arange(1, sr.shape[1] + 1)
            median = np.median(sr, axis=0)
            q25 = np.percentile(sr, 25, axis=0)
            q75 = np.percentile(sr, 75, axis=0)
            (line,) = ax.plot(steps, median, label=method)
            ax.fill_between(steps, q25, q75, alpha=0.2, color=line.get_color())
        ax.set_title(bname)
        ax.set_xlabel("Comparisons")
        ax.set_ylabel("Simple regret")
        ax.legend()
        ax.set_yscale("log")

    fig.tight_layout()
    plot_path = str(Path(out_path).with_suffix(".png"))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PFN checkpoint (.pt). If omitted, PFN agent is skipped.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training config .py (for model architecture).")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of BO seeds per (method, benchmark) pair.")
    parser.add_argument("--n_gp_benchmarks", type=int, default=5,
                        help="Number of random GP benchmark functions.")
    parser.add_argument("--noise_prob", type=float, default=0.05,
                        help="Oracle comparison flip probability.")
    parser.add_argument("--out", type=str, default="results/eval.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step info during BO loop.")
    parser.add_argument("--plot", action="store_true",
                        help="Save regret curves as a PNG next to --out.")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    # --- benchmarks ---
    benchmarks = make_benchmarks(
        n_gp_seeds=args.n_gp_benchmarks,
        noise_prob=args.noise_prob,
        device=args.device,
    )
    print(f"Benchmarks: {[b['name'] for b in benchmarks]}")

    # --- agents ---
    agents: dict[str, object] = {
        "random": RandomAgent(seed=0),
        # "gp_pbo": GPPBOAgent(lengthscale=0.2, outputscale=1.0),
        # "qeubo":  QEUBOAgent(fit_hyperparams=True),
    }
    if args.checkpoint and args.config:
        model = load_pfn_model(args.config, args.checkpoint, args.device)
        agents["pfn"] = PFNAgent(model, device=args.device)
    else:
        print("No checkpoint/config provided — skipping PFN agent.")

    # --- run ---
    results = {name: {} for name in agents}

    for bench in benchmarks:
        bname = bench["name"]
        oracle = bench["oracle"]
        print(f"\n=== Benchmark: {bname} (f* = {oracle.f_opt:.3f} at x* = {oracle.x_opt:.3f}) ===")

        for agent_name, agent in agents.items():
            regret_matrix = []
            for seed in range(args.n_seeds):
                if args.verbose:
                    print(f"  [{agent_name}] seed={seed}")
                run = run_bo_loop(
                    agent=agent,
                    oracle=oracle,
                    budget=args.budget,
                    n_init=args.n_init,
                    seed=seed,
                    verbose=args.verbose,
                )
                regret_matrix.append(run["simple_regret"])

                final_sr = run["simple_regret"][-1]
                print(f"  [{agent_name}] seed={seed:2d}  final SR={final_sr:.4f}")

            results[agent_name][bname] = {
                "simple_regret": torch.tensor(regret_matrix),  # (n_seeds, budget)
            }

        # save intermediate results + plot after every benchmark
        torch.save(results, args.out)
        if args.plot:
            _plot_results(results, args.out)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

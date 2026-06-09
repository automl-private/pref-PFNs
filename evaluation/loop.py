"""
Core BO loop. Runs a single agent on a single oracle for `budget` steps.
Returns a dict with the simple regret curve and metadata.
"""

from __future__ import annotations

import random

import torch

from .agents.base import PBOAgent, Comparison
from .oracle import Oracle


def _candidate_value(candidate):
    """Convert a grid tensor row to a scalar float or multidim point tuple."""
    candidate = torch.as_tensor(candidate)
    if candidate.ndim == 0 or candidate.numel() == 1:
        # Превращает scalar tensor в обычный Python float
        return float(candidate.reshape(-1)[0].item())
    # плоский вектор
    return tuple(float(v) for v in candidate.reshape(-1).tolist())

def _point_str(point) -> str:
    """Format scalar or multidim points for verbose BO-loop logging."""
    if isinstance(point, (tuple, list)):
        return "(" + ", ".join(f"{float(v):.3f}" for v in point) + ")"
    tensor = torch.as_tensor(point)
    if tensor.ndim > 0 and tensor.numel() > 1:
        return "(" + ", ".join(f"{float(v):.3f}" for v in tensor.reshape(-1).tolist()) + ")"
    return f"{float(tensor.reshape(-1)[0].item()):.3f}"


def run_bo_loop(
    agent: PBOAgent,
    oracle: Oracle,
    budget: int = 50,
    n_init: int = 5,
    seed: int = 0,
    n_grid: int = 500,
    verbose: bool = False,
) -> dict:
    """
    Args:
        agent:    PBOAgent instance.
        oracle:   Oracle instance.
        budget:   Total number of comparisons (including init).
        n_init:   Number of random initial comparisons.
        seed:     RNG seed for reproducibility.
        n_grid:   Size of the candidate pool (linspace over [0, 1]).
        verbose:  If True, print per-step info.

    Returns dict:
        {
            "simple_regret": list[float],   # length = budget
            "recommendations": list[float], # x_hat at each step
            "comparisons": list[Comparison],
        }
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    candidate_pool = oracle.x_grid   # use oracle's grid as candidate pool

    comparisons: list[Comparison] = []
    simple_regrets: list[float] = []
    recommendations: list[float] = []

    agent.reset()

    for t in range(budget):
        # --- suggest pair ---
        if t < n_init:
            idx1, idx2 = rng.sample(range(len(candidate_pool)), 2)
            x1 = _candidate_value(candidate_pool[idx1])
            x2 = _candidate_value(candidate_pool[idx2])
            phase = "init"
        else:
            x1, x2 = agent.suggest_pair(comparisons, candidate_pool)
            phase = "bo"

        # ищет ближайшие точки к x1, x2 на сетке и сравнивает
        winner, loser = oracle.compare(x1, x2)
        comparisons.append((winner, loser))

        # --- recommend current best ---
        x_hat = agent.recommend(comparisons, candidate_pool)
        sr = oracle.simple_regret(x_hat)

        simple_regrets.append(sr)
        recommendations.append(x_hat)

        if verbose:
            print(
                f"    t={t+1:3d}/{budget} [{phase:4s}]"
                f"  pair=({_point_str(x1)}, {_point_str(x2)})"
                f"  winner={_point_str(winner)}"
                f"  x_hat={_point_str(x_hat)}  SR={sr:.4f}"
            )

    return {
        "simple_regret": simple_regrets,
        "recommendations": recommendations,
        "comparisons": comparisons,
    }

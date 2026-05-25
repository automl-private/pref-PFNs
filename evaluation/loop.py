"""
Core BO loop. Runs a single agent on a single oracle for `budget` steps.
Returns a dict with the simple regret curve and metadata.
"""

from __future__ import annotations

import random

import torch

from .agents.base import PBOAgent, Comparison
from .oracle import Oracle


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
            pool = candidate_pool.tolist()
            x1, x2 = rng.sample(pool, 2)
            phase = "init"
        else:
            x1, x2 = agent.suggest_pair(comparisons, candidate_pool)
            phase = "bo"

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
                f"  pair=({x1:.3f}, {x2:.3f})  winner={winner:.3f}"
                f"  x_hat={x_hat:.3f}  SR={sr:.4f}"
            )

    return {
        "simple_regret": simple_regrets,
        "recommendations": recommendations,
        "comparisons": comparisons,
    }

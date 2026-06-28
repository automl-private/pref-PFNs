"""
Core BO loop. Runs a single agent on a single oracle for `budget` steps.
Returns a dict with the simple regret curve and metadata.
"""

from __future__ import annotations

import random

import torch

from .agents.base import PBOAgent, Comparison, Point, candidate_value

def _point_str(point) -> str:
    """Format scalar or multidim points for verbose BO-loop logging."""
    if isinstance(point, (tuple, list)):
        return "(" + ", ".join(f"{float(v):.3f}" for v in point) + ")"
    tensor = torch.as_tensor(point)
    if tensor.ndim > 0 and tensor.numel() > 1:
        return "(" + ", ".join(f"{float(v):.3f}" for v in tensor.reshape(-1).tolist()) + ")"
    return f"{float(tensor.reshape(-1)[0].item()):.3f}"

def _candidate_pool_for_step(
    oracle,
    *,
    n_grid: int,
    generator: torch.Generator,
) -> torch.Tensor:
    support = getattr(oracle, "support", "grid")

    if support == "grid":
        candidate_pool = getattr(oracle, "x_grid", None)
        if candidate_pool is None:
            raise ValueError("Grid oracle must expose x_grid.")
        return candidate_pool

    if support == "continuous_rff":
        input_dim = int(getattr(oracle, "input_dim"))
        pool = torch.rand(n_grid, input_dim, generator=generator, dtype=torch.float32)
        if input_dim == 1:
            return pool[:, 0]
        return pool

    raise ValueError(f"Unknown oracle support {support!r}.")


def run_bo_loop(
    agent: PBOAgent,
    oracle,
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
        n_grid:   Grid size in grid mode; per-step candidate-pool size in continuous mode.
        verbose:  If True, print per-step info.

    Returns dict:
        {
            "simple_regret": list[float],   # length = budget
            "recommendations": list[Point], # x_hat at each step
            "comparisons": list[Comparison],
        }
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    rng = random.Random(seed)
    torch.manual_seed(seed)

    candidate_generator = torch.Generator(device="cpu")
    candidate_generator.manual_seed(int(seed))

    comparisons: list[Comparison] = []
    simple_regrets: list[float] = []
    recommendations: list[Point] = []

    agent.reset()

    for t in range(budget):
        candidate_pool = _candidate_pool_for_step(
            oracle,
            n_grid=n_grid,
            generator=candidate_generator,
        )
        if len(candidate_pool) < 2:
            raise ValueError("candidate_pool must contain at least two candidates.")
        # --- suggest pair ---
        if t < n_init:
            idx1, idx2 = rng.sample(range(len(candidate_pool)), 2)
            x1 = candidate_value(candidate_pool[idx1])
            x2 = candidate_value(candidate_pool[idx2])
            phase = "init"
        else:
            x1, x2 = agent.suggest_pair(comparisons, candidate_pool)
            phase = "bo"

        f1_true = oracle.f_at(x1)
        f2_true = oracle.f_at(x2)

        # compares the candidates using the oracle's latent function
        winner, loser = oracle.compare(x1, x2)
        comparisons.append((winner, loser))
        agent.observe_pair(x1, x2, f1_true, f2_true)

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

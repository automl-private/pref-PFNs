"""
Oracle and test functions for preference-based BO evaluation.

The oracle knows the true f(x) but only reveals pairwise comparisons.
All functions operate on x in [0, 1]^d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import gpytorch
import torch


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def gp_sample(
    n_grid: int = 500,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a 1D GP function on a fixed grid. Returns (x_grid, f_grid).
    x_grid: (n_grid,), f_grid: (n_grid,)
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    x = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=torch.double)

    base_kernel = gpytorch.kernels.RBFKernel()
    base_kernel.lengthscale = lengthscale
    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    covar_module.outputscale = outputscale

    mean = torch.zeros(n_grid, dtype=torch.double, device=device)
    cov = covar_module(x.unsqueeze(-1)).add_jitter(1e-6).evaluate()
    dist = torch.distributions.MultivariateNormal(mean, cov)

    with torch.no_grad():
        f = dist.sample()
        if noise_std > 0:
            f = f + noise_std * torch.randn(n_grid, generator=rng, dtype=torch.double, device=device)

    return x.float(), f.float()


def branin_1d(n_grid: int = 500, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D slice of Branin at x2=0.5, rescaled to [0,1].
    """
    x = torch.linspace(0.0, 1.0, n_grid, device=device)
    # Branin domain: x1 in [-5, 10]
    x1 = x * 15.0 - 5.0
    x2 = torch.full_like(x1, 7.5)  # midpoint of [0, 15]
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
    # flip sign (Branin is a minimization problem, we maximize)
    f = -f
    return x, f


def sinusoidal_1d(n_grid: int = 500, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Simple multimodal 1D function."""
    x = torch.linspace(0.0, 1.0, n_grid, device=device)
    f = torch.sin(6 * math.pi * x) + 0.5 * torch.sin(2 * math.pi * x) + 0.1 * x
    return x, f


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

@dataclass
class Oracle:
    """
    Wraps a discrete 1D function defined on a fixed candidate grid.

    comparisons are noisy: the oracle flips the answer with probability
    `noise_prob` (simulates human preference noise).
    """
    x_grid: torch.Tensor       # (n_grid,)
    f_grid: torch.Tensor       # (n_grid,) — true latent utility
    noise_prob: float = 0.0    # probability of flipping the comparison
    seed: int = 0
    _rng: torch.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = torch.Generator()
        self._rng.manual_seed(self.seed)
        assert self.x_grid.shape == self.f_grid.shape

    # ------------------------------------------------------------------
    # ground truth
    # ------------------------------------------------------------------

    @property
    def x_opt(self) -> float:
        return self.x_grid[self.f_grid.argmax()].item()

    @property
    def f_opt(self) -> float:
        return self.f_grid.max().item()

    def f_at(self, x: float) -> float:
        """Nearest-neighbour lookup of f(x) on the grid."""
        idx = (self.x_grid - x).abs().argmin()
        return self.f_grid[idx].item()

    def snap_to_grid(self, x: float) -> float:
        """Return the nearest grid point to x."""
        idx = (self.x_grid - x).abs().argmin()
        return self.x_grid[idx].item()

    # ------------------------------------------------------------------
    # comparison
    # ------------------------------------------------------------------

    def compare(self, x1: float, x2: float) -> tuple[float, float]:
        """
        Returns (winner, loser).  May flip with probability noise_prob.
        """
        f1, f2 = self.f_at(x1), self.f_at(x2)
        if f1 >= f2:
            winner, loser = x1, x2
        else:
            winner, loser = x2, x1

        if self.noise_prob > 0:
            flip = torch.rand(1, generator=self._rng).item() < self.noise_prob
            if flip:
                winner, loser = loser, winner

        return winner, loser

    def simple_regret(self, x_recommended: float) -> float:
        return self.f_opt - self.f_at(x_recommended)


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def make_benchmarks(
    n_gp_seeds: int = 5,
    n_grid: int = 500,
    noise_prob: float = 0.05,
    gp_kwargs: dict | None = None,
    device: str = "cpu",
) -> list[dict]:
    """
    Returns a list of benchmark dicts:
        {"name": str, "oracle": Oracle}
    """
    gp_kwargs = gp_kwargs or {}
    benchmarks = []

    for seed in range(n_gp_seeds):
        x, f = gp_sample(n_grid=n_grid, seed=seed, device=device, **gp_kwargs)
        benchmarks.append({
            "name": f"gp_seed{seed}",
            "oracle": Oracle(x_grid=x, f_grid=f, noise_prob=noise_prob, seed=seed),
        })

    for name, fn in [("branin_1d", branin_1d), ("sinusoidal_1d", sinusoidal_1d)]:
        x, f = fn(n_grid=n_grid, device=device)
        benchmarks.append({
            "name": name,
            "oracle": Oracle(x_grid=x, f_grid=f, noise_prob=noise_prob),
        })

    return benchmarks

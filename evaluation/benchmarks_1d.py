"""Deterministic one-dimensional benchmark functions for preference BO."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Mapping

import torch

from evaluation.grid_designs import GridDesign, make_1d_grid


BenchmarkFn = Callable[[int, str, GridDesign, int], tuple[torch.Tensor, torch.Tensor]]


def _unit_grid(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> torch.Tensor:
    return make_1d_grid(
        n_grid,
        design=grid_design,
        seed=grid_seed,
        device=device,
        dtype=torch.float32,
    )


def _to_domain(x_unit: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + x_unit * (upper - lower)


def forrester_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    f = (6.0 * x - 2.0).square() * torch.sin(12.0 * x - 4.0)
    return x, f


def gramacy_lee_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, 0.5, 2.5)
    f = torch.sin(10.0 * math.pi * z) / (2.0 * z) + (z - 1.0).pow(4)
    return x, f


def higdon_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, 0.0, 20.0)
    left = torch.sin(math.pi * z / 5.0)
    right = 0.2 * torch.cos(4.0 * math.pi * z / 5.0)
    f = torch.where(z < 10.0, left, right)
    return x, f


def ackley_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, -32.768, 32.768)
    objective = -20.0 * torch.exp(-0.2 * z.abs()) - torch.exp(torch.cos(2.0 * math.pi * z))
    objective = objective + 20.0 + math.e
    return x, -objective


def rastrigin_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, -5.12, 5.12)
    objective = 10.0 + z.square() - 10.0 * torch.cos(2.0 * math.pi * z)
    return x, -objective


def griewank_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, -600.0, 600.0)
    objective = 1.0 + z.square() / 4000.0 - torch.cos(z)
    return x, -objective


def schwefel_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, -500.0, 500.0)
    objective = 418.9829 - z * torch.sin(torch.sqrt(z.abs()))
    return x, -objective


def weierstrass_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    z = _to_domain(x, -0.5, 0.5)
    a = 0.5
    b = 3.0
    k_max = 20
    f = torch.zeros_like(z)
    for k in range(k_max + 1):
        f = f + (a**k) * torch.cos(2.0 * math.pi * (b**k) * (z + 0.5))
    constant = sum((a**k) * math.cos(math.pi * (b**k)) for k in range(k_max + 1))
    objective = f - constant
    return x, -objective


def branin_slice_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    x1 = _to_domain(x, -5.0, 10.0)
    x2 = torch.full_like(x1, 7.5)
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    objective = a * (x2 - b * x1.square() + c * x1 - r).square()
    objective = objective + s * (1.0 - t) * torch.cos(x1) + s
    return x, -objective


def sinusoidal_1d(
    n_grid: int,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _unit_grid(n_grid, device=device, grid_design=grid_design, grid_seed=grid_seed)
    f = torch.sin(6.0 * math.pi * x) + 0.5 * torch.sin(2.0 * math.pi * x) + 0.1 * x
    return x, f


BENCHMARKS_1D: Mapping[str, BenchmarkFn] = OrderedDict(
    [
        ("forrester_1d", forrester_1d),
        ("gramacy_lee_1d", gramacy_lee_1d),
        ("higdon_1d", higdon_1d),
        ("ackley_1d", ackley_1d),
        ("rastrigin_1d", rastrigin_1d),
        ("griewank_1d", griewank_1d),
        ("schwefel_1d", schwefel_1d),
        ("weierstrass_1d", weierstrass_1d),
        ("branin_slice_1d", branin_slice_1d),
        ("sinusoidal_1d", sinusoidal_1d),
    ]
)


def normalize_f_grid(f_grid: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return f_grid.float()
    if mode == "std1":
        f = f_grid.float()
        std = f.std(unbiased=False).clamp_min(1e-12)
        return (f - f.mean()) / std
    raise ValueError(f"Unknown deterministic normalization mode: {mode!r}")


def make_benchmark_1d(
    name: str,
    *,
    n_grid: int,
    normalization: str,
    device: str = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        fn = BENCHMARKS_1D[name]
    except KeyError as err:
        available = ", ".join(BENCHMARKS_1D)
        raise ValueError(f"Unknown deterministic benchmark {name!r}. Available: {available}") from err
    x_grid, f_grid = fn(n_grid, device, grid_design, grid_seed)
    return x_grid.float(), normalize_f_grid(f_grid, normalization)

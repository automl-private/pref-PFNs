"""Deterministic benchmark functions for preference BO."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Mapping

import torch
from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Hartmann,
    Levy,
    Powell,
    Rastrigin,
    Rosenbrock,
)

from evaluation.grid_designs import GridDesign, make_unit_grid


BenchmarkFn = Callable[
    [int, str | torch.device, GridDesign, int],
    tuple[torch.Tensor, torch.Tensor],
]


def forrester_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    f = (6.0 * x - 2.0).square() * torch.sin(12.0 * x - 4.0)
    return x, f


def gramacy_lee_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = 0.5 + x * 2.0
    f = torch.sin(10.0 * math.pi * z) / (2.0 * z) + (z - 1.0).pow(4)
    return x, f


def higdon_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = x * 20.0
    left = torch.sin(math.pi * z / 5.0)
    right = 0.2 * torch.cos(4.0 * math.pi * z / 5.0)
    f = torch.where(z < 10.0, left, right)
    return x, f


def ackley_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = -32.768 + x * 65.536
    objective = -20.0 * torch.exp(-0.2 * z.abs()) - torch.exp(torch.cos(2.0 * math.pi * z))
    objective = objective + 20.0 + math.e
    return x, -objective


def rastrigin_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = -5.12 + x * 10.24
    objective = 10.0 + z.square() - 10.0 * torch.cos(2.0 * math.pi * z)
    return x, -objective


def griewank_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = -600.0 + x * 1200.0
    objective = 1.0 + z.square() / 4000.0 - torch.cos(z)
    return x, -objective


def schwefel_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = -500.0 + x * 1000.0
    objective = 418.9829 - z * torch.sin(torch.sqrt(z.abs()))
    return x, -objective


def weierstrass_1d(
    n_grid: int,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    z = -0.5 + x
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
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
    x1 = -5.0 + x * 15.0
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
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_unit_grid(n_grid, 1, design=grid_design, seed=grid_seed, device=device)
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

BOTORCH_BENCHMARKS_BY_DIM: Mapping[int, Mapping[str, Callable[[], object]]] = {
    2: OrderedDict(
        [
            ("branin_2d", lambda: Branin(negate=True)),
            ("beale_2d", lambda: Beale(negate=True)),
            ("ackley_2d", lambda: Ackley(dim=2, negate=True)),
            ("rosenbrock_2d", lambda: Rosenbrock(dim=2, negate=True)),
            ("levy_2d", lambda: Levy(dim=2, negate=True)),
        ]
    ),
    4: OrderedDict(
        [
            ("hartmann_4d", lambda: Hartmann(dim=4, negate=True)),
            ("ackley_4d", lambda: Ackley(dim=4, negate=True)),
            ("rosenbrock_4d", lambda: Rosenbrock(dim=4, negate=True)),
            ("levy_4d", lambda: Levy(dim=4, negate=True)),
            ("powell_4d", lambda: Powell(dim=4, negate=True)),
        ]
    ),
    6: OrderedDict(
        [
            ("hartmann_6d", lambda: Hartmann(dim=6, negate=True)),
            ("ackley_6d", lambda: Ackley(dim=6, negate=True)),
            ("rastrigin_6d", lambda: Rastrigin(dim=6, negate=True)),
            ("levy_6d", lambda: Levy(dim=6, negate=True)),
            ("rosenbrock_6d", lambda: Rosenbrock(dim=6, negate=True)),
        ]
    ),
}

BENCHMARK_NAMES_BY_DIM: Mapping[int, tuple[str, ...]] = {
    1: tuple(BENCHMARKS_1D.keys()),
    **{
        dim: tuple(benchmarks.keys())
        for dim, benchmarks in BOTORCH_BENCHMARKS_BY_DIM.items()
    },
}

DEFAULT_DETERMINISTIC_BENCHMARKS_BY_DIM: Mapping[int, tuple[str, ...]] = BENCHMARK_NAMES_BY_DIM


def normalize_f_grid(f_grid: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return f_grid.float()
    if mode == "std1":
        f = f_grid.float()
        std = f.std(unbiased=False).clamp_min(1e-12)
        return (f - f.mean()) / std
    raise ValueError(f"Unknown deterministic normalization mode: {mode!r}")


def make_benchmark(
    name: str,
    *,
    input_dim: int,
    n_grid: int,
    normalization: str,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dim = int(input_dim)

    if input_dim == 1:
        try:
            fn = BENCHMARKS_1D[name]
        except KeyError as err:
            available = ", ".join(BENCHMARKS_1D)
            raise ValueError(
                f"Unknown 1D deterministic benchmark {name!r}. Available: {available}"
            ) from err
        x_grid, f_grid = fn(n_grid, device, grid_design, grid_seed)
        return x_grid.float(), normalize_f_grid(f_grid, normalization)

    benchmarks = BOTORCH_BENCHMARKS_BY_DIM.get(input_dim)
    if benchmarks is None:
        available_dims = ", ".join(str(dim) for dim in BENCHMARK_NAMES_BY_DIM)
        raise ValueError(
            f"Deterministic benchmarks are not defined for input_dim={input_dim}. "
            f"Available dimensions: {available_dims}."
        )

    try:
        function = benchmarks[name]().to(device=device, dtype=torch.float32)
    except KeyError as err:
        available = ", ".join(benchmarks)
        raise ValueError(
            f"Unknown {input_dim}D deterministic benchmark {name!r}. "
            f"Available: {available}"
        ) from err

    x_grid = make_unit_grid(
        n_grid,
        input_dim,
        design=grid_design,
        seed=grid_seed,
        device=device,
    )
    bounds = function.bounds.to(device=x_grid.device, dtype=x_grid.dtype)
    x_domain = bounds[0] + x_grid * (bounds[1] - bounds[0])

    with torch.no_grad():
        f_grid = function(x_domain).reshape(-1)
    return x_grid.float(), normalize_f_grid(f_grid, normalization)


def make_benchmark_1d(
    name: str,
    *,
    n_grid: int,
    normalization: str,
    device: str | torch.device = "cpu",
    grid_design: GridDesign = "uniform",
    grid_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return make_benchmark(
        name,
        input_dim=1,
        n_grid=n_grid,
        normalization=normalization,
        device=device,
        grid_design=grid_design,
        grid_seed=grid_seed,
    )

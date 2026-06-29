"""Deterministic benchmark functions for preference BO."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from botorch.test_functions.synthetic import (
    Ackley,
    Beale,
    Branin,
    Griewank,
    Hartmann,
    Levy,
    Powell,
    Rastrigin,
    Rosenbrock,
)

from evaluation.grid_designs import GridDesign, make_unit_grid


@dataclass(frozen=True)
class ContinuousDeterministicBenchmark:
    name: str
    input_dim: int
    norm_mean: float
    norm_std: float
    x_opt: object
    f_opt: float
    support: str = "continuous_rff"

    def evaluate(self, x, *, batch_size: int = 2048) -> torch.Tensor:
        x_tensor = torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.input_dim)
        values = []
        for start in range(0, x_tensor.shape[0], batch_size):
            chunk = x_tensor[start : start + batch_size]
            raw = evaluate_raw_benchmark(self.name, self.input_dim, chunk)
            values.append((raw - self.norm_mean) / self.norm_std)
        return torch.cat(values)


def evaluate_raw_benchmark(
    name: str,
    input_dim: int,
    x_unit: torch.Tensor,
) -> torch.Tensor:
    benchmarks = BOTORCH_BENCHMARKS_BY_DIM.get(input_dim, {})
    if name in benchmarks:
        x = torch.as_tensor(x_unit, dtype=torch.float32).reshape(-1, input_dim)
        function = benchmarks[name]().to(device=x.device, dtype=x.dtype)
        bounds = function.bounds.to(device=x.device, dtype=x.dtype)
        x_domain = bounds[0] + x * (bounds[1] - bounds[0])
        with torch.no_grad():
            return function(x_domain).reshape(-1)

    if input_dim != 1:
        raise ValueError(f"Unknown {input_dim}D deterministic benchmark {name!r}.")

    x = torch.as_tensor(x_unit, dtype=torch.float32).reshape(-1)
    if name == "forrester_1d":
        return (6.0 * x - 2.0).square() * torch.sin(12.0 * x - 4.0)
    if name == "gramacy_lee_1d":
        z = 0.5 + x * 2.0
        return torch.sin(10.0 * math.pi * z) / (2.0 * z) + (z - 1.0).pow(4)
    if name == "higdon_1d":
        z = x * 20.0
        left = torch.sin(math.pi * z / 5.0)
        right = 0.2 * torch.cos(4.0 * math.pi * z / 5.0)
        return torch.where(z < 10.0, left, right)
    if name == "schwefel_1d":
        z = -500.0 + x * 1000.0
        return -(418.9829 - z * torch.sin(torch.sqrt(z.abs())))
    if name == "weierstrass_1d":
        z = -0.5 + x
        a = 0.5
        b = 3.0
        k_max = 20
        value = torch.zeros_like(z)
        for k in range(k_max + 1):
            value = value + (a**k) * torch.cos(2.0 * math.pi * (b**k) * (z + 0.5))
        constant = sum((a**k) * math.cos(math.pi * (b**k)) for k in range(k_max + 1))
        return -(value - constant)
    if name == "branin_slice_1d":
        function = Branin(negate=True)
        x1 = -5.0 + x * 15.0
        x2 = torch.full_like(x1, 7.5)
        x_domain = torch.stack([x1, x2], dim=-1).reshape(-1, 2)
        with torch.no_grad():
            return function.to(device=x.device, dtype=x.dtype)(x_domain).reshape(-1)
    if name == "sinusoidal_1d":
        return (
            torch.sin(6.0 * math.pi * x)
            + 0.5 * torch.sin(2.0 * math.pi * x)
            + 0.1 * x
        )
    raise ValueError(f"Unknown 1D deterministic benchmark {name!r}.")


BENCHMARKS_1D = (
    "forrester_1d",
    "gramacy_lee_1d",
    "higdon_1d",
    "ackley_1d",
    "rastrigin_1d",
    "griewank_1d",
    "schwefel_1d",
    "weierstrass_1d",
    "branin_slice_1d",
    "sinusoidal_1d",
)

BOTORCH_BENCHMARKS_BY_DIM: Mapping[int, Mapping[str, Callable[[], object]]] = {
    1: OrderedDict(
        [
            ("ackley_1d", lambda: Ackley(dim=1, negate=True)),
            ("rastrigin_1d", lambda: Rastrigin(dim=1, negate=True)),
            ("griewank_1d", lambda: Griewank(dim=1, negate=True)),
        ]
    ),
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
    1: BENCHMARKS_1D,
    **{
        dim: tuple(benchmarks.keys())
        for dim, benchmarks in BOTORCH_BENCHMARKS_BY_DIM.items()
        if dim != 1
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
    names = BENCHMARK_NAMES_BY_DIM.get(input_dim)
    if names is None:
        available_dims = ", ".join(str(dim) for dim in BENCHMARK_NAMES_BY_DIM)
        raise ValueError(
            f"Deterministic benchmarks are not defined for input_dim={input_dim}. "
            f"Available dimensions: {available_dims}."
        )
    if name not in names:
        available = ", ".join(names)
        raise ValueError(
            f"Unknown {input_dim}D deterministic benchmark {name!r}. "
            f"Available: {available}"
        )

    x_grid = make_unit_grid(
        n_grid,
        input_dim,
        design=grid_design,
        seed=grid_seed,
        device=device,
    )
    f_grid = evaluate_raw_benchmark(name, input_dim, x_grid)
    return x_grid.float(), normalize_f_grid(f_grid, normalization)


def make_continuous_benchmark(
    name: str,
    *,
    input_dim: int,
    normalization: str,
    reference_size: int,
    reference_seed: int,
    grid_design: GridDesign = "lhs",
) -> ContinuousDeterministicBenchmark:
    input_dim = int(input_dim)
    reference_x = make_unit_grid(
        reference_size,
        input_dim,
        design=grid_design,
        seed=reference_seed,
    )
    raw_reference = evaluate_raw_benchmark(name, input_dim, reference_x)

    if normalization == "raw":
        norm_mean = 0.0
        norm_std = 1.0
    elif normalization == "std1":
        norm_mean = float(raw_reference.mean().item())
        norm_std = float(raw_reference.std(unbiased=False).clamp_min(1e-12).item())
    else:
        raise ValueError(f"Unknown deterministic normalization mode: {normalization!r}")

    reference_values = (raw_reference - norm_mean) / norm_std
    best_idx = int(reference_values.argmax().item())
    best_point = reference_x.reshape(-1, input_dim)[best_idx]
    if input_dim == 1:
        x_opt = float(best_point.reshape(-1)[0].item())
    else:
        x_opt = tuple(float(v) for v in best_point.reshape(-1).tolist())
    f_opt = float(reference_values[best_idx].item())

    benchmarks = BOTORCH_BENCHMARKS_BY_DIM.get(input_dim, {})
    if name in benchmarks:
        function = benchmarks[name]()
        exact_optimal_value = getattr(function, "_optimal_value", None)
        if exact_optimal_value is not None:
            exact_optimal_value = float(exact_optimal_value)
            if getattr(function, "negate", False):
                exact_optimal_value = -exact_optimal_value
            f_opt = float((exact_optimal_value - norm_mean) / norm_std)

    return ContinuousDeterministicBenchmark(
        name=name,
        input_dim=input_dim,
        norm_mean=norm_mean,
        norm_std=norm_std,
        x_opt=x_opt,
        f_opt=f_opt,
    )


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

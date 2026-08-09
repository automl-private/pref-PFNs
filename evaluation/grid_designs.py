"""Candidate-grid construction utilities for BO benchmarks."""

from __future__ import annotations

from typing import Literal

import torch


GridDesign = Literal["uniform", "lhs"]


def make_unit_grid(
    n_grid: int,
    input_dim: int = 1,
    *,
    design: GridDesign = "uniform",
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create candidate points on [0, 1]^d."""
    input_dim = int(input_dim)

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    if input_dim == 1 and design == "uniform":
        return torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)

    if input_dim == 1 and design == "lhs":
        bins = torch.arange(n_grid, dtype=torch.float64)
        offsets = torch.rand(n_grid, generator=generator, dtype=torch.float64)
        x = ((bins + offsets) / float(n_grid)).sort().values
        return x.to(device=device, dtype=dtype)

    if design == "lhs":
        base = torch.arange(n_grid, dtype=torch.float64).unsqueeze(-1)
        offsets = torch.rand(n_grid, input_dim, generator=generator, dtype=torch.float64)
        x = (base + offsets) / float(n_grid)
        for dim in range(input_dim):
            perm = torch.randperm(n_grid, generator=generator)
            x[:, dim] = x[perm, dim]
        return x.to(device=device, dtype=dtype)

    if design == "uniform":
        return torch.rand(
            n_grid,
            input_dim,
            generator=generator,
            dtype=dtype,
        ).to(device=device)

    raise ValueError(f"Unknown grid design {design!r}. Expected 'uniform' or 'lhs'.")

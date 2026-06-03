"""Candidate-grid construction utilities for one-dimensional BO benchmarks."""

from __future__ import annotations

from typing import Literal

import torch


GridDesign = Literal["uniform", "lhs"]


def make_1d_grid(
    n_grid: int,
    *,
    design: GridDesign = "uniform",
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a one-dimensional candidate grid on [0, 1].

    Args:
        n_grid: Number of candidate points.
        design: Grid design. ``"uniform"`` preserves the previous behavior and
            returns evenly spaced points including both endpoints. ``"lhs"``
            returns a sorted Latin-hypercube design with one point per interval.
        seed: Random seed used only for ``design="lhs"``.
        device: Output tensor device.
        dtype: Output tensor dtype.

    Returns:
        Tensor of shape ``(n_grid,)`` with values in ``[0, 1]``.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    if design == "uniform":
        return torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)

    if design == "lhs":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        bins = torch.arange(n_grid, dtype=torch.float64)
        offsets = torch.rand(n_grid, generator=generator, dtype=torch.float64)
        x = ((bins + offsets) / float(n_grid)).sort().values
        return x.to(device=device, dtype=dtype)

    raise ValueError(f"Unknown grid design {design!r}. Expected 'uniform' or 'lhs'.")

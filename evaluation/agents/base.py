"""Abstract base class for all PBO agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor

Point = Union[float, tuple[float, ...]]
Comparison = tuple[Point, Point]


def candidate_value(candidate) -> Point:
    """Convert a candidate tensor row to a scalar float or multidim point tuple."""
    candidate = torch.as_tensor(candidate)
    if candidate.ndim == 0 or candidate.numel() == 1:
        return float(candidate.reshape(-1)[0].item())
    return tuple(float(v) for v in candidate.reshape(-1).detach().cpu().tolist())


def candidate_tensor(candidate, *, dtype: torch.dtype, device=None) -> Tensor:
    """Convert a scalar or multidim point into a tensor with the requested dtype/device."""
    return torch.as_tensor(candidate, dtype=dtype, device=device)


def candidate_matrix(candidate_pool: Tensor, *, dtype: torch.dtype | None = None, device=None) -> Tensor:
    """Return candidates as shape (M, d), preserving M for both 1D and multidim pools."""
    x = candidate_pool if dtype is None and device is None else candidate_pool.to(dtype=dtype, device=device)
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x.reshape(x.shape[0], -1)


def nearest_candidate_index(candidate_pool: Tensor, candidate) -> int:
    """Find the nearest grid index to a scalar or multidim candidate."""
    pool = candidate_matrix(candidate_pool)
    point = candidate_tensor(candidate, dtype=pool.dtype, device=pool.device).reshape(-1)
    if point.numel() == 1 and pool.shape[1] == 1:
        return int((pool[:, 0] - point[0]).abs().argmin().item())
    distances = torch.linalg.norm(pool - point, dim=-1)
    return int(distances.argmin().item())


class PBOAgent(ABC):
    """
    Interface every agent must implement.

    comparisons: list of (winner_x, loser_x) observed so far
    candidate_pool: tensor of candidate point values to choose from
    """

    @abstractmethod
    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Point, Point]:
        """
        Return the next pair (x1, x2) to compare.
        The oracle will decide which one wins.
        """

    @abstractmethod
    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> Point:
        """
        Return the current best x estimate.
        """

    def reset(self):
        """Optional: clear any cached state between runs."""

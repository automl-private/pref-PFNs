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


def candidate_matrix(candidate_pool: Tensor, *, dtype: torch.dtype | None = None, device=None) -> Tensor:
    """Return candidates as shape (M, d), preserving M for both 1D and multidim pools."""
    x = candidate_pool if dtype is None and device is None else candidate_pool.to(dtype=dtype, device=device)
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x.reshape(x.shape[0], -1)

def _build_pairwise_tensors(
    comparisons: list[Comparison],
    dtype=torch.float64,
    device=None,
) -> tuple[Tensor, Tensor]:
    """
    Преобразует историю сравнений в формат `PairwiseGP`.

    На входе список пар `(winner_x, loser_x)`, где точки могут быть скалярами
    или многомерными tuple/list/tensor. Функция собирает все уникальные точки
    в порядке первого появления и кодирует каждое сравнение индексами этих
    точек.

    Возвращает:
        datapoints: shape `(n_unique, d)`, все уникальные наблюденные точки.
        comp_idx: shape `(m, 2)`, индексы `[winner_idx, loser_idx]`.
    """
    def key(point):
        """Делает из точки hashable tuple-ключ с фиксированным dtype."""
        tensor = torch.as_tensor(point, dtype=dtype, device=device).reshape(-1)
        return tuple(float(v) for v in tensor.detach().cpu().tolist())

    seen: dict[tuple[float, ...], int] = {}
    for w, l in comparisons:
        wk = key(w)
        lk = key(l)
        if wk not in seen:
            seen[wk] = len(seen)
        if lk not in seen:
            seen[lk] = len(seen)

    datapoints = torch.tensor(
        sorted(seen.keys(), key=lambda x: seen[x]),
        dtype=dtype,
        device=device,
    )  # (n, d)

    comp_idx = torch.tensor(
        [[seen[key(w)], seen[key(l)]] for w, l in comparisons],
        dtype=torch.long,
        device=device,
    )  # (m, 2)

    return datapoints, comp_idx


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

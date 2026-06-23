"""
Random baseline agent.

suggest_pair: samples two random points from the continuous domain [0, 1]^d.
recommend:    returns the best observed point by true utility, matching PABBO rs.
"""

from __future__ import annotations

import random
import torch

from .base import PBOAgent, Comparison, Point


class RandomAgent(PBOAgent):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._best_point: Point | None = None
        self._best_value = -float("inf")

    def reset(self):
        """Clears the best-observed recommendation between independent BO runs."""
        self._best_point = None
        self._best_value = -float("inf")

    def _random_point(self, candidate_pool: torch.Tensor) -> Point:
        """Samples one uniform point from [0, 1]^d using candidate_pool only for d."""
        candidates = torch.as_tensor(candidate_pool)
        if candidates.ndim == 1:
            input_dim = 1
        else:
            input_dim = candidates.reshape(candidates.shape[0], -1).shape[-1]
        if input_dim == 1:
            return float(self._rng.random())
        return tuple(float(self._rng.random()) for _ in range(int(input_dim)))

    def observe_pair(self, x1: Point, x2: Point, f1: float, f2: float) -> None:
        """Stores the true best-observed point among all random queries."""
        for point, value in ((x1, f1), (x2, f2)):
            value = float(value)
            if self._best_point is None or value > self._best_value:
                self._best_point = point
                self._best_value = value

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        """Samples two independent random points from [0, 1]^d."""
        return self._random_point(candidate_pool), self._random_point(candidate_pool)

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        """Returns the best observed true-utility point, or a random fallback."""
        if self._best_point is not None:
            return self._best_point
        return self._random_point(candidate_pool)

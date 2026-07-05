"""Preference-only random incumbent-duel baseline."""

from __future__ import annotations

import random
import torch

from .base import PBOAgent, Comparison, Point, candidate_value


class RandomAgent(PBOAgent):
    """Random challenger baseline with preference-only incumbent recommendation."""

    def __init__(self, seed: int = 0, support: str = "grid"):
        self._rng = random.Random(seed)
        self.support = support

    def reset(self):
        """Keeps the agent stateless between independent BO runs."""

    def _continuous_random_point(self, candidate_pool: torch.Tensor) -> Point:
        """Samples one uniform point from [0, 1]^d using candidate_pool only for d."""
        candidates = torch.as_tensor(candidate_pool)
        if candidates.ndim == 1:
            input_dim = 1
        else:
            input_dim = candidates.reshape(candidates.shape[0], -1).shape[-1]
        if input_dim == 1:
            return float(self._rng.random())
        return tuple(float(self._rng.random()) for _ in range(int(input_dim)))

    def _random_point(self, candidate_pool: torch.Tensor) -> Point:
        """Samples from the finite grid or from the continuous domain."""
        if self.support == "grid":
            idx = self._rng.randrange(len(candidate_pool))
            return candidate_value(candidate_pool[idx])
        if self.support == "continuous_rff":
            return self._continuous_random_point(candidate_pool)
        raise ValueError(f"Unknown RandomAgent support {self.support!r}.")

    def _random_pair(self, candidate_pool: torch.Tensor) -> tuple[Point, Point]:
        """Samples the initial random duel before an incumbent exists."""
        if self.support == "grid":
            idx1, idx2 = self._rng.sample(range(len(candidate_pool)), 2)
            return candidate_value(candidate_pool[idx1]), candidate_value(candidate_pool[idx2])
        return self._random_point(candidate_pool), self._random_point(candidate_pool)

    def _random_challenger(self, candidate_pool: torch.Tensor, incumbent: Point) -> Point:
        """Samples a random challenger, avoiding the incumbent on finite grids."""
        if self.support != "grid":
            return self._random_point(candidate_pool)

        indices = list(range(len(candidate_pool)))
        self._rng.shuffle(indices)
        for idx in indices:
            challenger = candidate_value(candidate_pool[idx])
            if challenger != incumbent:
                return challenger
        return candidate_value(candidate_pool[indices[0]])

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        """Duels the current incumbent against a random challenger."""
        if not comparisons:
            return self._random_pair(candidate_pool)

        incumbent = comparisons[-1][0]
        challenger = self._random_challenger(candidate_pool, incumbent)
        return incumbent, challenger

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        """Returns the current preference-only incumbent."""
        if comparisons:
            return comparisons[-1][0]
        return self._random_point(candidate_pool)

"""
Random baseline agent.

suggest_pair: picks two random candidates.
recommend:    returns the candidate that has won the most comparisons
              (falls back to random if no comparisons yet).
"""

from __future__ import annotations

import random
from collections import Counter

import torch

from .base import PBOAgent, Comparison


class RandomAgent(PBOAgent):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[float, float]:
        pool = candidate_pool.tolist()
        x1, x2 = self._rng.sample(pool, 2)
        return x1, x2

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        if not comparisons:
            return self._rng.choice(candidate_pool.tolist())
        wins = Counter(w for w, _ in comparisons)
        return max(wins, key=wins.get)

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

from .base import PBOAgent, Comparison, candidate_value


class RandomAgent(PBOAgent):
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple:
        idx1, idx2 = self._rng.sample(range(len(candidate_pool)), 2)
        return candidate_value(candidate_pool[idx1]), candidate_value(candidate_pool[idx2])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ):
        if not comparisons:
            idx = self._rng.randrange(len(candidate_pool))
            return candidate_value(candidate_pool[idx])
        wins = Counter(w for w, _ in comparisons)
        return max(wins, key=wins.get)

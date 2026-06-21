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

from .base import PBOAgent, Comparison, Point, candidate_value


class RandomAgent(PBOAgent):
    def __init__(self, seed: int = 0, support: str = "grid"):
        self._rng = random.Random(seed)
        self.support = support

    @staticmethod
    # Функция принимает всю историю сравнений
    def _incumbent_from_history(comparisons: list[Comparison]):
        # (winner, loser)
        # Пока истории нет или мы ещё её не прочитал
        incumbent = None # лучшая текущая точка по последовательной логике дуэлей
        # Идём по истории сравнений в хронологическом порядке
        for winner, loser in comparisons:
            if incumbent is None:
                # Если это первое обработанное сравнение,
                # то текущим incumbent становится победитель первого сравнения
                incumbent = winner
            # Если старый incumbent проиграл, заменяем его на победителя текущей дуэли
            elif loser == incumbent:
                incumbent = winner
        return incumbent

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        if self.support == "grid":
            idx1, idx2 = self._rng.sample(range(len(candidate_pool)), 2)
            return candidate_value(candidate_pool[idx1]), candidate_value(candidate_pool[idx2])

        if self.support == "continuous_rff":
            incumbent = self._incumbent_from_history(comparisons)
            if incumbent is None:
                idx1, idx2 = self._rng.sample(range(len(candidate_pool)), 2)
                return candidate_value(candidate_pool[idx1]), candidate_value(candidate_pool[idx2])

            idx = self._rng.randrange(len(candidate_pool))
            challenger = candidate_value(candidate_pool[idx])
            return incumbent, challenger

        raise ValueError(f"Unknown RandomAgent support {self.support!r}.")
    
    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        if self.support == "grid":
            if not comparisons:
                idx = self._rng.randrange(len(candidate_pool))
                return candidate_value(candidate_pool[idx])
            wins = Counter(w for w, _ in comparisons)
            return max(wins, key=wins.get)

        if self.support == "continuous_rff":
            incumbent = self._incumbent_from_history(comparisons)
            if incumbent is not None:
                return incumbent

            idx = self._rng.randrange(len(candidate_pool))
            return candidate_value(candidate_pool[idx])

        raise ValueError(f"Unknown RandomAgent support {self.support!r}.")

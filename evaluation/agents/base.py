"""Abstract base class for all PBO agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

# A comparison is (winner_x, loser_x), both floats
Comparison = tuple[float, float]


class PBOAgent(ABC):
    """
    Interface every agent must implement.

    comparisons: list of (winner_x, loser_x) observed so far
    candidate_pool: 1-D tensor of candidate x values to choose from
    """

    @abstractmethod
    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool,           # torch.Tensor shape (n,)
    ) -> tuple[float, float]:
        """
        Return the next pair (x1, x2) to compare.
        The oracle will decide which one wins.
        """

    @abstractmethod
    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool,           # torch.Tensor shape (n,)
    ) -> float:
        """
        Return the current best x estimate.
        """

    def reset(self):
        """Optional: clear any cached state between runs."""

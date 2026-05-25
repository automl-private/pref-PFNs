"""
PFN-based preference BO agent.

Uses the trained preference PFN model to:
  - recommend: argmax of mean posterior E[f(x)] over candidate pool
  - suggest_pair: Thompson Sampling — sample two f draws, compare argmax
"""

from __future__ import annotations

import torch

from .base import PBOAgent, Comparison


def _build_context(
    comparisons: list[Comparison],
    dtype=torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack comparisons into (x_ctx, y_ctx) tensors expected by the PFN.

    x_ctx: (1, n_ctx, 2)  — each row is [winner_x, loser_x]
    y_ctx: (1, n_ctx)     — zeros (targets are unused at context positions)
    """
    if not comparisons:
        x_ctx = torch.zeros(1, 0, 2, dtype=dtype, device=device)
        y_ctx = torch.zeros(1, 0, dtype=dtype, device=device)
    else:
        pairs = torch.tensor(comparisons, dtype=dtype, device=device)  # (n, 2)
        x_ctx = pairs.unsqueeze(0)                                      # (1, n, 2)
        y_ctx = torch.zeros(1, len(comparisons), dtype=dtype, device=device)
    return x_ctx, y_ctx


def _query_logits(
    model,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """
    Run model forward pass.

    x_query: (n_query,) — 1D candidate positions
    Returns logits: (n_query, n_bars)
    """
    # Model expects test_x: (1, n_query, 2), second feature zero-padded
    n = x_query.shape[0]
    test_x = torch.zeros(1, n, 2, dtype=x_ctx.dtype, device=x_ctx.device)
    test_x[0, :, 0] = x_query

    with torch.no_grad():
        logits = model(x_ctx, y_ctx, test_x=test_x)  # (1, n_query, n_bars)

    return logits[0]  # (n_query, n_bars)


class PFNAgent(PBOAgent):
    """
    Args:
        model:        Trained transformer model (with model.criterion = BarDistribution).
        n_ts_samples: Number of Thompson Sampling draws for suggest_pair.
        device:       torch device string.
    """

    def __init__(
        self,
        model,
        n_ts_samples: int = 2,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.n_ts_samples = n_ts_samples
        self.device = device
        self.criterion = model.criterion

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    def _posterior_mean(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        """Returns E[f(x)] for each x in candidate_pool. Shape: (n,)"""
        x_ctx, y_ctx = _build_context(comparisons, dtype=self._dtype, device=self.device)
        x_query = candidate_pool.to(dtype=x_ctx.dtype, device=self.device)
        logits = _query_logits(self.model, x_ctx, y_ctx, x_query)
        return self.criterion.mean(logits)  # (n,)

    def _thompson_sample(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        """
        Draw one sample f ~ posterior, return argmax x.
        Uses BarDistribution.sample() which draws from the piecewise-constant dist.
        """
        x_ctx, y_ctx = _build_context(comparisons, dtype=self._dtype, device=self.device)
        x_query = candidate_pool.to(dtype=self._dtype, device=self.device)
        logits = _query_logits(self.model, x_ctx, y_ctx, x_query)  # (n, n_bars)

        # sample one f value per query point
        f_sample = self.criterion.sample(logits)  # (n,)
        best_idx = f_sample.argmax()
        return candidate_pool[best_idx].item()

    # ------------------------------------------------------------------

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Thompson Sampling acquisition:
        draw n_ts_samples independent f samples, take the top-2 argmaxes.
        """
        argmaxes = [
            self._thompson_sample(comparisons, candidate_pool)
            for _ in range(self.n_ts_samples)
        ]
        if len(set(argmaxes)) == 1:
            # both samples agree — add a random challenger
            pool = candidate_pool.tolist()
            challenger = pool[torch.randint(len(pool), (1,)).item()]
            return argmaxes[0], challenger

        return argmaxes[0], argmaxes[1]

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        if not comparisons:
            return candidate_pool[candidate_pool.shape[0] // 2].item()
        mean = self._posterior_mean(comparisons, candidate_pool)
        return candidate_pool[mean.argmax()].item()

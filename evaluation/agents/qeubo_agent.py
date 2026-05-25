"""
qEUBO agent (Astudillo & Frazier 2023).

Model:   PairwiseGP with RBF kernel + probit likelihood (botorch)
Acquire: qExpectedUtilityOfBestOption optimized over discrete candidate pool
Recommend: argmax of posterior mean over candidate pool

Requires: pip install botorch
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import PBOAgent, Comparison

try:
    from botorch.models.pairwise_gp import (
        PairwiseGP,
        PairwiseLaplaceMarginalLogLikelihood,
    )
    from botorch.acquisition.preference import qExpectedUtilityOfBestOption
    from botorch.optim import optimize_acqf_discrete
    from botorch.fit import fit_gpytorch_mll

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


def _require_botorch():
    if not _BOTORCH_AVAILABLE:
        raise ImportError(
            "botorch is required for qEUBO agent.\n"
            "Install with:  pip install botorch"
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_pairwise_tensors(
    comparisons: list[Comparison],
    dtype=torch.float64,
) -> tuple[Tensor, Tensor]:
    """
    Convert list of (winner_x, loser_x) into botorch PairwiseGP format.

    Returns:
        datapoints:  (n_unique, 1) — all unique x values seen
        comparisons: (m, 2) LongTensor — [winner_idx, loser_idx] into datapoints
    """
    # Collect unique x values preserving insertion order
    seen: dict[float, int] = {}
    for w, l in comparisons:
        if w not in seen:
            seen[w] = len(seen)
        if l not in seen:
            seen[l] = len(seen)

    datapoints = torch.tensor(
        sorted(seen.keys(), key=lambda x: seen[x]),
        dtype=dtype,
    ).unsqueeze(-1)  # (n, 1)

    comp_idx = torch.tensor(
        [[seen[w], seen[l]] for w, l in comparisons],
        dtype=torch.long,
    )  # (m, 2)

    return datapoints, comp_idx


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QEUBOAgent(PBOAgent):
    """
    Args:
        fit_hyperparams:  Whether to optimize GP kernel hyperparameters each step.
                          Slower but more adaptive.
        max_fit_iter:     Max iterations for hyperparameter fitting.
        num_acqf_samples: MC samples for qEUBO (higher = more accurate, slower).
        dtype:            torch dtype for all tensors (float64 recommended for GP).
    """

    def __init__(
        self,
        fit_hyperparams: bool = True,
        max_fit_iter: int = 100,
        num_acqf_samples: int = 512,
        dtype=torch.float64,
    ):
        _require_botorch()
        self.fit_hyperparams = fit_hyperparams
        self.max_fit_iter = max_fit_iter
        self.num_acqf_samples = num_acqf_samples
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_model(self, comparisons: list[Comparison]) -> "PairwiseGP":
        datapoints, comp_idx = _build_pairwise_tensors(comparisons, dtype=self.dtype)

        model = PairwiseGP(datapoints, comp_idx)
        model.train()

        if self.fit_hyperparams:
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, max_attempts=1, options={"maxiter": self.max_fit_iter})

        model.eval()
        return model

    def _posterior_mean(self, model: "PairwiseGP", candidate_pool: Tensor) -> Tensor:
        """
        Returns posterior mean f(x) for each x in candidate_pool. Shape: (n,)
        candidate_pool: (n,) 1-D tensor
        """
        X = candidate_pool.to(dtype=self.dtype).unsqueeze(-1)  # (n, 1)
        with torch.no_grad():
            posterior = model.posterior(X)
        return posterior.mean.squeeze(-1).squeeze(-1)  # (n,)

    # ------------------------------------------------------------------
    # PBOAgent interface
    # ------------------------------------------------------------------

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[float, float]:
        if not comparisons:
            # No data yet — return a random pair
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_pool[idx[0]].item(), candidate_pool[idx[1]].item()

        model = self._fit_model(comparisons)

        from botorch.sampling.normal import SobolQMCNormalSampler
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))
        acqf = qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

        # choices: (n_candidates, 1) for 1-D input
        choices = candidate_pool.to(dtype=self.dtype).unsqueeze(-1)  # (n, 1)

        # optimize_acqf_discrete with q=2 returns a pair (2, 1)
        X_next, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=2,
            choices=choices,
            unique=True,
        )  # X_next: (2, 1)

        x1 = X_next[0, 0].item()
        x2 = X_next[1, 0].item()
        return x1, x2

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> float:
        if not comparisons:
            return candidate_pool[len(candidate_pool) // 2].item()

        model = self._fit_model(comparisons)
        mean = self._posterior_mean(model, candidate_pool)
        return candidate_pool[mean.argmax()].item()

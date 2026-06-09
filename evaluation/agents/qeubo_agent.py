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

from .base import PBOAgent, Comparison, candidate_matrix, candidate_value

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
        datapoints:  (n_unique, d) — all unique x values seen
        comparisons: (m, 2) LongTensor — [winner_idx, loser_idx] into datapoints
    """
    # Collect unique x values preserving insertion order
    def key(point):
        tensor = torch.as_tensor(point, dtype=dtype).reshape(-1)
        return tuple(float(v) for v in tensor.tolist())

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
    )  # (n, d)

    comp_idx = torch.tensor(
        [[seen[key(w)], seen[key(l)]] for w, l in comparisons],
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
        candidate_pool: (n,) or (n, d) tensor
        """
        X = candidate_matrix(candidate_pool, dtype=self.dtype)
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
    ) -> tuple:
        if not comparisons:
            # No data yet — return a random pair
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)

        from botorch.sampling.normal import SobolQMCNormalSampler
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))
        acqf = qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

        choices = candidate_matrix(candidate_pool, dtype=self.dtype)

        # optimize_acqf_discrete with q=2 returns a pair (2, d)
        X_next, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=2,
            choices=choices,
            unique=True,
        )  # X_next: (2, d)

        return candidate_value(X_next[0]), candidate_value(X_next[1])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ):
        if not comparisons:
            return candidate_value(candidate_pool[len(candidate_pool) // 2])

        model = self._fit_model(comparisons)
        mean = self._posterior_mean(model, candidate_pool)
        return candidate_value(candidate_pool[mean.argmax()])

"""
Fixed-hyperparameter qEUBO agent.

This mirrors the BoTorch qEUBO estimator used in scripts/pbo_botorch_qeubo_eval.py:
- PairwiseGP is built on the full candidate grid.
- RBF hyperparameters are fixed to known GP prior values.
- The latent utility scale is normalized by noise_std through outputscale / noise_std**2.
- Empty-context acquisition uses a known-prior MC qEUBO fallback.
- Pair selection materializes the full qEUBO matrix over the candidate grid.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import (
    Comparison,
    PBOAgent,
    candidate_matrix,
    candidate_value,
    nearest_candidate_index,
)

try:
    from botorch.acquisition.preference import qExpectedUtilityOfBestOption
    from botorch.models.pairwise_gp import PairwiseGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.kernels import RBFKernel, ScaleKernel

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


def _require_botorch():
    if not _BOTORCH_AVAILABLE:
        raise ImportError(
            "botorch is required for FixedHyperparamQEUBOAgent.\n"
            "Install with: pip install botorch"
        )


def _rbf_kernel(
    x: Tensor,
    *,
    lengthscale: float,
    outputscale: float,
) -> Tensor:
    x = candidate_matrix(x)
    d2 = torch.cdist(x, x).square()
    return outputscale * torch.exp(-0.5 * d2 / (lengthscale**2))


def _compute_qeubo_from_samples(F: Tensor) -> Tensor:
    Fi = F[:, :, None]
    Fj = F[:, None, :]
    return torch.maximum(Fi, Fj).mean(dim=0)


class FixedHyperparamQEUBOAgent(PBOAgent):
    """
    Args:
        lengthscale: Known latent GP RBF lengthscale.
        outputscale: Known latent GP output scale.
        noise_std: Gaussian comparison noise scale used to normalize PairwiseGP utility.
        mean_constant: Known latent GP mean constant.
        jitter: Numerical jitter.
        xtol: PairwiseGP Laplace solver tolerance.
        maxfev: PairwiseGP Laplace solver maximum function evaluations.
        num_acqf_samples: MC samples for both prior fallback and posterior qEUBO.
        batch_eval_size: Number of candidate pairs evaluated per acqf forward pass.
    """

    def __init__(
        self,
        *,
        lengthscale: float = 0.2,
        outputscale: float = 1.0,
        noise_std: float = 0.05,
        mean_constant: float = 0.0,
        jitter: float = 1e-6,
        xtol: float = 1e-6,
        maxfev: int = 100,
        num_acqf_samples: int = 512,
        batch_eval_size: int = 2048,
        dtype=torch.float64,
    ) -> None:
        _require_botorch()
        self.lengthscale = float(lengthscale)
        self.outputscale = float(outputscale)
        self.noise_std = float(noise_std)
        self.mean_constant = float(mean_constant)
        self.jitter = float(jitter)
        self.xtol = float(xtol)
        self.maxfev = int(maxfev)
        self.num_acqf_samples = int(num_acqf_samples)
        self.batch_eval_size = int(batch_eval_size)
        self.dtype = dtype

    def _pool_indices(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> Tensor:
        rows = []
        for winner, loser in comparisons:
            winner_idx = nearest_candidate_index(candidate_pool, winner)
            loser_idx = nearest_candidate_index(candidate_pool, loser)
            rows.append([winner_idx, loser_idx])
        return torch.tensor(rows, dtype=torch.long)

    def _fit_model(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> "PairwiseGP":
        train_X = candidate_matrix(candidate_pool, dtype=self.dtype)
        train_comp = self._pool_indices(comparisons, candidate_pool)

        base_kernel = RBFKernel(ard_num_dims=train_X.shape[-1])
        base_kernel.lengthscale = self.lengthscale

        covar_module = ScaleKernel(base_kernel)
        covar_module.outputscale = self.outputscale / (self.noise_std**2)

        base_kernel.raw_lengthscale.requires_grad_(False)
        covar_module.raw_outputscale.requires_grad_(False)

        model = PairwiseGP(
            train_X,
            train_comp,
            covar_module=covar_module,
            jitter=self.jitter,
            xtol=self.xtol,
            maxfev=self.maxfev,
        ).eval()
        return model

    def _known_prior_qeubo(self, candidate_pool: Tensor) -> Tensor:
        x = candidate_matrix(candidate_pool, dtype=self.dtype)
        K = _rbf_kernel(
            x,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
        )
        M = x.shape[0]
        K = K + self.jitter * torch.eye(M, dtype=self.dtype, device=x.device)
        L = torch.linalg.cholesky(K)
        z = torch.randn(self.num_acqf_samples, M, dtype=self.dtype, device=x.device)
        F = self.mean_constant + z @ L.T
        return _compute_qeubo_from_samples(F).float()

    def _posterior_qeubo(self, model: "PairwiseGP", candidate_pool: Tensor) -> Tensor:
        x = candidate_matrix(candidate_pool, dtype=self.dtype)
        M = x.shape[0]
        x1 = x.repeat_interleave(M, dim=0)
        x2 = x.repeat(M, 1)
        pair_grid = torch.stack([x1, x2], dim=1)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))
        acqf = qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

        batch_size = pair_grid.shape[0] if self.batch_eval_size <= 0 else self.batch_eval_size
        values = []
        with torch.no_grad():
            for start in range(0, pair_grid.shape[0], batch_size):
                values.append(acqf(pair_grid[start : start + batch_size]).cpu())
        return torch.cat(values).reshape(M, M).float()

    def _qeubo_matrix(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> Tensor:
        if not comparisons:
            return self._known_prior_qeubo(candidate_pool)
        model = self._fit_model(comparisons, candidate_pool)
        return self._posterior_qeubo(model, candidate_pool)

    def _posterior_mean(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> Tensor:
        if not comparisons:
            return torch.full((candidate_pool.shape[0],), self.mean_constant, dtype=torch.float32)
        model = self._fit_model(comparisons, candidate_pool)
        X = candidate_matrix(candidate_pool, dtype=self.dtype)
        with torch.no_grad():
            posterior = model.posterior(X)
        return posterior.mean.squeeze(-1).squeeze(-1).detach().cpu().float()

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple:
        qeubo = self._qeubo_matrix(comparisons, candidate_pool)
        idx = torch.arange(qeubo.shape[0])
        qeubo[idx, idx] = -torch.inf
        flat_idx = torch.argmax(qeubo)
        i = int(flat_idx // qeubo.shape[1])
        j = int(flat_idx % qeubo.shape[1])
        return candidate_value(candidate_pool[i]), candidate_value(candidate_pool[j])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ):
        mean = self._posterior_mean(comparisons, candidate_pool)
        return candidate_value(candidate_pool[mean.argmax()])

"""
GP preference model agent (Chu & Ghahramani 2005 style).

Posterior over f is approximated via Laplace approximation under a
Thurstone-Mosteller (probit) preference likelihood.

Model:
    f ~ GP(0, k_RBF(x, x'))
    P(x_i > x_j | f) = Phi( (f_i - f_j) / sqrt(2) )   [probit]

Laplace approximation:
    f_MAP = argmax  log p(comparisons | f) + log p(f)
    posterior cov  = (K^{-1} + W)^{-1}   where W = -d²log p / df²

We work on the candidate_pool grid: all candidates are inducing points.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import (
    PBOAgent,
    Comparison,
    candidate_matrix,
    candidate_value,
    nearest_candidate_index,
)


# ---------------------------------------------------------------------------
# Probit preference log-likelihood and its derivatives
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_LOG2PI = math.log(2 * math.pi)


def _probit(z: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(z / _SQRT2))


def _log_probit(z: Tensor) -> Tensor:
    return torch.special.log_ndtr(z)


def _pref_log_lik(
    f: Tensor,          # (n,)
    winner_idx: list[int],
    loser_idx: list[int],
) -> Tensor:
    """Sum of log Phi( (f_winner - f_loser) / sqrt(2) )."""
    if not winner_idx:
        return torch.tensor(0.0, dtype=f.dtype, device=f.device)
    fw = f[winner_idx]
    fl = f[loser_idx]
    z = (fw - fl) / _SQRT2
    return _log_probit(z).sum()


# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------

def _rbf_kernel(x: Tensor, lengthscale: float, outputscale: float) -> Tensor:
    """x: (n,) or (n, d) -> K: (n, n)."""
    x = candidate_matrix(x)
    sq_dist = torch.cdist(x, x).square()
    return outputscale * torch.exp(-0.5 * sq_dist / lengthscale**2)


# ---------------------------------------------------------------------------
# Laplace approximation
# ---------------------------------------------------------------------------

def _laplace_posterior(
    x_pool: Tensor,
    winner_idx: list[int],
    loser_idx: list[int],
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    jitter: float = 1e-4,
    n_iter: int = 50,
    lr: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """
    Returns (f_mean, f_cov) on x_pool via Laplace approximation.
    f_mean: (n,), f_cov: (n, n)
    """
    n = x_pool.shape[0]
    K = _rbf_kernel(x_pool, lengthscale, outputscale)
    K = K + jitter * torch.eye(n, dtype=K.dtype, device=K.device)
    K_inv = torch.linalg.inv(K)

    if not winner_idx:
        return torch.zeros(n, dtype=K.dtype), K

    # Newton steps to find f_MAP
    f = torch.zeros(n, dtype=K.dtype, requires_grad=False)
    _SQRT2PI = math.sqrt(2 * math.pi)
    for _ in range(n_iter):
        f = f.detach().requires_grad_(True)
        ll = _pref_log_lik(f, winner_idx, loser_idx)
        lp = -0.5 * (f @ K_inv @ f)
        objective = ll + lp
        objective.backward()
        with torch.no_grad():
            grad = f.grad.nan_to_num(0.0).clamp(-10.0, 10.0)
            f = f + lr * grad

    f_map = f.detach()

    # Hessian of log-likelihood at f_MAP (diagonal W matrix)
    # d²/df_i² log Phi(z_ij) where z_ij = (f_i - f_j)/sqrt(2)
    W = torch.zeros(n, dtype=K.dtype)
    if winner_idx:
        fw = f_map[winner_idx]
        fl = f_map[loser_idx]
        z = (fw - fl) / _SQRT2
        phi = torch.exp(-0.5 * z ** 2) / _SQRT2PI
        Phi = _probit(z).clamp(1e-8, 1 - 1e-8)
        lam = (phi / Phi) ** 2  # (Fisher info per comparison)
        lam = lam.nan_to_num(0.0)
        for i, (wi, li) in enumerate(zip(winner_idx, loser_idx)):
            W[wi] += lam[i] / 2
            W[li] += lam[i] / 2

    A = K_inv + torch.diag(W)
    f_cov = torch.linalg.inv(A + jitter * torch.eye(n, dtype=K.dtype))

    return f_map, f_cov


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GPPBOAgent(PBOAgent):
    """
    Args:
        lengthscale: RBF kernel lengthscale.
        outputscale: RBF kernel output scale.
        n_ts_samples: Thompson Sampling draws for suggest_pair.
    """

    def __init__(
        self,
        lengthscale: float = 0.2,
        outputscale: float = 1.0,
        n_ts_samples: int = 2,
        support: str = "grid",
    ):
        assert support == "grid", "Continuous support is not implemented for GPPBOAgent yet."
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.n_ts_samples = n_ts_samples
        self.support = support

    def _pool_indices(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[list[int], list[int]]:
        """Map x values to indices in candidate_pool."""
        def snap(x):
            return nearest_candidate_index(candidate_pool, x)

        winner_idx = [snap(w) for w, _ in comparisons]
        loser_idx  = [snap(l) for _, l in comparisons]
        return winner_idx, loser_idx

    def _posterior(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wi, li = self._pool_indices(comparisons, candidate_pool)
        return _laplace_posterior(
            candidate_pool,
            wi, li,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
        )

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ):
        if not comparisons:
            return candidate_value(candidate_pool[candidate_pool.shape[0] // 2])
        f_mean, _ = self._posterior(comparisons, candidate_pool)
        return candidate_value(candidate_pool[f_mean.argmax()])

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple:
        f_mean, f_cov = self._posterior(comparisons, candidate_pool)

        argmaxes = []
        eye = torch.eye(len(f_mean), dtype=f_cov.dtype, device=f_cov.device)
        cov = (f_cov + f_cov.T) / 2 + 1e-4 * eye  # symmetrize + jitter
        dist = torch.distributions.MultivariateNormal(
            f_mean,
            covariance_matrix=cov,
        )
        for _ in range(self.n_ts_samples):
            sample = dist.sample()
            argmaxes.append(candidate_value(candidate_pool[sample.argmax()]))

        if len(set(argmaxes)) == 1:
            challenger_idx = torch.randint(len(candidate_pool), (1,)).item()
            return argmaxes[0], candidate_value(candidate_pool[challenger_idx])

        return argmaxes[0], argmaxes[1]

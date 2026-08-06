from dataclasses import dataclass
from functools import partial

import gpytorch
import torch

from pfns.priors import Batch
from pfns.priors.prior import PriorConfig

torch.set_default_dtype(torch.double)


def make_gp_prior(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    jitter=1e-6,
):
    gp_dim = X.shape[-1]

    mean_module = gpytorch.means.ConstantMean()
    mean_module.initialize(constant=mean_constant)

    # ARD-capable RBF kernel.
    # - scalar lengthscale -> isotropic
    # - lengthscale with shape (gp_dim,) -> ARD
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=gp_dim)

    lengthscale = torch.as_tensor(
        lengthscale,
        dtype=X.dtype,
        device=X.device,
    )

    if lengthscale.ndim == 0:
        # isotropic case
        base_kernel.lengthscale = lengthscale
    else:
        # ARD case
        assert lengthscale.numel() == gp_dim, (
            f"ARD lengthscale must have {gp_dim} entries, "
            f"got {lengthscale.numel()}."
        )
        base_kernel.lengthscale = lengthscale.reshape(1, gp_dim)

    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    covar_module.outputscale = outputscale

    return gpytorch.distributions.MultivariateNormal(
        mean_module(X),
        covar_module(X),
    ).add_jitter(jitter)


def sample_gp_batch(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=None,
    jitter=1e-6,
):
    with torch.no_grad():
        f_dist = make_gp_prior(
            X,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            jitter=jitter,
        )

        f = f_dist.rsample()

        if noise_std is None:
            y = f
        else:
            y = f + noise_std * torch.randn_like(f)

    return f.detach(), y.detach()


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=2,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    pool_size=100,          # finite candidate pool size K
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    noise_per_comparison=True,
    jitter=1e-6,
    **kwargs,
):
    """Preferential qEUBO prior over a finite candidate pool.

    `noise_per_comparison` selects the comparison likelihood:

    - ``True`` (default, standard preferential model): the latent `f` is drawn once on the
      pool and **fresh observation noise is drawn for every comparison**, so
      `P(i > j) = P(f_i + e_i > f_j + e_j)` independently per comparison and repeated or
      chained comparisons can contradict each other. This matches the probit likelihood used
      by BoTorch's `PairwiseGP`, by `pref_gp_1d_qeubo_regret_v3.py`, and by the ground-truth
      sampler in `scripts/pbo_ground_truth.py`.
    - ``False`` (legacy): noise is drawn **once per pool point** and frozen, so every
      comparison is read off a single fixed total order on `y` and contradictions are
      impossible. This was the behaviour of this file before the fix and is retained only so
      the training distribution of the existing `*_pool_*` checkpoints stays reproducible.

    The distinction only has an effect when a pool point takes part in more than one
    comparison, i.e. it grows with the average degree `2 * n_ctx / pool_size` of the
    comparison graph. It is negligible at the historical default (degree <= 1.98) and
    first-order once the degree is raised deliberately.
    """
    assert num_features % 2 == 0, (
        "Preferential GP qEUBO prior expects num_features = 2 * gp_dim, "
        f"got num_features={num_features}."
    )
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 2, "pool_size must be at least 2 for non-diagonal comparisons."

    gp_dim = num_features // 2

    # ------------------------------------------------------------
    # finite candidate pool per batch element
    # ------------------------------------------------------------
    # pool_X: (B, K, gp_dim)
    pool_X = torch.rand(
        batch_size,
        pool_size,
        gp_dim,
        device=device,
    )

    # Sample GP values on the finite pool
    # Fs, Ys: (B, K)
    # With per-comparison noise the pool carries only the latent f; the noise is added later,
    # once per comparison. With the legacy per-point noise it is frozen here.
    Fs, Ys = sample_gp_batch(
        pool_X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None if noise_per_comparison else noise_std,
        jitter=jitter,
    )

    new_X = torch.zeros(
        batch_size,
        seq_len,
        num_features,
        device=device,
        dtype=pool_X.dtype,
    )
    qeubo = torch.zeros(
        batch_size,
        seq_len,
        device=device,
        dtype=Fs.dtype,
    )

    B = batch_size
    K = pool_size
    batch_idx = torch.arange(B, device=device)

    # ------------------------------------------------------------
    # sample context pairs without replacement within pair
    # ------------------------------------------------------------
    n_ctx = single_eval_pos
    if n_ctx > 0:
        idx0_ctx = torch.randint(K, size=(B, n_ctx), device=device)
        idx1_ctx = torch.randint(K - 1, size=(B, n_ctx), device=device)

        # transform idx1 so idx1_ctx != idx0_ctx
        idx1_ctx = idx1_ctx + (idx1_ctx >= idx0_ctx).long()

        x0 = pool_X[batch_idx[:, None], idx0_ctx]  # (B, n_ctx, gp_dim)
        x1 = pool_X[batch_idx[:, None], idx1_ctx]  # (B, n_ctx, gp_dim)

        if noise_per_comparison:
            f0_ctx = Fs[batch_idx[:, None], idx0_ctx]  # (B, n_ctx)
            f1_ctx = Fs[batch_idx[:, None], idx1_ctx]  # (B, n_ctx)

            if noise_std is None or noise_std == 0.0:
                y0 = f0_ctx
                y1 = f1_ctx
            else:
                # Fresh noise per comparison occurrence
                y0 = f0_ctx + noise_std * torch.randn_like(f0_ctx)
                y1 = f1_ctx + noise_std * torch.randn_like(f1_ctx)
        else:
            # Legacy: a single frozen noisy value per pool point
            y0 = Ys[batch_idx[:, None], idx0_ctx]  # (B, n_ctx)
            y1 = Ys[batch_idx[:, None], idx1_ctx]  # (B, n_ctx)

        prefer_x0 = (y0 > y1).unsqueeze(-1)        # (B, n_ctx, 1)

        first_x = torch.where(prefer_x0, x0, x1)
        second_x = torch.where(prefer_x0, x1, x0)

        new_X[:, :n_ctx, :] = torch.cat([first_x, second_x], dim=-1)

    # ------------------------------------------------------------
    # sample query pairs with replacement
    # ------------------------------------------------------------
    n_query = seq_len - single_eval_pos
    if n_query > 0:
        idx0_q = torch.randint(K, size=(B, n_query), device=device)
        idx1_q = torch.randint(K, size=(B, n_query), device=device)

        x0 = pool_X[batch_idx[:, None], idx0_q]  # (B, n_query, gp_dim)
        x1 = pool_X[batch_idx[:, None], idx1_q]  # (B, n_query, gp_dim)

        f0 = Fs[batch_idx[:, None], idx0_q]      # (B, n_query)
        f1 = Fs[batch_idx[:, None], idx1_q]      # (B, n_query)

        new_X[:, n_ctx:, :] = torch.cat([x0, x1], dim=-1)
        qeubo[:, n_ctx:] = torch.maximum(f0, f1)

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPqEUBOPoolPriorConfig(PriorConfig):
    lengthscale: float | tuple[float, ...] = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6

    # New finite-pool parameter
    pool_size: int = 100

    # Draw fresh observation noise for every comparison (standard preferential likelihood)
    # rather than freezing one noisy value per pool point. See `get_batch` for details.
    # Set to False only to reproduce the training distribution of checkpoints produced before
    # 2026-08-06, i.e. all existing `pfn_pref_gp_*d_qeubo_*_pool_*.pt`.
    noise_per_comparison: bool = True

    def create_get_batch_method(self):
        return partial(
            get_batch,
            pool_size=self.pool_size,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            noise_per_comparison=self.noise_per_comparison,
            jitter=self.jitter,
        )
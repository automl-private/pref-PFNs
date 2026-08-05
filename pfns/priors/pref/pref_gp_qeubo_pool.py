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
    jitter=1e-6,
    **kwargs,
):
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
    Fs, Ys = sample_gp_batch(
        pool_X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
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

        y0 = Ys[batch_idx[:, None], idx0_ctx]      # (B, n_ctx)
        y1 = Ys[batch_idx[:, None], idx1_ctx]      # (B, n_ctx)

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

    def create_get_batch_method(self):
        return partial(
            get_batch,
            pool_size=self.pool_size,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
        )
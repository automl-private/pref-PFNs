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
    mean_module = gpytorch.means.ConstantMean()
    mean_module.initialize(constant=mean_constant)

    base_kernel = gpytorch.kernels.RBFKernel()
    base_kernel.lengthscale = lengthscale

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


def _sample_x_pool(
    batch_size: int,
    pool_size: int,
    gp_dim: int,
    device,
    dtype,
    shared_x_pool: bool,
):
    """
    Returns X_pool of shape (B, pool_size, gp_dim).
    Always includes 0 and 1, plus pool_size - 2 uniform random points.
    """
    assert gp_dim == 1
    assert pool_size >= 2

    n_pools = 1 if shared_x_pool else batch_size

    endpoints = torch.tensor([0.0, 1.0], device=device, dtype=dtype).view(1, 2, 1)
    if pool_size > 2:
        rand_x = torch.rand(n_pools, pool_size - 2, gp_dim, device=device, dtype=dtype)
        X_pool = torch.cat([endpoints.expand(n_pools, -1, -1), rand_x], dim=1)
    else:
        X_pool = endpoints.expand(n_pools, -1, -1)

    if shared_x_pool:
        X_pool = X_pool.expand(batch_size, -1, -1)

    return X_pool


def _sample_distinct_pair_indices(batch_size: int, seq_len: int, pool_size: int, device):
    """
    Sample (i0, i1) with i0 != i1 for each token, shape (B, L).
    Sampling is with replacement across tokens, which is cheap and gives reuse.
    """
    i0 = torch.randint(pool_size, (batch_size, seq_len), device=device)
    i1 = torch.randint(pool_size - 1, (batch_size, seq_len), device=device)
    i1 = i1 + (i1 >= i0).long()
    return i0, i1


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=2,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    jitter=1e-6,
    pool_size=200,
    shared_x_pool=False,
    grid_size=101,
    **kwargs,
):
    assert num_features == 2, "pref_gp_1d only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len

    gp_dim = num_features // 2
    assert gp_dim == 1, "This qEUBO regret construction assumes a 1D domain [0,1]."

    dtype = torch.get_default_dtype()

    # 1) Sample a pool of x-values per batch element (or optionally one shared pool).
    X_pool = _sample_x_pool(
        batch_size=batch_size,
        pool_size=pool_size,
        gp_dim=gp_dim,
        device=device,
        dtype=dtype,
        shared_x_pool=shared_x_pool,
    )  # (B, N, 1)

    # 2) Fixed grid for approximating f*
    grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype).view(1, grid_size, 1)
    grid = grid.expand(batch_size, -1, -1)  # (B, M, 1)

    # 3) Joint GP draw on pool + grid so regrets and comparisons live on same sample path.
    #    Important: do NOT add observation noise here; comparison noise is resampled per occurrence below.
    X_joint = torch.cat([X_pool, grid], dim=1)  # (B, N + M, 1)
    Fs_joint, _ = sample_gp_batch(
        X_joint,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None,
        jitter=jitter,
    )

    Fs_pool = Fs_joint[:, :pool_size]        # (B, N)
    grid_Fs = Fs_joint[:, pool_size:]        # (B, M)
    f_star = grid_Fs.max(dim=1).values       # (B,)

    # 4) Randomly form seq_len distinct pairs from the pool, with reuse across tokens.
    i0, i1 = _sample_distinct_pair_indices(batch_size, seq_len, pool_size, device)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)

    x0 = X_pool[batch_idx, i0]   # (B, L, 1)
    x1 = X_pool[batch_idx, i1]   # (B, L, 1)

    f0 = Fs_pool[batch_idx, i0]  # (B, L)
    f1 = Fs_pool[batch_idx, i1]  # (B, L)

    # 5) Resample comparison noise per occurrence (not per pooled point).
    if noise_std is None or noise_std == 0.0:
        y0 = f0
        y1 = f1
    else:
        y0 = f0 + noise_std * torch.randn_like(f0)
        y1 = f1 + noise_std * torch.randn_like(f1)

    # 6) Context: reorder so preferred point is first according to noisy comparison.
    #    Query: keep original random order.
    prefer_x0 = (y0 > y1).unsqueeze(-1)  # (B, L, 1)

    first_x_context = torch.where(prefer_x0, x0, x1)
    second_x_context = torch.where(prefer_x0, x1, x0)

    token_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    context_mask = token_idx < single_eval_pos   # (1, L, 1)
    query_mask = ~context_mask                   # (1, L, 1)

    first_x = torch.where(context_mask, first_x_context, x0)
    second_x = torch.where(context_mask, second_x_context, x1)

    new_X = torch.cat([first_x, second_x], dim=-1)  # (B, L, 2)

    # 7) Query targets: qEUBO regret = max(f(x0), f(x1)) - f*
    qeubo_val = torch.maximum(f0, f1)  # (B, L)
    qeubo_regret = (qeubo_val - f_star.unsqueeze(1)).clamp(-10.0, 0.0)

    # Context positions get dummy target 0
    qeubo_regret = torch.where(
        query_mask.squeeze(-1),
        qeubo_regret,
        torch.zeros_like(qeubo_regret),
    )

    return Batch(
        x=new_X,
        y=qeubo_regret,
        target_y=qeubo_regret,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DqEUBORegretV2PriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    pool_size: int = 200
    shared_x_pool: bool = False
    grid_size: int = 101

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            pool_size=self.pool_size,
            shared_x_pool=self.shared_x_pool,
            grid_size=self.grid_size,
        )
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
    Returns X_pool of shape (B, N, 1).
    Includes 0 and 1, plus N-2 uniform random points.
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


def _sample_pair_indices(batch_size: int, n_pairs: int, pool_size: int, device):
    """
    Unconstrained pair sampling. Identity pairs are allowed by design.
    """
    i0 = torch.randint(pool_size, (batch_size, n_pairs), device=device)
    i1 = torch.randint(pool_size, (batch_size, n_pairs), device=device)
    return i0, i1


def _effective_query_mode_probs(
    context_len: int,
    base_rr: float,
    base_wr: float,
    base_ww: float,
):
    """
    Smooth ramp-up of winner-based query modes for small context sizes.

    For C <= 1:
        rr = 1, wr = 0, ww = 0

    For C >= 2:
        wr(C) = base_wr * C / (C + 1)
        ww(C) = base_ww * (C - 1) / (C + 2)
        rr(C) = 1 - wr(C) - ww(C)
    """
    if context_len == 0:
        return 1.0, 0.0, 0.0

    C = float(context_len)
    p_wr = base_wr * C / (C + 1.0)
    p_ww = base_ww * (C - 1.0) / (C + 2.0)
    p_rr = 1.0 - p_wr - p_ww
    return p_rr, p_wr, p_ww


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
    query_random_prob=0.50,
    query_winner_random_prob=0.30,
    query_winner_winner_prob=0.20,
    query_swap_prob=0.5,
    **kwargs,
):
    assert num_features == 2, "pref_gp_1d only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert abs(
        query_random_prob + query_winner_random_prob + query_winner_winner_prob - 1.0
    ) < 1e-12

    gp_dim = num_features // 2
    assert gp_dim == 1, "This qEUBO regret construction assumes a 1D domain [0,1]."

    dtype = torch.get_default_dtype()
    context_len = single_eval_pos
    query_len = seq_len - single_eval_pos

    # ------------------------------------------------------------------
    # 1) Sample pooled x-values per batch element
    # ------------------------------------------------------------------
    X_pool = _sample_x_pool(
        batch_size=batch_size,
        pool_size=pool_size,
        gp_dim=gp_dim,
        device=device,
        dtype=dtype,
        shared_x_pool=shared_x_pool,
    )  # (B, N, 1)

    # ------------------------------------------------------------------
    # 2) Sample latent GP jointly on pool + fixed grid
    # ------------------------------------------------------------------
    grid = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype).view(1, grid_size, 1)
    grid = grid.expand(batch_size, -1, -1)  # (B, M, 1)

    X_joint = torch.cat([X_pool, grid], dim=1)  # (B, N + M, 1)
    Fs_joint, _ = sample_gp_batch(
        X_joint,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None,  # comparison noise is added only for context outcomes
        jitter=jitter,
    )

    Fs_pool = Fs_joint[:, :pool_size]   # (B, N)
    grid_Fs = Fs_joint[:, pool_size:]   # (B, M)
    f_star = grid_Fs.max(dim=1).values  # (B,)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)

    # ------------------------------------------------------------------
    # 3) Context: random/random comparisons, reordered by noisy preference
    # ------------------------------------------------------------------
    if context_len > 0:
        ctx_i0, ctx_i1 = _sample_pair_indices(batch_size, context_len, pool_size, device)

        ctx_x0 = X_pool[batch_idx, ctx_i0]   # (B, C, 1)
        ctx_x1 = X_pool[batch_idx, ctx_i1]   # (B, C, 1)

        ctx_f0 = Fs_pool[batch_idx, ctx_i0]  # (B, C)
        ctx_f1 = Fs_pool[batch_idx, ctx_i1]  # (B, C)

        if noise_std is None or noise_std == 0.0:
            ctx_y0 = ctx_f0
            ctx_y1 = ctx_f1
        else:
            # Fresh noise per comparison occurrence
            ctx_y0 = ctx_f0 + noise_std * torch.randn_like(ctx_f0)
            ctx_y1 = ctx_f1 + noise_std * torch.randn_like(ctx_f1)

        ctx_prefer_x0 = (ctx_y0 > ctx_y1).unsqueeze(-1)

        # Context encodes preference by order: (winner, loser)
        ctx_first_x = torch.where(ctx_prefer_x0, ctx_x0, ctx_x1)
        ctx_second_x = torch.where(ctx_prefer_x0, ctx_x1, ctx_x0)

        ctx_winner_idx = torch.where(ctx_y0 > ctx_y1, ctx_i0, ctx_i1)  # (B, C)
    else:
        ctx_first_x = torch.empty(batch_size, 0, 1, device=device, dtype=dtype)
        ctx_second_x = torch.empty(batch_size, 0, 1, device=device, dtype=dtype)
        ctx_winner_idx = torch.empty(batch_size, 0, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # 4) Queries: mixture of rr / winner-random / winner-winner
    # ------------------------------------------------------------------
    if query_len > 0:
        p_rr_eff, p_wr_eff, p_ww_eff = _effective_query_mode_probs(
            context_len=context_len,
            base_rr=query_random_prob,
            base_wr=query_winner_random_prob,
            base_ww=query_winner_winner_prob,
        )

        u = torch.rand(batch_size, query_len, device=device)
        mode_rr = u < p_rr_eff
        mode_wr = (u >= p_rr_eff) & (u < p_rr_eff + p_wr_eff)
        mode_ww = ~(mode_rr | mode_wr)

        # random/random proposals
        rr_i0, rr_i1 = _sample_pair_indices(batch_size, query_len, pool_size, device)

        if context_len > 0:
            # winner/random proposals
            wr_ctx_pos = torch.randint(context_len, (batch_size, query_len), device=device)
            wr_winner = ctx_winner_idx[batch_idx, wr_ctx_pos]
            wr_rand = torch.randint(pool_size, (batch_size, query_len), device=device)

            # winner/winner proposals
            ww_ctx_pos0 = torch.randint(context_len, (batch_size, query_len), device=device)
            ww_ctx_pos1 = torch.randint(context_len, (batch_size, query_len), device=device)
            ww_i0 = ctx_winner_idx[batch_idx, ww_ctx_pos0]
            ww_i1 = ctx_winner_idx[batch_idx, ww_ctx_pos1]

            qry_i0 = torch.where(
                mode_rr,
                rr_i0,
                torch.where(mode_wr, wr_winner, ww_i0),
            )
            qry_i1 = torch.where(
                mode_rr,
                rr_i1,
                torch.where(mode_wr, wr_rand, ww_i1),
            )
        else:
            # If there is no context, everything falls back to random/random.
            qry_i0, qry_i1 = rr_i0, rr_i1

        qry_x0 = X_pool[batch_idx, qry_i0]   # (B, Q, 1)
        qry_x1 = X_pool[batch_idx, qry_i1]   # (B, Q, 1)

        qry_f0 = Fs_pool[batch_idx, qry_i0]  # (B, Q)
        qry_f1 = Fs_pool[batch_idx, qry_i1]  # (B, Q)

        # Randomize query orientation to avoid positional bias.
        swap_q = torch.rand(batch_size, query_len, 1, device=device) < query_swap_prob
        qry_first_x = torch.where(swap_q, qry_x1, qry_x0)
        qry_second_x = torch.where(swap_q, qry_x0, qry_x1)

        qry_qeubo_val = torch.maximum(qry_f0, qry_f1)
        qry_regret = (qry_qeubo_val - f_star.unsqueeze(1)).clamp(-10.0, 0.0)
    else:
        qry_first_x = torch.empty(batch_size, 0, 1, device=device, dtype=dtype)
        qry_second_x = torch.empty(batch_size, 0, 1, device=device, dtype=dtype)
        qry_regret = torch.empty(batch_size, 0, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 5) Assemble sequence
    # ------------------------------------------------------------------
    first_x = torch.cat([ctx_first_x, qry_first_x], dim=1)     # (B, L, 1)
    second_x = torch.cat([ctx_second_x, qry_second_x], dim=1)  # (B, L, 1)
    new_X = torch.cat([first_x, second_x], dim=-1)             # (B, L, 2)

    ctx_regret = torch.zeros(batch_size, context_len, device=device, dtype=dtype)
    qeubo_regret = torch.cat([ctx_regret, qry_regret], dim=1)

    return Batch(
        x=new_X,
        y=qeubo_regret,
        target_y=qeubo_regret,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DqEUBORegretV3PriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6

    pool_size: int = 200
    shared_x_pool: bool = False
    grid_size: int = 101

    query_random_prob: float = 0.50
    query_winner_random_prob: float = 0.30
    query_winner_winner_prob: float = 0.20
    query_swap_prob: float = 0.5

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
            query_random_prob=self.query_random_prob,
            query_winner_random_prob=self.query_winner_random_prob,
            query_winner_winner_prob=self.query_winner_winner_prob,
            query_swap_prob=self.query_swap_prob,
        )
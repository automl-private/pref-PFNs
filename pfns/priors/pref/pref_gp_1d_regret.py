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


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=1,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    jitter=1e-6,
    **kwargs,
):
    assert num_features == 2, "pref_gp_1d only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len

    gp_dim = num_features // 2
    assert gp_dim == 1, "This regret construction assumes a 1D domain [0,1]."

    # Need:
    # - 2 * single_eval_pos points for pairwise-comparison context
    # - (seq_len - single_eval_pos) points for queries
    # Total = seq_len + single_eval_pos
    n_main = single_eval_pos + seq_len
    X = torch.rand(batch_size, n_main, gp_dim, device=device)

    # Fixed grid for approximate optimizer / incumbent value f*
    M = 101
    grid = torch.linspace(0.0, 1.0, M, device=device, dtype=X.dtype).view(1, M, 1)
    grid = grid.expand(batch_size, M, gp_dim)  # (B, M, 1)

    # Jointly sample on main inputs + grid so everything lies on the same GP sample path
    X_joint = torch.cat([X, grid], dim=1)  # (B, n_main + M, 1)

    Fs_joint, Ys_joint = sample_gp_batch(
        X_joint,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    # Split back into main samples and grid samples
    Fs = Fs_joint[:, :n_main]            # (B, n_main)
    Ys = Ys_joint[:, :n_main]            # (B, n_main)
    grid_Fs = Fs_joint[:, n_main:]       # (B, M)

    # Approximate optimum value on the fixed grid
    f_star = grid_Fs.max(dim=1).values   # (B,)

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    new_Fs = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)
    new_Ys = torch.zeros(batch_size, seq_len, device=device, dtype=Ys.dtype)

    # First single_eval_pos tokens: pairwise comparison context
    # pair t uses original indices (2t, 2t+1)
    for t in range(single_eval_pos):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]   # (B, 1)
        x1 = X[:, i1, :]   # (B, 1)
        y0 = Ys[:, i0]     # (B,)
        y1 = Ys[:, i1]     # (B,)

        prefer_x0 = (y0 > y1).unsqueeze(-1)  # (B, 1)

        best_x = torch.where(prefer_x0, x0, x1)   # (B, 1)
        worst_x = torch.where(prefer_x0, x1, x0)  # (B, 1)

        new_X[:, t, :] = torch.cat([best_x, worst_x], dim=-1)

    # Remaining tokens: utility queries [x, 0]
    num_queries = seq_len - single_eval_pos
    if num_queries > 0:
        src_start = 2 * single_eval_pos
        src_end = src_start + num_queries

        query_X = X[:, src_start:src_end, :]   # (B, num_queries, 1)
        query_Fs = Fs[:, src_start:src_end]    # (B, num_queries)
        query_Ys = Ys[:, src_start:src_end]    # (B, num_queries)

        new_X[:, single_eval_pos:, :gp_dim] = query_X
        # second half remains zero-padded
        new_Ys[:, single_eval_pos:] = query_Ys

        # Regret target:
        #   target = f* - f(x)
        # clipped at 0 to guard against imperfect grid optimization
        regret_Fs = (query_Fs - f_star.unsqueeze(1)).clamp(-10.0, 0.0)

        new_Fs[:, single_eval_pos:] = regret_Fs

    # Context targets remain zeroed out
    return Batch(
       x=new_X,
        y=new_Ys,
        target_y=new_Fs,
        single_eval_pos=single_eval_pos,
    )

@dataclass(frozen=True)
class PrefGP1DRegretPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
        )
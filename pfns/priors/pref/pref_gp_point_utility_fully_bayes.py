from dataclasses import dataclass
from functools import partial
import math

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



def _sample_log_uniform(
    batch_size: int,
    low: float,
    high: float,
    *,
    device,
) -> torch.Tensor:
    """Samples one positive hyperparameter per synthetic task."""
    if low <= 0.0 or high <= 0.0 or low > high:
        raise ValueError(f"Expected 0 < low <= high, got low={low}, high={high}.")

    log_values = torch.empty(batch_size, device=device).uniform_(math.log(low), math.log(high))
    return torch.exp(log_values)


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=2,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    lengthscale_min=0.05,
    lengthscale_max=1.0,
    outputscale_min=0.3,
    outputscale_max=3.0,
    mean_constant=0.0,
    noise_std_min=0.01,
    noise_std_max=0.15,
    jitter=1e-6,
    **kwargs,
):
    """
    Samples preference-context tasks with point-utility queries.

    Context tokens are pairwise comparisons encoded as [winner_x, loser_x].
    Query tokens contain one point encoded as [x, 0], and the target is f(x).
    """
    assert num_features % 2 == 0, "Features must store context pairs as [x1, x2]."
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len

    gp_dim = num_features // 2

    # We need two GP points per context comparison and one GP point per query.
    X = torch.rand(batch_size, seq_len + single_eval_pos, gp_dim, device=device)

    # Each synthetic task gets its own GP hyperparameters.
    lengthscale = _sample_log_uniform(
        batch_size,
        lengthscale_min,
        lengthscale_max,
        device=device,
    )
    outputscale = _sample_log_uniform(
        batch_size,
        outputscale_min,
        outputscale_max,
        device=device,
    )
    noise_std = _sample_log_uniform(
        batch_size,
        noise_std_min,
        noise_std_max,
        device=device,
    )

    Fs, Ys = sample_gp_batch(
        X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    # Token dimension stays 2 * gp_dim for both context pairs and point queries.
    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    observed_utility = torch.zeros(batch_size, seq_len, device=device, dtype=Ys.dtype)
    latent_utility = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    # First tokens are preference comparisons: best point first, worst point second.
    for t in range(single_eval_pos):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]
        x1 = X[:, i1, :]
        y0 = Ys[:, i0]
        y1 = Ys[:, i1]

        prefer_x0 = (y0 > y1).unsqueeze(-1)
        best_x = torch.where(prefer_x0, x0, x1)
        worst_x = torch.where(prefer_x0, x1, x0)

        new_X[:, t, :] = torch.cat([best_x, worst_x], dim=-1)

    # Remaining tokens are point queries: [x, 0] with target f(x).
    num_queries = seq_len - single_eval_pos
    if num_queries > 0:
        src_start = 2 * single_eval_pos
        src_end = src_start + num_queries

        query_X = X[:, src_start:src_end, :]
        query_Fs = Fs[:, src_start:src_end]
        query_Ys = Ys[:, src_start:src_end]

        new_X[:, single_eval_pos:, :gp_dim] = query_X
        observed_utility[:, single_eval_pos:] = query_Ys
        latent_utility[:, single_eval_pos:] = query_Fs

    return Batch(
        x=new_X,
        y=observed_utility,
        target_y=latent_utility,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPPointUtilityFullyBayesPriorConfig(PriorConfig):
    lengthscale_min: float = 0.05
    lengthscale_max: float = 1.0
    outputscale_min: float = 0.3
    outputscale_max: float = 3.0
    mean_constant: float = 0.0
    noise_std_min: float = 0.01
    noise_std_max: float = 0.15
    jitter: float = 1e-6

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale_min=self.lengthscale_min,
            lengthscale_max=self.lengthscale_max,
            outputscale_min=self.outputscale_min,
            outputscale_max=self.outputscale_max,
            mean_constant=self.mean_constant,
            noise_std_min=self.noise_std_min,
            noise_std_max=self.noise_std_max,
            jitter=self.jitter,
        )

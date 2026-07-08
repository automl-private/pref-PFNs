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
    batch_size = X.shape[0]
    gp_dim = X.shape[-1]

    lengthscale = torch.as_tensor(lengthscale, dtype=X.dtype, device=X.device)
    outputscale = torch.as_tensor(outputscale, dtype=X.dtype, device=X.device)

    if lengthscale.numel() == 1:
        lengthscale = lengthscale.reshape(1).expand(batch_size)
    else:
        lengthscale = lengthscale.reshape(batch_size)

    if outputscale.numel() == 1:
        outputscale = outputscale.reshape(1).expand(batch_size)
    else:
        outputscale = outputscale.reshape(batch_size)

    mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size])).to(
        device=X.device,
        dtype=X.dtype,
    )
    mean_module.constant = torch.full(
        (batch_size,),
        float(mean_constant),
        dtype=X.dtype,
        device=X.device,
    )

    base_kernel = gpytorch.kernels.RBFKernel(
        batch_shape=torch.Size([batch_size]),
        ard_num_dims=gp_dim,
    ).to(device=X.device, dtype=X.dtype)
    base_kernel.lengthscale = lengthscale.reshape(batch_size, 1, 1).expand(batch_size, 1, gp_dim)

    covar_module = gpytorch.kernels.ScaleKernel(
        base_kernel,
        batch_shape=torch.Size([batch_size]),
    ).to(device=X.device, dtype=X.dtype)
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
            noise_std = torch.as_tensor(noise_std, dtype=f.dtype, device=f.device)
            if noise_std.numel() == 1:
                y = f + noise_std * torch.randn_like(f)
            else:
                y = f + noise_std.reshape(f.shape[0], 1) * torch.randn_like(f)

    return f.detach(), y.detach()


def _sample_log_uniform(
    batch_size: int,
    low: float,
    high: float,
    *,
    device,
) -> torch.Tensor:
    """Samples one positive hyperparameter per batch element."""
    if low <= 0.0 or high <= 0.0 or low > high:
        raise ValueError(f"Expected 0 < low <= high, got low={low}, high={high}.")
    # If X is log-uniformly distributed between a and b, then \(\log(X)\) is uniformly distributed between \(\log(a)\) and \(\log(b)\).
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
    Samples pair-score qEUBO tasks from a fully Bayesian GP prior.

    Each batch element gets its own GP hyperparameters:
    `lengthscale`, `outputscale`, and preference-observation `noise_std`.
    The PFN sees only pairwise comparisons and query pairs, not these sampled
    hyperparameters.
    """
    assert num_features % 2 == 0, "Pair features must be concatenated as [x1, x2]."
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len

    gp_dim = num_features // 2

    # Each token is a pair, so we first sample two original GP inputs per token.
    X = torch.rand(batch_size, 2 * seq_len, gp_dim, device=device)

    # Fully Bayesian part: sample one GP hyperparameter set per synthetic task.
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

    # PFN input stores one pair per token as [x_first, x_second].
    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    qeubo = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    for t in range(seq_len):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]
        x1 = X[:, i1, :]

        if t < single_eval_pos:
            # Context comparisons are ordered as winner first, loser second.
            y0 = Ys[:, i0]
            y1 = Ys[:, i1]
            prefer_x0 = (y0 > y1).unsqueeze(-1)
            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)
        else:
            # Query pairs keep random order; target uses noiseless latent utility.
            f0 = Fs[:, i0]
            f1 = Fs[:, i1]
            first_x = x0
            second_x = x1
            qeubo[:, t] = torch.maximum(f0, f1)

        new_X[:, t, :] = torch.cat([first_x, second_x], dim=-1)

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPqEUBOFullyBayesPriorConfig(PriorConfig):
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

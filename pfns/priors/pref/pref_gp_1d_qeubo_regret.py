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
    assert gp_dim == 1, "This qEUBO regret construction assumes a 1D domain [0,1]."

    # Need 2 * seq_len original GP inputs, since each token is a pair
    X = torch.rand(batch_size, 2 * seq_len, gp_dim, device=device)

    # Fixed grid for approximate f*
    M = 101
    grid = torch.linspace(0.0, 1.0, M, device=device, dtype=X.dtype).view(1, M, 1)
    grid = grid.expand(batch_size, M, gp_dim)  # (B, M, 1)

    # Jointly sample so X and grid are on the same GP sample path
    X_joint = torch.cat([X, grid], dim=1)  # (B, 2 * seq_len + M, 1)

    Fs_joint, Ys_joint = sample_gp_batch(
        X_joint,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    Fs = Fs_joint[:, : 2 * seq_len]      # (B, 2 * seq_len)
    Ys = Ys_joint[:, : 2 * seq_len]      # (B, 2 * seq_len)
    grid_Fs = Fs_joint[:, 2 * seq_len :] # (B, M)

    # Approximate optimum value on the fixed grid
    f_star = grid_Fs.max(dim=1).values   # (B,)

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    qeubo_regret = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    # Token t uses original pair (2t, 2t+1)
    for t in range(seq_len):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]   # (B, gp_dim)
        x1 = X[:, i1, :]   # (B, gp_dim)

        if t < single_eval_pos:
            # Context: reorder so best is first according to noisy observations
            y0 = Ys[:, i0]  # (B,)
            y1 = Ys[:, i1]  # (B,)

            prefer_x0 = (y0 > y1).unsqueeze(-1)  # (B, 1)

            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)

            # qeubo_regret[:, t] stays 0 for context positions
        else:
            # Query: keep original random order
            f0 = Fs[:, i0]  # (B,)
            f1 = Fs[:, i1]  # (B,)

            first_x = x0
            second_x = x1

            qeubo_val = torch.maximum(f0, f1)
            qeubo_regret[:, t] = (qeubo_val - f_star).clamp(-10.0, 0.0)

        new_X[:, t, :] = torch.cat([first_x, second_x], dim=-1)

    return Batch(
        x=new_X,
        y=qeubo_regret,
        target_y=qeubo_regret,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DqEUBORegretPriorConfig(PriorConfig):
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
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
    """Builds an RBF GP prior with fixed hyperparameters."""
    mean_module = gpytorch.means.ConstantMean()
    mean_module.initialize(constant=mean_constant)

    gp_dim = X.shape[-1]
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=gp_dim)
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
    """Samples latent GP utilities and noisy observed utilities."""
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
    num_features=4,
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
    """
    Samples qEUBO pair-score tasks with signed and absolute difference features.

    Each token is [x_first, x_second, x_first - x_second, |x_first - x_second|].
    Context pairs are ordered as [winner, loser, winner - loser, |winner - loser|].
    Query targets are max(F(x_first), F(x_second)).
    """
    assert num_features % 4 == 0, "Signed-abs-diff tokens must be [x1, x2, x1 - x2, |x1 - x2|]."
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len

    gp_dim = num_features // 4

    # Each token is a pair, so we sample two original GP inputs per token.
    X = torch.rand(batch_size, 2 * seq_len, gp_dim, device=device)

    # Fs is the noiseless utility; Ys is the noisy utility used to order context comparisons.
    Fs, Ys = sample_gp_batch(
        X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    # The model input stores [first point, second point, signed diff, absolute diff].
    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)

    # Context targets stay zero; query targets are qEUBO-style max utility.
    qeubo = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    for t in range(seq_len):
        # Pair t uses original GP points 2t and 2t+1.
        i0 = 2 * t
        i1 = 2 * t + 1

        # x0 and x1 are candidate points in [0, 1]^d.
        x0 = X[:, i0, :]
        x1 = X[:, i1, :]

        if t < single_eval_pos:
            # Context uses noisy preferences and always puts the winner first.
            y0 = Ys[:, i0]
            y1 = Ys[:, i1]
            prefer_x0 = (y0 > y1).unsqueeze(-1)
            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)
        else:
            # Query keeps the random pair order and predicts best latent utility in the pair.
            f0 = Fs[:, i0]
            f1 = Fs[:, i1]
            first_x = x0
            second_x = x1
            qeubo[:, t] = torch.maximum(f0, f1)

        # Signed difference shows direction; absolute difference shows proximity.
        diff_x = first_x - second_x
        abs_diff_x = diff_x.abs()

        # Full token: [x, x', x - x', |x - x'|].
        new_X[:, t, :] = torch.cat([first_x, second_x, diff_x, abs_diff_x], dim=-1)

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPqEUBOSignedAbsDiffPriorConfig(PriorConfig):
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

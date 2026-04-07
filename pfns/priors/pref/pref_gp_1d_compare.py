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

    # Need: 2 * seq_len for pairwise-comparison context / queries
    X = torch.rand(batch_size, 2 * seq_len, gp_dim, device=device)

    Fs, Ys = sample_gp_batch(
        X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    # X:  (B, seq_len + single_eval_pos, 1)
    # Fs: (B, seq_len + single_eval_pos)
    # Ys: (B, seq_len + single_eval_pos)

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    best_index = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    # new X --> pair t uses original indices (2t, 2t+1)
    # context: re-order such that best is always first + best_index = 0 (based on noisy observations)
    # query: retain random order, set best_index 1 if second is better (based on noiseless ground-truth)
    for t in range(seq_len):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]   # (B, gp_dim)
        x1 = X[:, i1, :]   # (B, gp_dim)


        if t < single_eval_pos:
            # Context: reorder so best is first, target stays 0 (noisy observations)
            y0 = Ys[:, i0]     # (B,)
            y1 = Ys[:, i1]     # (B,)
            
            prefer_x0 = (y0 > y1).unsqueeze(-1)  # (B, 1)

            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)

            # best_index[:, t] remains 0
        else:
            # Query: keep original order, target is 1 iff second is better (ground truth)
            f0 = Fs[:, i0]     # (B,)
            f1 = Fs[:, i1]     # (B,)
        
            first_x = x0
            second_x = x1
            best_index[:, t] = (f1 > f0).to(Fs.dtype)

        new_X[:, t, :] = torch.cat([first_x, second_x], dim=-1)

    # As requested: zero out targets on context positions
    return Batch(
        x=new_X,
        y=best_index,
        target_y=best_index,
        single_eval_pos=single_eval_pos,
    )

@dataclass(frozen=True)
class PrefGP1DComparePriorConfig(PriorConfig):
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
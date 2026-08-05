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
    num_references=100,
    **kwargs,
):
    assert num_features == 2, "pref_gp_1d only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert num_references > 0

    gp_dim = num_features // 2  # = 1

    # Need:
    # - num_references reference points
    # - 2 * single_eval_pos points for pairwise-comparison context
    # - (seq_len - single_eval_pos) points for percentile queries
    # Total non-reference points = seq_len + single_eval_pos
    X_all = torch.rand(
        batch_size,
        num_references + seq_len + single_eval_pos,
        gp_dim,
        device=device,
    )

    Fs_all, Ys_all = sample_gp_batch(
        X_all,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    # Split into references and the sequence tokens
    ref_Fs = Fs_all[:, :num_references]          # (B, N)
    ref_Ys = Ys_all[:, :num_references]          # optional, if you want noisy references instead

    X = X_all[:, num_references:, :]             # (B, seq_len + single_eval_pos, 1)
    Fs = Fs_all[:, num_references:]              # (B, seq_len + single_eval_pos)
    Ys = Ys_all[:, num_references:]              # (B, seq_len + single_eval_pos)

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    target_perc = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    # First single_eval_pos tokens: pairwise preference context
    # token t uses original indices (2t, 2t+1)
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

        # Store [preferred, non-preferred]
        new_X[:, t, :] = torch.cat([best_x, worst_x], dim=-1)

    # Remaining tokens: percentile queries [x, 0]
    num_queries = seq_len - single_eval_pos
    if num_queries > 0:
        src_start = 2 * single_eval_pos
        src_end = src_start + num_queries

        query_X = X[:, src_start:src_end, :]    # (B, Q, 1)
        query_Fs = Fs[:, src_start:src_end]     # (B, Q)

        # Put query x into first coordinate, leave second coordinate zero
        new_X[:, single_eval_pos:, :gp_dim] = query_X

        # Empirical percentile target: k / N where
        # k = number of reference f-values strictly smaller than query f
        k = (ref_Fs.unsqueeze(1) < query_Fs.unsqueeze(-1)).sum(dim=-1)  # (B, Q)
        target_perc[:, single_eval_pos:] = k.to(Fs.dtype) / num_references

    return Batch(
        x=new_X,
        y=target_perc,
        target_y=target_perc,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DPercentilePriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    num_references: int = 100
    

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            num_references=self.num_references
        )
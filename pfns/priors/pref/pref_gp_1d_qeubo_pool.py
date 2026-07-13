from dataclasses import dataclass
from functools import partial

import torch

from pfns.priors import Batch
from pfns.priors.pref.pref_gp_1d_qeubo import make_gp_prior, sample_gp_batch
from pfns.priors.prior import PriorConfig


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
    pool_size=100,
    **kwargs,
):
    assert num_features >= 2 and num_features % 2 == 0
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 2

    gp_dim = num_features // 2

    pool_X = torch.rand(batch_size, pool_size, gp_dim, device=device)
    pool_Fs, pool_Ys = sample_gp_batch(
        pool_X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=noise_std,
        jitter=jitter,
    )

    point_indices = torch.randint(
        pool_size,
        (batch_size, 2 * seq_len),
        device=device,
    )
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    X = pool_X[batch_indices, point_indices]
    Fs = pool_Fs[batch_indices, point_indices]
    Ys = pool_Ys[batch_indices, point_indices]

    new_X = torch.zeros(
        batch_size,
        seq_len,
        num_features,
        device=device,
        dtype=X.dtype,
    )
    qeubo = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    for t in range(seq_len):
        i0 = 2 * t
        i1 = 2 * t + 1

        x0 = X[:, i0, :]
        x1 = X[:, i1, :]

        if t < single_eval_pos:
            y0 = Ys[:, i0]
            y1 = Ys[:, i1]
            prefer_x0 = (y0 > y1).unsqueeze(-1)
            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)
        else:
            first_x = x0
            second_x = x1
            qeubo[:, t] = torch.maximum(Fs[:, i0], Fs[:, i1])

        new_X[:, t, :] = torch.cat([first_x, second_x], dim=-1)

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DqEUBOPoolPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    pool_size: int = 100

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            pool_size=self.pool_size,
        )

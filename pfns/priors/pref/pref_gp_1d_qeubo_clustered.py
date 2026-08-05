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


def _sample_log_uniform_int_01_1000(device="cpu") -> int:
    """
    Sample N by:
      exp(U), U ~ Uniform(log(0.1), log(1000))
    then round and clamp to {1, ..., 1000}.

    This naturally gives substantial mass to N=1, then decreasing mass on larger scales.
    """
    low, high = 0.1, 1000.0
    u = torch.rand((), device=device)
    val = torch.exp(
        torch.log(torch.tensor(low, device=device))
        + u * (math.log(high) - math.log(low))
    )
    return int(torch.clamp(torch.round(val), min=1, max=1000).item())


def _sample_points_with_endpoints(
    batch_size: int,
    n_points: int,
    gp_dim: int,
    device,
    dtype,
    endpoint_prob: float = 0.05,
):
    """
    Sample points in [0,1], with small probability to snap each scalar coordinate
    exactly to 0 or 1.
    """
    X = torch.rand(batch_size, n_points, gp_dim, device=device, dtype=dtype)

    if endpoint_prob > 0:
        u = torch.rand_like(X)
        X = torch.where(u < (endpoint_prob / 2), torch.zeros_like(X), X)
        X = torch.where(
            (u >= (endpoint_prob / 2)) & (u < endpoint_prob),
            torch.ones_like(X),
            X,
        )

    return X


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
    endpoint_prob=0.05,
    qeubo_model=None,
    **kwargs,
):
    """
    Persistent-pool qEUBO context generation.

    Context generation:
      - sample one N per batch call from LogUniform(0.1, 1000), round/clamp to {1,...,1000}
      - initialize a persistent pool of N candidate pairs
      - for each context step:
          * score current pool with qeubo_model
          * pick argmax per batch element
          * reveal noisy winner using fresh comparison noise
          * add ordered (better, worse) pair to context
          * replace the selected pool entry with one fresh pair
      - total unique candidate pairs used for context generation:
          N + max(single_eval_pos - 1, 0)

    Query generation:
      - use remaining sampled x's as random pairs
      - target is max(f0, f1) from the latent GP

    Noise model:
      - latent GP values f are sampled once
      - noisy comparison observations are resampled fresh only for selected context pairs
    """
    if qeubo_model is None and "model" in kwargs:
        qeubo_model = kwargs["model"]
    if qeubo_model is None and "quebo_model" in kwargs:
        qeubo_model = kwargs["quebo_model"]

    assert num_features == 2, "pref_gp_1d_qeubo_clustered only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert qeubo_model is not None, "Pass qeubo_model (or model) to get_batch."

    gp_dim = num_features // 2
    dtype = torch.get_default_dtype()

    # Sample one N per batch call
    N = _sample_log_uniform_int_01_1000(device=device)
    print(N)

    n_query = seq_len - single_eval_pos
    n_replacements = max(single_eval_pos - 1, 0)
    total_context_candidate_pairs = N + n_replacements
    total_points = 2 * total_context_candidate_pairs + 2 * n_query

    # Sample all x's once
    X = _sample_points_with_endpoints(
        batch_size=batch_size,
        n_points=total_points,
        gp_dim=gp_dim,
        device=device,
        dtype=dtype,
        endpoint_prob=endpoint_prob,
    )

    # Sample latent GP once; no global noisy Ys
    Fs, _ = sample_gp_batch(
        X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None,
        jitter=jitter,
    )

    # Layout:
    #   first  2*N                         -> initial persistent pool
    #   next   2*(single_eval_pos - 1)    -> replacement reservoir
    #   final  2*n_query                  -> query pairs
    init_pool_X = X[:, : 2 * N, :].view(batch_size, N, 2 * gp_dim)         # (B, N, 2)
    init_pool_Fs = Fs[:, : 2 * N].view(batch_size, N, 2)                   # (B, N, 2)

    if n_replacements > 0:
        repl_start = 2 * N
        repl_end = repl_start + 2 * n_replacements
        repl_X = X[:, repl_start:repl_end, :].view(batch_size, n_replacements, 2 * gp_dim)  # (B, R, 2)
        repl_Fs = Fs[:, repl_start:repl_end].view(batch_size, n_replacements, 2)             # (B, R, 2)
    else:
        repl_X = torch.zeros(batch_size, 0, 2 * gp_dim, device=device, dtype=dtype)
        repl_Fs = torch.zeros(batch_size, 0, 2, device=device, dtype=dtype)

    if n_query > 0:
        query_start = 2 * total_context_candidate_pairs
        query_X = X[:, query_start:, :].view(batch_size, n_query, 2 * gp_dim)  # (B, Q, 2)
        query_Fs = Fs[:, query_start:].view(batch_size, n_query, 2)             # (B, Q, 2)
    else:
        query_X = torch.zeros(batch_size, 0, 2 * gp_dim, device=device, dtype=dtype)
        query_Fs = torch.zeros(batch_size, 0, 2, device=device, dtype=dtype)

    # Mutable current pool
    pool_X = init_pool_X.clone()    # (B, N, 2)
    pool_Fs = init_pool_Fs.clone()  # (B, N, 2)

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=dtype)
    qeubo = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)

    # Use eval/no_grad while scoring candidate pools
    use_model = N > 1 and single_eval_pos > 0
    if use_model:
        was_training = qeubo_model.training
        qeubo_model.eval()
    else:
        was_training = None

    with torch.no_grad():
        batch_idx = torch.arange(batch_size, device=device)

        for t in range(single_eval_pos):
            if N == 1:
                best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                x_ctx = new_X[:, :t, :]
                y_ctx = qeubo[:, :t]  # context targets remain zero

                logits = qeubo_model(x_ctx, y_ctx, test_x=pool_X)
                scores = qeubo_model.criterion.mean(logits)

                if scores.ndim == 1:
                    scores = scores.unsqueeze(0)
                elif scores.ndim == 3 and scores.shape[-1] == 1:
                    scores = scores[..., 0]

                best_idx = torch.argmax(scores, dim=1)  # (B,)

            chosen_pairs = pool_X[batch_idx, best_idx]   # (B, 2)
            chosen_Fs = pool_Fs[batch_idx, best_idx]     # (B, 2)

            x0 = chosen_pairs[:, :gp_dim]
            x1 = chosen_pairs[:, gp_dim:]

            f0 = chosen_Fs[:, 0]
            f1 = chosen_Fs[:, 1]

            # Fresh noise per comparison event
            if noise_std is None or noise_std == 0:
                y0 = f0
                y1 = f1
            else:
                y0 = f0 + noise_std * torch.randn_like(f0)
                y1 = f1 + noise_std * torch.randn_like(f1)

            prefer_x0 = (y0 > y1).unsqueeze(-1)  # (B, 1)

            first_x = torch.where(prefer_x0, x0, x1)
            second_x = torch.where(prefer_x0, x1, x0)

            new_X[:, t, :] = torch.cat([first_x, second_x], dim=-1)
            # qeubo[:, t] remains zero on context positions

            # Replace selected pool entry with fresh pair, except after final context step
            if t < single_eval_pos - 1:
                repl_idx = t  # use replacement pair t
                pool_X[batch_idx, best_idx] = repl_X[:, repl_idx, :]
                pool_Fs[batch_idx, best_idx] = repl_Fs[:, repl_idx, :]

    if was_training:
        qeubo_model.train()

    # Queries + qEUBO targets
    if n_query > 0:
        new_X[:, single_eval_pos:, :] = query_X
        qeubo[:, single_eval_pos:] = torch.maximum(query_Fs[:, :, 0], query_Fs[:, :, 1])

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGP1DqEUBOPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    endpoint_prob: float = 0.05

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            endpoint_prob=self.endpoint_prob,
        )
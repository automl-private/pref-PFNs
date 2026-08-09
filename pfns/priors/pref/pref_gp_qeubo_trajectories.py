from dataclasses import dataclass
from functools import partial

import gpytorch
import torch

from evaluation.agents.qeubo_agent import QEUBOAgent
from evaluation.agents.base import candidate_value
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
    mean_module = gpytorch.means.ConstantMean().to(device=X.device, dtype=X.dtype)
    mean_module.initialize(constant=mean_constant)

    base_kernel = gpytorch.kernels.RBFKernel().to(device=X.device, dtype=X.dtype)
    base_kernel.lengthscale = lengthscale

    covar_module = gpytorch.kernels.ScaleKernel(base_kernel).to(
        device=X.device,
        dtype=X.dtype,
    )
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


def evaluate_rff(
    points,
    weights,
    phases,
    coefficients,
    scale,
    mean_constant=0.0,
):
    """Evaluate one sampled RFF function at arbitrary points."""
    points = points.reshape(-1, weights.shape[0])
    features = torch.cos(points @ weights + phases)
    return mean_constant + scale * (features @ coefficients)


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
    n_init=1,
    support="grid",
    rff_num_features=4096,
    **kwargs,
):
    assert num_features >= 2 and num_features % 2 == 0
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert n_init >= 0
    assert support in {"grid", "continuous_rff"}

    gp_dim = num_features // 2

    # Need 2 * seq_len original GP inputs, since each token is a pair
    X = torch.rand(batch_size, 2 * seq_len, gp_dim, device=device) # TODO: add pool size instead of seq_len part, compare to points from final population

    if support == "grid":
        Fs, Ys = sample_gp_batch(
            X,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            noise_std=noise_std,
            jitter=jitter,
        )
    else:
        assert rff_num_features > 0
        rff_weights = (
            torch.randn(batch_size, gp_dim, rff_num_features, device=device)
            / lengthscale
        )
        rff_phases = 2 * torch.pi * torch.rand(
            batch_size, rff_num_features, device=device
        )
        rff_coefficients = torch.randn(
            batch_size, rff_num_features, device=device
        )
        rff_scale = (2 * outputscale / rff_num_features) ** 0.5

        Fs = torch.stack(
            [
                evaluate_rff(
                    X[batch_index],
                    rff_weights[batch_index],
                    rff_phases[batch_index],
                    rff_coefficients[batch_index],
                    rff_scale,
                    mean_constant,
                )
                for batch_index in range(batch_size)
            ]
        )

    # X:  (B, 2 * seq_len, gp_dim)
    # Fs: (B, 2 * seq_len)
    # Ys: (B, 2 * seq_len), grid support only

    new_X = torch.zeros(batch_size, seq_len, num_features, device=device, dtype=X.dtype)
    qeubo = torch.zeros(batch_size, seq_len, device=device, dtype=Fs.dtype)
    agent = QEUBOAgent(
        fit_hyperparams=False,
        device=device,
        support=support,
        gp_lengthscale=lengthscale,
        gp_outputscale=outputscale,
    )

    if support == "grid":
        comparisons = torch.empty(
            batch_size,
            0,
            2,
            dtype=torch.long,
            device=device,
        )
        batch_indices = torch.arange(batch_size, device=device)
        for t in range(single_eval_pos):
            if t < n_init:
                pair_indices = torch.rand(
                    batch_size,
                    len(X[0]),
                    device=device,
                ).topk(2, dim=-1).indices
            else:
                pair_indices = agent.suggest_pairs_batched_grid(X, comparisons)

            observed = Ys.gather(1, pair_indices)
            prefer_second = observed[:, 1] > observed[:, 0]
            ordered_indices = pair_indices.clone()
            ordered_indices[prefer_second] = ordered_indices[prefer_second].flip(-1)
            comparisons = torch.cat(
                [comparisons, ordered_indices.unsqueeze(1)],
                dim=1,
            )
            pair = X[
                batch_indices.unsqueeze(-1),
                ordered_indices,
            ]
            new_X[:, t] = pair.reshape(batch_size, num_features)
    else:
        comparisons = [[] for _ in range(batch_size)]
        for t in range(single_eval_pos):
            for batch_index in range(batch_size):
                candidate_pool = X[batch_index]
                if t < n_init:
                    pair = candidate_pool[
                        torch.randperm(len(candidate_pool), device=device)[:2]
                    ]
                else:
                    x0, x1 = agent.suggest_pair(
                        comparisons[batch_index], candidate_pool
                    )
                    pair = torch.tensor([x0, x1], device=device, dtype=X.dtype)

                observed = evaluate_rff(
                    pair,
                    rff_weights[batch_index],
                    rff_phases[batch_index],
                    rff_coefficients[batch_index],
                    rff_scale,
                    mean_constant,
                )
                observed = observed + noise_std * torch.randn_like(observed)
                prefer_second = observed[1] > observed[0]

                if prefer_second:
                    pair = pair.flip(0)

                comparisons[batch_index].append(
                    (candidate_value(pair[0]), candidate_value(pair[1]))
                )
                new_X[batch_index, t] = pair.reshape(-1)

    query_pairs = X.reshape(batch_size, seq_len, 2, gp_dim)
    query_values = Fs.reshape(batch_size, seq_len, 2)
    new_X[:, single_eval_pos:] = query_pairs[:, single_eval_pos:].reshape(
        batch_size,
        seq_len - single_eval_pos,
        num_features,
    )
    qeubo[:, single_eval_pos:] = query_values[:, single_eval_pos:].max(dim=-1).values

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )
# g(x, x')
# classification whether the 1st element is better or not


@dataclass(frozen=True)
class PrefGP1DqEUBOPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    n_init: int = 1
    support: str = "grid"
    rff_num_features: int = 4096

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            n_init=self.n_init,
            support=self.support,
            rff_num_features=self.rff_num_features,
        )

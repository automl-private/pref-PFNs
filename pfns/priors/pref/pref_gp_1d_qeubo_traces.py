from dataclasses import dataclass
from functools import partial
import math

import gpytorch
import torch

from pfns.priors import Batch
from pfns.priors.prior import PriorConfig

torch.set_default_dtype(torch.double)


# ============================================================
# GP helpers
# ============================================================
def make_gp_prior(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    jitter=1e-6,
):
    device = X.device
    dtype = X.dtype

    mean_module = gpytorch.means.ConstantMean().to(device=device, dtype=dtype)
    mean_module.initialize(constant=mean_constant)

    base_kernel = gpytorch.kernels.RBFKernel().to(device=device, dtype=dtype)
    base_kernel.lengthscale = torch.as_tensor(lengthscale, device=device, dtype=dtype)

    covar_module = gpytorch.kernels.ScaleKernel(base_kernel).to(device=device, dtype=dtype)
    covar_module.outputscale = torch.as_tensor(outputscale, device=device, dtype=dtype)

    mvn = gpytorch.distributions.MultivariateNormal(
        mean_module(X),
        covar_module(X),
    )
    return mvn.add_jitter(jitter)


def sample_gp_batch(
    X,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=None,
    jitter=1e-6,
):
    """
    Original helper, kept for completeness.
    X: (B, T, D)
    returns:
        f, y: (B, T)
    """
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


def sample_gp_shared_x_batch(
    X_shared,
    batch_size,
    lengthscale,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=None,
    jitter=1e-6,
):
    """
    Efficient helper for the vectorized cached rollout setup.

    X_shared: (T, D), shared x-locations for the whole batch
    returns:
        f, y: (B, T), independent draws for each batch element
    """
    with torch.no_grad():
        f_dist = make_gp_prior(
            X_shared,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            jitter=jitter,
        )

        f = f_dist.rsample(sample_shape=torch.Size([batch_size]))  # (B, T)

        if noise_std is None:
            y = f
        else:
            y = f + noise_std * torch.randn_like(f)

    return f.detach(), y.detach()


# ============================================================
# Sampling helpers
# ============================================================
def _sample_log_uniform_int_01_1000(device="cpu") -> int:
    """
    Sample N by:
      exp(U), U ~ Uniform(log(0.1), log(1000))
    then round and clamp to {1, ..., 1000}.
    """
    low, high = 0.1, 1000.0
    u = torch.rand((), device=device)
    val = torch.exp(
        torch.log(torch.tensor(low, device=device))
        + u * (math.log(high) - math.log(low))
    )
    return int(torch.clamp(torch.round(val), min=1, max=1000).item())


def _sample_points_with_endpoints(
    n_points: int,
    gp_dim: int,
    device,
    dtype,
    endpoint_prob: float = 0.05,
):
    """
    Sample shared points in [0,1], with small probability to snap each scalar coordinate
    exactly to 0 or 1.
    Returns:
        X: (T, D)
    """
    X = torch.rand(n_points, gp_dim, device=device, dtype=dtype)

    if endpoint_prob > 0:
        u = torch.rand_like(X)
        X = torch.where(u < (endpoint_prob / 2), torch.zeros_like(X), X)
        X = torch.where(
            (u >= (endpoint_prob / 2)) & (u < endpoint_prob),
            torch.ones_like(X),
            X,
        )

    return X


# ============================================================
# Cached batched rollout state
# ============================================================
def _init_state_batch(
    batch_size,
    seq_len,
    max_context_size,
    gp_dim,
    device,
    dtype,
    lengthscale,
    outputscale,
    mean_constant,
    noise_std,
    jitter,
    endpoint_prob,
):
    """
    Initialize one fresh batched rollout state at depth 0.

    Important simplification:
      - x-locations are shared across batch elements
      - latent GP function samples are independent across batch elements
    """
    N = _sample_log_uniform_int_01_1000(device=device)

    n_replacements = max(max_context_size - 1, 0)
    n_query_reservoir = seq_len  # enough for any emitted depth 0..max_context_size

    total_context_candidate_pairs = N + n_replacements
    total_pairs = total_context_candidate_pairs + n_query_reservoir
    total_points = 2 * total_pairs

    # Shared x-locations across the batch
    X_shared = _sample_points_with_endpoints(
        n_points=total_points,
        gp_dim=gp_dim,
        device=device,
        dtype=dtype,
        endpoint_prob=endpoint_prob,
    )  # (2 * total_pairs, gp_dim)

    # Independent latent function draws per batch element on the shared grid
    Fs, _ = sample_gp_shared_x_batch(
        X_shared,
        batch_size=batch_size,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None,
        jitter=jitter,
    )  # (B, 2 * total_pairs)

    # Expand shared X to batch only when storing state tensors
    X_batched = X_shared.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2 * total_pairs, gp_dim)

    # Initial persistent pool
    pool_X = X_batched[:, : 2 * N, :].view(batch_size, N, 2 * gp_dim)  # (B, N, 2)
    pool_Fs = Fs[:, : 2 * N].view(batch_size, N, 2)                    # (B, N, 2)

    # Replacement reservoir
    if n_replacements > 0:
        repl_start = 2 * N
        repl_end = repl_start + 2 * n_replacements

        repl_X = X_batched[:, repl_start:repl_end, :].view(batch_size, n_replacements, 2 * gp_dim)
        repl_Fs = Fs[:, repl_start:repl_end].view(batch_size, n_replacements, 2)
    else:
        repl_X = torch.zeros(batch_size, 0, 2 * gp_dim, device=device, dtype=dtype)
        repl_Fs = torch.zeros(batch_size, 0, 2, device=device, dtype=dtype)

    # Query reservoir (always length seq_len; later we take prefix seq_len - depth)
    query_start = 2 * total_context_candidate_pairs
    query_X = X_batched[:, query_start:, :].view(batch_size, n_query_reservoir, 2 * gp_dim)
    query_Fs = Fs[:, query_start:].view(batch_size, n_query_reservoir, 2)

    state = {
        "depth": 0,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "N": N,
        "pool_X": pool_X,
        "pool_Fs": pool_Fs,
        "repl_X": repl_X,
        "repl_Fs": repl_Fs,
        "query_X": query_X,
        "query_Fs": query_Fs,
        "ctx_X": torch.zeros(batch_size, 0, 2 * gp_dim, device=device, dtype=dtype),
    }
    return state


def _materialize_state_batch(state, seq_len, device, dtype):
    """
    Turn one batched rollout state into one training Batch payload.
    Returns:
        x: (B, seq_len, 2)
        y: (B, seq_len)
    """
    depth = state["depth"]
    batch_size = state["batch_size"]

    x = torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype)
    y = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    # context
    if depth > 0:
        x[:, :depth, :] = state["ctx_X"]

    # queries
    n_query = seq_len - depth
    if n_query > 0:
        qx = state["query_X"][:, :n_query, :]
        qf = state["query_Fs"][:, :n_query, :]
        x[:, depth:, :] = qx
        y[:, depth:] = torch.maximum(qf[:, :, 0], qf[:, :, 1])

    return x, y


def _extend_state_batch(
    state,
    qeubo_model,
    max_context_size,
    noise_std,
    device,
    dtype,
):
    """
    Extend one batched state by one context step.
    Returns:
        successor state, or None if already terminal.
    """
    depth = state["depth"]
    if depth >= max_context_size:
        return None

    batch_size = state["batch_size"]
    N = state["N"]

    pool_X = state["pool_X"]   # (B, N, 2)
    pool_Fs = state["pool_Fs"] # (B, N, 2)

    if N == 1:
        best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        x_ctx = state["ctx_X"]  # (B, depth, 2)
        y_ctx = torch.zeros(batch_size, depth, device=device, dtype=dtype)

        with torch.no_grad():
            logits = qeubo_model(x_ctx, y_ctx, test_x=pool_X)
            scores = qeubo_model.criterion.mean(logits)

            if scores.ndim == 1:
                scores = scores.unsqueeze(0)
            elif scores.ndim == 3 and scores.shape[-1] == 1:
                scores = scores[..., 0]

            best_idx = torch.argmax(scores, dim=1)  # (B,)

    batch_idx = torch.arange(batch_size, device=device)

    chosen_pairs = pool_X[batch_idx, best_idx]  # (B, 2)
    chosen_fs = pool_Fs[batch_idx, best_idx]    # (B, 2)

    x0 = chosen_pairs[:, :1]
    x1 = chosen_pairs[:, 1:]

    f0 = chosen_fs[:, 0]
    f1 = chosen_fs[:, 1]

    # fresh comparison noise per selected comparison event
    if noise_std is None or noise_std == 0:
        y0 = f0
        y1 = f1
    else:
        y0 = f0 + noise_std * torch.randn(batch_size, device=device, dtype=dtype)
        y1 = f1 + noise_std * torch.randn(batch_size, device=device, dtype=dtype)

    prefer_first = (y0 > y1).unsqueeze(-1)  # (B, 1)

    ordered = torch.cat(
        [
            torch.where(prefer_first, x0, x1),
            torch.where(prefer_first, x1, x0),
        ],
        dim=1,
    )  # (B, 2)

    new_ctx_X = torch.cat([state["ctx_X"], ordered.unsqueeze(1)], dim=1)  # (B, depth+1, 2)

    next_state = {
        "depth": depth + 1,
        "batch_size": batch_size,
        "seq_len": state["seq_len"],
        "N": N,
        "pool_X": pool_X.clone(),
        "pool_Fs": pool_Fs.clone(),
        "repl_X": state["repl_X"],
        "repl_Fs": state["repl_Fs"],
        "query_X": state["query_X"],
        "query_Fs": state["query_Fs"],
        "ctx_X": new_ctx_X,
    }

    # replace selected pool entry with the next fresh replacement, unless terminal
    if depth < max_context_size - 1:
        repl_idx = depth  # replacement reservoir is indexed by current depth
        next_state["pool_X"][batch_idx, best_idx] = state["repl_X"][:, repl_idx, :]
        next_state["pool_Fs"][batch_idx, best_idx] = state["repl_Fs"][:, repl_idx, :]

    return next_state


# ============================================================
# Main prior entry point
# ============================================================
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
    endpoint_prob=0.05,
    qeubo_model=None,
    rollout_state_caches=None,
    **kwargs,
):
    """
    Cached batched rollout version.

    Interpretation:
      - input single_eval_pos is treated as MAX context size
      - actual returned context size is chosen internally from {0} U available cached depths
      - Batch.single_eval_pos is set to that chosen depth

    rollout_state_caches:
      - dict(depth -> batched rollout state)
      - persisted by the data loader between calls
    """
    if qeubo_model is None and "model" in kwargs:
        qeubo_model = kwargs["model"]
    if qeubo_model is None and "quebo_model" in kwargs:
        qeubo_model = kwargs["quebo_model"]

    # make sure to create tensors on the same device where the model lives
    model_device = next(qeubo_model.parameters()).device
    device = model_device

    assert num_features == 2, "pref_gp_1d_qeubo_clustered only supports num_features=2"
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos < seq_len
    assert qeubo_model is not None, "Pass qeubo_model (or model) to get_batch."
    single_eval_pos = 99 # TODO: FIX THIS WEIRDNESS???

    gp_dim = num_features // 2
    dtype = torch.get_default_dtype()
    max_context_size = int(single_eval_pos)  # should be set to max training size!

    # initialize / validate cache
    if rollout_state_caches is None:
        raise RuntimeError("rollout_state_caches must be a persistent dict, got None")
    
    if not isinstance(rollout_state_caches, dict):
        raise RuntimeError(f"rollout_state_caches must be a dict, got {type(rollout_state_caches)}")
    
    if len(rollout_state_caches) > 0:
        any_state = next(iter(rollout_state_caches.values()))
    
        if any_state["batch_size"] != batch_size:
            raise RuntimeError(
                f"rollout_state_caches batch_size mismatch: "
                f"cache has {any_state['batch_size']}, get_batch got {batch_size}"
            )
    
        if any_state["seq_len"] != seq_len:
            raise RuntimeError(
                f"rollout_state_caches seq_len mismatch: "
                f"cache has {any_state['seq_len']}, get_batch got {seq_len}"
            )
    
        cached_device = any_state["ctx_X"].device
        if cached_device != device:
            raise RuntimeError(
                f"rollout_state_caches device mismatch: "
                f"cache has {cached_device}, get_batch uses {device}"
            )

    # select a state with probability that decays exponentially with context size 
    available_depths = sorted(d for d in rollout_state_caches.keys() if d <= max_context_size)
    choices = [0] + available_depths
    
    B = 0.9
    weights = torch.tensor([B ** d for d in choices], device=device, dtype=torch.double)
    probs = weights / weights.sum()
    
    choice_idx = torch.multinomial(probs, num_samples=1).item()
    chosen_depth = int(choices[choice_idx])
    print(f"{chosen_depth}", end=' ')

    # consume or create state
    if chosen_depth == 0:
        state = _init_state_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            max_context_size=max_context_size,
            gp_dim=gp_dim,
            device=device,
            dtype=dtype,
            lengthscale=lengthscale,
            outputscale=outputscale,
            mean_constant=mean_constant,
            noise_std=noise_std,
            jitter=jitter,
            endpoint_prob=endpoint_prob,
        )
    else:
        state = rollout_state_caches.pop(chosen_depth)

    # emit current state as batch
    x, y = _materialize_state_batch(state, seq_len=seq_len, device=device, dtype=dtype)

    # extend once and store only successor
    if chosen_depth < max_context_size:
        was_training = qeubo_model.training
        qeubo_model.eval()
    
        next_state = _extend_state_batch(
            state,
            qeubo_model=qeubo_model,
            max_context_size=max_context_size,
            noise_std=noise_std,
            device=device,
            dtype=dtype,
        )
    
        if was_training:
            qeubo_model.train()
    
        if next_state is not None:
            rollout_state_caches[next_state["depth"]] = next_state  # overwrite with more recent

    return Batch(
        x=x,
        y=y,
        target_y=y,
        single_eval_pos=chosen_depth,
    )


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class PrefGP1DqEUBOTracesPriorConfig(PriorConfig):
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
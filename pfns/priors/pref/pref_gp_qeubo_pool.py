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
    gp_dim = X.shape[-1]

    mean_module = gpytorch.means.ConstantMean()
    mean_module.initialize(constant=mean_constant)

    # ARD-capable RBF kernel.
    # - scalar lengthscale -> isotropic
    # - lengthscale with shape (gp_dim,) -> ARD
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=gp_dim)

    lengthscale = torch.as_tensor(
        lengthscale,
        dtype=X.dtype,
        device=X.device,
    )

    if lengthscale.ndim == 0:
        # isotropic case
        base_kernel.lengthscale = lengthscale
    else:
        # ARD case
        assert lengthscale.numel() == gp_dim, (
            f"ARD lengthscale must have {gp_dim} entries, "
            f"got {lengthscale.numel()}."
        )
        base_kernel.lengthscale = lengthscale.reshape(1, gp_dim)

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


def _incumbent_duel_context(Fs, Ys, B, K, n_ctx, device, *, noise_std,
                            noise_per_comparison, incumbent_prob):
    """King-of-the-hill contexts: a running champion is repeatedly challenged.

    Motivation (`research-backlog.md` item 7). Uniformly sampled pairs give every pool point the
    same expected degree, ~2*n_ctx/K, so the comparison graph is a sparse near-forest. A real
    preferential-BO loop looks nothing like that: it keeps an incumbent and duels it against new
    candidates, so one point's degree grows *linearly* while most points are never touched, and
    the comparisons concentrate on the high-utility region as the loop exploits. That structural
    gap -- not context *length*, which the session-4 2x2 varied and which changed nothing -- is
    the untested form of the covariate-shift hypothesis.

    Each step, with probability `incumbent_prob`, the champion is challenged by a uniformly drawn
    other point; otherwise a uniform pair is drawn, and if that pair's winner beats the champion
    it takes the title. `incumbent_prob` is a continuous knob from the uniform prior
    (`0.0`, reproducing `context_policy="uniform"` in distribution) to a pure incumbent duel
    (`1.0`), so this supports a dose-response study rather than an on/off cell.

    Returns `(win_idx, lose_idx)`, each `(B, n_ctx)` -- already resolved, because the next pair
    depends on who won.
    """
    champion = torch.randint(K, size=(B,), device=device)
    win_all = torch.empty((B, n_ctx), dtype=torch.long, device=device)
    lose_all = torch.empty((B, n_ctx), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    for t in range(n_ctx):
        # Uniform pair, used directly on exploration steps and as the challenger otherwise.
        a = torch.randint(K, size=(B,), device=device)
        b = torch.randint(K - 1, size=(B,), device=device)
        b = b + (b >= a).long()

        use_inc = torch.rand(B, device=device) < incumbent_prob
        # Challenger must differ from the champion: reuse `a`, nudged off the champion.
        chal = torch.randint(K - 1, size=(B,), device=device)
        chal = chal + (chal >= champion).long()

        i = torch.where(use_inc, champion, a)
        j = torch.where(use_inc, chal, b)

        if noise_per_comparison and noise_std:
            yi = Fs[batch_idx, i] + noise_std * torch.randn(B, device=device, dtype=Fs.dtype)
            yj = Fs[batch_idx, j] + noise_std * torch.randn(B, device=device, dtype=Fs.dtype)
        elif noise_per_comparison:
            yi, yj = Fs[batch_idx, i], Fs[batch_idx, j]
        else:
            yi, yj = Ys[batch_idx, i], Ys[batch_idx, j]

        i_wins = yi > yj
        win_all[:, t] = torch.where(i_wins, i, j)
        lose_all[:, t] = torch.where(i_wins, j, i)
        # The winner holds the title. On an incumbent step this keeps the champion unless it was
        # beaten; on an exploration step a strong newcomer can take over, which is what makes the
        # incumbent drift toward the optimum instead of being frozen at its random start.
        champion = win_all[:, t]

    return win_all, lose_all


def _elo_softmax_context(Fs, Ys, B, K, n_ctx, device, *, noise_std, noise_per_comparison,
                         elo_k, T_ctx, exclude_seen_pairs=False):
    """Elo-ranked softmax contexts: pairs are drawn by softmax over running Elo ratings.

    Ported from `scripts/prior_lab.py`, which is the specification; a parity test lives in
    `scripts/test_elo_prior_parity.py`. Design rationale and the measurements behind every
    constant are in `.claude/design-decisions.md` ("The first prior-training spec").

    Why Elo and not a win count (Copeland): the update is scaled by the SURPRISE,
    `e = sigmoid(r_w - r_l)`, so beating an established leader moves a rating by nearly the full
    `k` while beating a weak point moves it by almost nothing. Measured consequence: after beating
    the current leader a point lands at the ~92nd percentile under Elo against the ~80th under
    Copeland, and under a softmax sampler that decides whether a real contender is ever re-compared.

    `T_ctx` is `(B,)` -- ONE temperature per task, drawn U(0,1) by the caller, so a batch spans
    concentrated and near-uniform comparison graphs rather than one regime.

    Returns `(win_idx, lose_idx, elo)`; the ratings are returned because the query sampler needs
    them, and recomputing them there would not match (the trajectory is stochastic).
    """
    elo = torch.zeros((B, K), device=device, dtype=Fs.dtype)
    win_all = torch.empty((B, n_ctx), dtype=torch.long, device=device)
    lose_all = torch.empty((B, n_ctx), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    Tc = T_ctx.clamp_min(1e-3).unsqueeze(1)                       # (B, 1)

    # `exclude_seen_pairs` -- NOW A PARAMETER, AND DEFAULT FALSE. [owner, 2026-08-16]
    #
    # This used to be hardcoded on, which made the trained prior deviate from the agreed spec.
    # The spec is: "for queries, I reject duplicate comparisons, for context, reject self-pairs."
    # Self-pairs are already excluded structurally by sampling without replacement; a REPEATED
    # PAIR in the context was never meant to be banned, and the owner's reason is that repetition
    # is genuinely informative once answers are noisy -- the prior must therefore put some
    # probability on it or the model can never learn when it is worth a slot.
    #
    # The old justification ("at noise_std=0.05 a repeat re-measures a nearly deterministic
    # outcome") is regime-specific and was doing global work. Keep the switch so the earlier
    # checkpoints remain reproducible: every config that produced one pins it to True.
    seen_pair = (torch.zeros((B, K, K), dtype=torch.bool, device=device)
                 if exclude_seen_pairs else None)

    for t in range(n_ctx):
        # Two DISTINCT points per context comparison: a self-comparison carries no information
        # and would waste a slot in the graph. Sampling without replacement enforces this
        # structurally, which is also the owner's rule (self-pairs are rejected in the context and
        # kept in the queries).
        probs = torch.softmax(elo / Tc, dim=1)                    # (B, K)
        pair = torch.multinomial(probs, 2, replacement=False)     # (B, 2)
        i, j = pair[:, 0], pair[:, 1]
        if seen_pair is not None:
            for _ in range(20):                                   # same try budget as the lab
                clash = seen_pair[batch_idx, i, j]
                if not bool(clash.any()):
                    break
                redo = torch.multinomial(probs, 2, replacement=False)
                i = torch.where(clash, redo[:, 0], i)
                j = torch.where(clash, redo[:, 1], j)
            seen_pair[batch_idx, i, j] = True
            seen_pair[batch_idx, j, i] = True

        if noise_per_comparison and noise_std:
            yi = Fs[batch_idx, i] + noise_std * torch.randn(B, device=device, dtype=Fs.dtype)
            yj = Fs[batch_idx, j] + noise_std * torch.randn(B, device=device, dtype=Fs.dtype)
        elif noise_per_comparison:
            yi, yj = Fs[batch_idx, i], Fs[batch_idx, j]
        else:
            yi, yj = Ys[batch_idx, i], Ys[batch_idx, j]

        i_wins = yi > yj
        w = torch.where(i_wins, i, j)
        l = torch.where(i_wins, j, i)
        win_all[:, t] = w
        lose_all[:, t] = l

        # Elo update, antisymmetric so the ratings stay zero-sum
        e = torch.sigmoid(elo[batch_idx, w] - elo[batch_idx, l])
        delta = elo_k * (1.0 - e)
        elo[batch_idx, w] += delta
        elo[batch_idx, l] -= delta

    return win_all, lose_all, elo


def _elo_softmax_queries(elo, B, K, n_query, device, *, n_random_diag):
    """Query pairs by softmax over Elo, with a SEPARATE temperature per candidate.

    One temperature per candidate, not per pair. With a single shared low temperature the query
    set collapses -- measured 0.1 distinct points at K=100, because nearly every draw is a
    self-pair. Per-candidate temperatures give 49.6 distinct points while still placing the leader
    in 84% of pairs; a single temperature cannot buy both.

    SELF-PAIRS ARE KEPT. The recommender is `argmax(diag(qEUBO))` -- on the diagonal
    `qEUBO(x, x) = E[f(x)]`, so the acquisition model doubles as a utility model. If self-pairs
    never appear in training the diagonal is strict extrapolation and inference regret, our primary
    metric, is read off a region the model has never seen.

    `n_random_diag` self-pairs are additionally forced in with UNIFORMLY drawn points. The
    naturally occurring ones sit at Elo percentile 0.971, i.e. always the incumbent, which would
    train the diagonal only where the sampler is already confident; the uniform draw reaches the
    low and middle of the pool, and lands on points absent from the context 19% of the time.
    """
    # Oversample, then keep the first `n_query` DISTINCT pairs per task. Deduplication is part
    # of the spec ("for queries, I reject duplicate comparisons") and is not optional: without it
    # repeated collisions all survive and self-pairs reach 18.2 per task against the lab's 2.5,
    # over-weighting the recommender's diagonal about sevenfold. The parity test caught this.
    M = int(n_query * 3)
    Tq = torch.rand((B, M, 2), device=device, dtype=elo.dtype).clamp_min(1e-3)
    logits = elo[:, None, None, :] / Tq[..., None]              # (B, M, 2, K)
    idx = torch.multinomial(
        torch.softmax(logits, dim=-1).reshape(-1, K), 1
    ).reshape(B, M, 2)
    lo = torch.minimum(idx[..., 0], idx[..., 1])
    hi = torch.maximum(idx[..., 0], idx[..., 1])
    key = lo * K + hi                                            # (B, M)

    srt, _ = torch.sort(key, dim=1)
    first = torch.ones_like(srt, dtype=torch.bool)
    first[:, 1:] = srt[:, 1:] != srt[:, :-1]
    rank = torch.cumsum(first.long(), dim=1) - 1                 # position among distinct keys
    take = first & (rank < n_query)

    out = torch.full((B, n_query), -1, dtype=torch.long, device=device)
    bsel = torch.arange(B, device=device)[:, None].expand_as(rank)
    out[bsel[take], rank[take]] = srt[take]

    # Any task that produced fewer than n_query distinct pairs is topped up uniformly. Measured
    # at K=100 this affects a small minority of tasks; the fallback keeps the tensor shape fixed
    # rather than letting the query count vary across a batch.
    short = out < 0
    if bool(short.any()):
        a = torch.randint(K, size=(B, n_query), device=device)
        b = torch.randint(K, size=(B, n_query), device=device)
        out = torch.where(short, torch.minimum(a, b) * K + torch.maximum(a, b), out)

    idx0_q, idx1_q = out // K, out % K

    if n_random_diag > 0:
        # Overwrite the first `n_random_diag` slots rather than appending, so the query count is
        # exactly `n_query` and the forced diagonal cannot be crowded out.
        m = min(n_random_diag, n_query)
        d = torch.randint(K, size=(B, m), device=device)
        idx0_q[:, :m] = d
        idx1_q[:, :m] = d

    return idx0_q, idx1_q


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=2,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    pool_size=100,          # finite candidate pool size K
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    noise_per_comparison=True,
    context_policy="uniform",
    incumbent_prob=1.0,
    query_policy="uniform",
    elo_k=1.0,
    n_random_diag_queries=0,
    exclude_seen_pairs=False,
    target="value",
    jitter=1e-6,
    **kwargs,
):
    """Preferential qEUBO prior over a finite candidate pool.

    `noise_per_comparison` selects the comparison likelihood:

    - ``True`` (default, standard preferential model): the latent `f` is drawn once on the
      pool and **fresh observation noise is drawn for every comparison**, so
      `P(i > j) = P(f_i + e_i > f_j + e_j)` independently per comparison and repeated or
      chained comparisons can contradict each other. This matches the probit likelihood used
      by BoTorch's `PairwiseGP`, by `pref_gp_1d_qeubo_regret_v3.py`, and by the ground-truth
      sampler in `scripts/pbo_ground_truth.py`.
    - ``False`` (legacy): noise is drawn **once per pool point** and frozen, so every
      comparison is read off a single fixed total order on `y` and contradictions are
      impossible. This was the behaviour of this file before the fix and is retained only so
      the training distribution of the existing `*_pool_*` checkpoints stays reproducible.

    The distinction only has an effect when a pool point takes part in more than one
    comparison, i.e. it grows with the average degree `2 * n_ctx / pool_size` of the
    comparison graph. It is negligible at the historical default (degree <= 1.98) and
    first-order once the degree is raised deliberately.
    """
    assert num_features % 2 == 0, (
        "Preferential GP qEUBO prior expects num_features = 2 * gp_dim, "
        f"got num_features={num_features}."
    )
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 2, "pool_size must be at least 2 for non-diagonal comparisons."

    gp_dim = num_features // 2

    # ------------------------------------------------------------
    # finite candidate pool per batch element
    # ------------------------------------------------------------
    # pool_X: (B, K, gp_dim)
    pool_X = torch.rand(
        batch_size,
        pool_size,
        gp_dim,
        device=device,
    )

    # Sample GP values on the finite pool
    # Fs, Ys: (B, K)
    # With per-comparison noise the pool carries only the latent f; the noise is added later,
    # once per comparison. With the legacy per-point noise it is frozen here.
    Fs, Ys = sample_gp_batch(
        pool_X,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None if noise_per_comparison else noise_std,
        jitter=jitter,
    )

    new_X = torch.zeros(
        batch_size,
        seq_len,
        num_features,
        device=device,
        dtype=pool_X.dtype,
    )
    qeubo = torch.zeros(
        batch_size,
        seq_len,
        device=device,
        dtype=Fs.dtype,
    )

    B = batch_size
    K = pool_size
    batch_idx = torch.arange(B, device=device)

    # ------------------------------------------------------------
    # sample context pairs without replacement within pair
    # ------------------------------------------------------------
    n_ctx = single_eval_pos
    # ONE temperature per task, U(0,1). Drawn even when unused so the RNG stream does not depend
    # on the policy, which keeps a uniform-context control comparable seed-for-seed.
    T_ctx = torch.rand(B, device=device, dtype=Fs.dtype)
    elo_ratings = None
    if n_ctx > 0:
        if context_policy == "uniform":
            idx0_ctx = torch.randint(K, size=(B, n_ctx), device=device)
            idx1_ctx = torch.randint(K - 1, size=(B, n_ctx), device=device)

            # transform idx1 so idx1_ctx != idx0_ctx
            idx1_ctx = idx1_ctx + (idx1_ctx >= idx0_ctx).long()
            x0 = pool_X[batch_idx[:, None], idx0_ctx]  # (B, n_ctx, gp_dim)
            x1 = pool_X[batch_idx[:, None], idx1_ctx]  # (B, n_ctx, gp_dim)

            if noise_per_comparison:
                f0_ctx = Fs[batch_idx[:, None], idx0_ctx]  # (B, n_ctx)
                f1_ctx = Fs[batch_idx[:, None], idx1_ctx]  # (B, n_ctx)

                if noise_std is None or noise_std == 0.0:
                    y0 = f0_ctx
                    y1 = f1_ctx
                else:
                    # Fresh noise per comparison occurrence
                    y0 = f0_ctx + noise_std * torch.randn_like(f0_ctx)
                    y1 = f1_ctx + noise_std * torch.randn_like(f1_ctx)
            else:
                # Legacy: a single frozen noisy value per pool point
                y0 = Ys[batch_idx[:, None], idx0_ctx]  # (B, n_ctx)
                y1 = Ys[batch_idx[:, None], idx1_ctx]  # (B, n_ctx)

            prefer_x0 = (y0 > y1).unsqueeze(-1)        # (B, n_ctx, 1)

            win_idx = torch.where(prefer_x0[..., 0], idx0_ctx, idx1_ctx)
            lose_idx = torch.where(prefer_x0[..., 0], idx1_ctx, idx0_ctx)

        elif context_policy == "elo_softmax":
            # Like the incumbent duel, this generator must resolve each comparison itself: the
            # next pair depends on the ratings, which depend on who won.
            win_idx, lose_idx, elo_ratings = _elo_softmax_context(
                Fs, Ys, B, K, n_ctx, device,
                noise_std=noise_std,
                noise_per_comparison=noise_per_comparison,
                elo_k=elo_k, T_ctx=T_ctx,
                exclude_seen_pairs=exclude_seen_pairs,
            )

        elif context_policy == "incumbent":
            # The duel generator must resolve each comparison itself, because the *next* pair
            # depends on who won. Re-deciding the outcome here would draw fresh noise and could
            # flip it, leaving a "champion" that never actually won -- the structure this policy
            # exists to create would silently not be there.
            win_idx, lose_idx = _incumbent_duel_context(
                Fs, Ys, B, K, n_ctx, device,
                noise_std=noise_std,
                noise_per_comparison=noise_per_comparison,
                incumbent_prob=incumbent_prob,
            )
        else:
            raise ValueError(f"unknown context_policy {context_policy!r}; "
                             f"expected 'uniform', 'incumbent' or 'elo_softmax'")

        new_X[:, :n_ctx, :] = torch.cat(
            [pool_X[batch_idx[:, None], win_idx],
             pool_X[batch_idx[:, None], lose_idx]], dim=-1)

        if query_policy == "elo_softmax" and elo_ratings is None:
            # The context was produced by another policy, so replay its comparisons through the
            # Elo update to obtain ratings. Order matters (the update is path dependent), and
            # `win_idx`/`lose_idx` preserve it. This is what makes the 2x2 ablation well defined:
            # "uniform context + Elo queries" ranks the pool using exactly the comparisons the
            # uniform context happened to produce.
            elo_ratings = torch.zeros((B, K), device=device, dtype=Fs.dtype)
            for t in range(n_ctx):
                w, l = win_idx[:, t], lose_idx[:, t]
                e = torch.sigmoid(elo_ratings[batch_idx, w] - elo_ratings[batch_idx, l])
                d = elo_k * (1.0 - e)
                elo_ratings[batch_idx, w] += d
                elo_ratings[batch_idx, l] -= d

    # ------------------------------------------------------------
    # sample query pairs with replacement
    # ------------------------------------------------------------
    n_query = seq_len - single_eval_pos
    if n_query > 0:
        if query_policy == "uniform":
            # With replacement, so self-pairs arise at rate ~1/K. That is what keeps the
            # recommender's diagonal in-distribution; see `_elo_softmax_queries`.
            idx0_q = torch.randint(K, size=(B, n_query), device=device)
            idx1_q = torch.randint(K, size=(B, n_query), device=device)
            if n_random_diag_queries > 0:
                m = min(n_random_diag_queries, n_query)
                d = torch.randint(K, size=(B, m), device=device)
                idx0_q[:, :m] = d
                idx1_q[:, :m] = d
        elif query_policy == "elo_softmax":
            if elo_ratings is None:
                # n_ctx == 0: no comparisons, so every rating is 0 and the softmax is uniform at
                # any temperature. Fall back explicitly rather than relying on that coincidence.
                idx0_q = torch.randint(K, size=(B, n_query), device=device)
                idx1_q = torch.randint(K, size=(B, n_query), device=device)
            else:
                idx0_q, idx1_q = _elo_softmax_queries(
                    elo_ratings, B, K, n_query, device,
                    n_random_diag=n_random_diag_queries)
        else:
            raise ValueError(f"unknown query_policy {query_policy!r}; "
                             f"expected 'uniform' or 'elo_softmax'")

        x0 = pool_X[batch_idx[:, None], idx0_q]  # (B, n_query, gp_dim)
        x1 = pool_X[batch_idx[:, None], idx1_q]  # (B, n_query, gp_dim)

        f0 = Fs[batch_idx[:, None], idx0_q]      # (B, n_query)
        f1 = Fs[batch_idx[:, None], idx1_q]      # (B, n_query)

        new_X[:, n_ctx:, :] = torch.cat([x0, x1], dim=-1)
        # `target`: "value" regresses max(f0,f1); "pool_regret" regresses max(f0,f1) - max_pool f.
        #
        # Why pool_regret [owner, 2026-08-16]. Under a KNOWN task the regret of any pair containing
        # the pool optimum is exactly 0, so the target looks flat -- but the model predicts under
        # the POSTERIOR, where the optimum is itself uncertain, and there
        #     P(regret ~ 0 | context, pair) = P(this pair contains the optimum).
        # That is exactly qEUBO's exploration term, re-expressed as probability mass near zero
        # instead of as a shift in the mean. Measured on the exact posterior at d=6, that mass
        # varies across incumbent-containing pairs by ~0.004 on a mean of 0.16-0.20 (real, not
        # sampler noise: two-chain correlation 0.75-0.82 at 24k draws), while the corresponding
        # difference in the MEAN is 0.016 -- a third of one bin on the uniform head the elo cells
        # use. The regret configs' log-spaced borders resolve 0.0001-0.002 near zero, which is
        # where that mass sits, so this parameterisation must be paired with such a head.
        if target == "pool_regret":
            qeubo[:, n_ctx:] = torch.maximum(f0, f1) - Fs.max(dim=1, keepdim=True).values
        else:
            qeubo[:, n_ctx:] = torch.maximum(f0, f1)

    return Batch(
        x=new_X,
        y=qeubo,
        target_y=qeubo,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPqEUBOPoolPriorConfig(PriorConfig):
    lengthscale: float | tuple[float, ...] = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6

    # New finite-pool parameter
    pool_size: int = 100

    # Draw fresh observation noise for every comparison (standard preferential likelihood)
    # rather than freezing one noisy value per pool point. See `get_batch` for details.
    # Set to False only to reproduce the training distribution of checkpoints produced before
    # 2026-08-06, i.e. all existing `pfn_pref_gp_*d_qeubo_*_pool_*.pt`.
    noise_per_comparison: bool = True

    # How context comparisons are chosen. "uniform" draws independent pairs, as every existing
    # checkpoint was trained; "incumbent" runs a king-of-the-hill duel, which reshapes the
    # comparison *graph* rather than its size. See `_incumbent_duel_context` and
    # `research-backlog.md` item 7. The default preserves the existing distribution exactly.
    context_policy: str = "uniform"
    incumbent_prob: float = 1.0

    # Elo-softmax sampling. `context_policy="elo_softmax"` and `query_policy="elo_softmax"` are
    # INDEPENDENT switches so the 2x2 ablation (context x query) is a config change, not a code
    # change. Elo ratings are always accumulated over whatever context was generated, so
    # `query_policy="elo_softmax"` is well defined even with a uniform context.
    # See `.claude/design-decisions.md` and `.claude/proposal-first-prior-training.md`.
    query_policy: str = "uniform"
    elo_k: float = 1.0
    n_random_diag_queries: int = 0

    # Ban a repeated PAIR in the context. Default False, which is the agreed spec: queries reject
    # literal duplicates, the context rejects only self-pairs. True reproduces the training
    # distribution of every elo checkpoint trained before 2026-08-16, where this was hardcoded on;
    # those configs pin it explicitly. See `.claude/design-decisions.md`.
    exclude_seen_pairs: bool = False

    # "value" = max(f0,f1), what every checkpoint before 2026-08-16 was trained on.
    # "pool_regret" = max(f0,f1) - max_pool f. MUST be paired with a bar head whose borders are
    # dense near zero (see the 1-D regret configs' log-spaced borders); on a uniform head the
    # near-optimal region collapses into one bin.
    target: str = "value"

    def create_get_batch_method(self):
        return partial(
            get_batch,
            pool_size=self.pool_size,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            noise_per_comparison=self.noise_per_comparison,
            context_policy=self.context_policy,
            incumbent_prob=self.incumbent_prob,
            query_policy=self.query_policy,
            exclude_seen_pairs=self.exclude_seen_pairs,
            target=self.target,
            elo_k=self.elo_k,
            n_random_diag_queries=self.n_random_diag_queries,
            jitter=self.jitter,
        )
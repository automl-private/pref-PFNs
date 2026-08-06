"""Regression tests for the comparison-noise model of the finite-pool preferential prior.

Background. `pref_gp_qeubo_pool.get_batch` originally drew observation noise **once per pool
point** and then read every comparison off those frozen values. Because the pool reuses points,
that makes the whole context a sub-order of a single total order on `y`: the comparison graph is
necessarily acyclic and no two comparisons can ever contradict. The standard preferential
likelihood -- used by BoTorch's `PairwiseGP`, by `pref_gp_1d_qeubo_regret_v3.py`, and by
`scripts/pbo_ground_truth.py` -- instead draws fresh noise for every comparison, so cycles are
possible. `noise_per_comparison=True` is now the default; `False` retains the old behaviour so
the training distribution of pre-2026-08-06 pool checkpoints stays reproducible.

The observable difference is exactly the presence of directed cycles, and it scales with the
average degree `2 * n_ctx / pool_size` of the comparison graph. These tests pin both the
qualitative difference and its degree-dependence.
"""

import pytest
import torch

from pfns.priors.pref.pref_gp_qeubo_pool import get_batch

GP_DIM = 2
BATCH_SIZE = 8


def _make_batch(pool_size, n_ctx, noise_per_comparison, seed, noise_std=0.05):
    """Return (batch, pool_X). The pool is replayed from the same seed so context tokens can
    be mapped back to pool indices."""
    torch.manual_seed(seed)
    pool_X = torch.rand(BATCH_SIZE, pool_size, GP_DIM)
    torch.manual_seed(seed)
    batch = get_batch(
        batch_size=BATCH_SIZE,
        seq_len=n_ctx + 10,
        num_features=2 * GP_DIM,
        single_eval_pos=n_ctx,
        pool_size=pool_size,
        lengthscale=0.2,
        noise_std=noise_std,
        noise_per_comparison=noise_per_comparison,
    )
    return batch, pool_X


def _context_edges(batch, pool_X, batch_index, n_ctx):
    """Map context tokens back to (winner_idx, loser_idx) pool indices."""
    edges = set()
    for t in range(n_ctx):
        winner = batch.x[batch_index, t, :GP_DIM]
        loser = batch.x[batch_index, t, GP_DIM:]
        points = pool_X[batch_index]
        edges.add(
            (
                int((points - winner).abs().sum(-1).argmin()),
                int((points - loser).abs().sum(-1).argmin()),
            )
        )
    return edges


def _has_cycle(edges, pool_size):
    """Iterative DFS three-colouring; True iff the directed graph contains any cycle."""
    adj = {v: [] for v in range(pool_size)}
    for winner, loser in edges:
        adj[winner].append(loser)

    UNVISITED, ON_STACK, DONE = 0, 1, 2
    state = [UNVISITED] * pool_size
    for start in range(pool_size):
        if state[start] != UNVISITED:
            continue
        state[start] = ON_STACK
        stack = [(start, iter(adj[start]))]
        while stack:
            node, neighbours = stack[-1]
            for nxt in neighbours:
                if state[nxt] == ON_STACK:
                    return True
                if state[nxt] == UNVISITED:
                    state[nxt] = ON_STACK
                    stack.append((nxt, iter(adj[nxt])))
                    break
            else:
                state[node] = DONE
                stack.pop()
    return False


def _cycle_rate(pool_size, n_ctx, noise_per_comparison, n_seeds=25, noise_std=0.05):
    cyclic = total = 0
    for seed in range(n_seeds):
        batch, pool_X = _make_batch(
            pool_size, n_ctx, noise_per_comparison, seed, noise_std=noise_std
        )
        for b in range(BATCH_SIZE):
            edges = _context_edges(batch, pool_X, b, n_ctx)
            cyclic += _has_cycle(edges, pool_size)
            total += 1
    return cyclic / total


# Degrees span the historical training regime (<= 1.98) and the higher-degree regime that
# deliberate pool-size reduction targets.
DEGREE_CASES = [(100, 20), (100, 99), (20, 40), (8, 30)]


@pytest.mark.parametrize("pool_size,n_ctx", DEGREE_CASES)
def test_legacy_noise_never_produces_cycles(pool_size, n_ctx):
    """Frozen per-point noise makes the context a sub-order of one total order, so the
    comparison graph is acyclic by construction at any degree."""
    assert _cycle_rate(pool_size, n_ctx, noise_per_comparison=False) == 0.0


def test_per_comparison_noise_produces_cycles_at_high_degree():
    """Fresh noise per comparison must allow contradictions once points are actually reused."""
    assert _cycle_rate(8, 30, noise_per_comparison=True) > 0.05


def test_cycle_rate_is_negligible_at_historical_degree():
    """At the settings the existing pool checkpoints were trained under (pool_size=100), the
    two noise models are nearly indistinguishable. This is why the fix does not invalidate
    those checkpoints -- see .claude/research-backlog.md item 10."""
    assert _cycle_rate(100, 20, noise_per_comparison=True) == 0.0
    assert _cycle_rate(100, 99, noise_per_comparison=True) < 0.05


def test_cycle_rate_increases_with_degree():
    """The divergence between the two models is monotone in comparison-graph degree."""
    rates = [_cycle_rate(p, n, noise_per_comparison=True) for p, n in DEGREE_CASES]
    assert rates == sorted(rates), rates
    assert rates[0] == 0.0 and rates[-1] > rates[0]


@pytest.mark.parametrize("noise_per_comparison", [True, False])
def test_noiseless_limit_is_acyclic(noise_per_comparison):
    """With noise_std=0 the comparison is exactly f_i > f_j, a strict total order, so neither
    mode can produce a cycle."""
    assert _cycle_rate(8, 30, noise_per_comparison, noise_std=0.0) == 0.0


@pytest.mark.parametrize("noise_per_comparison", [True, False])
def test_batch_invariants_unchanged(noise_per_comparison):
    """The fix must not disturb shapes, dtypes, or the context/query target split."""
    n_ctx, seq_len = 20, 30
    torch.manual_seed(0)
    batch = get_batch(
        batch_size=BATCH_SIZE,
        seq_len=seq_len,
        num_features=2 * GP_DIM,
        single_eval_pos=n_ctx,
        pool_size=100,
        lengthscale=0.2,
        noise_std=0.05,
        noise_per_comparison=noise_per_comparison,
    )
    assert batch.x.shape == (BATCH_SIZE, seq_len, 2 * GP_DIM)
    assert batch.target_y.shape == (BATCH_SIZE, seq_len)
    assert batch.single_eval_pos == n_ctx
    assert torch.equal(batch.y, batch.target_y)
    # The qEUBO target is written only at query positions.
    assert torch.all(batch.target_y[:, :n_ctx] == 0)
    assert torch.all(batch.target_y[:, n_ctx:] != 0)

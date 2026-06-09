#!/usr/bin/env python3
"""
Run paper-style log10(simple regret) evaluation on 1D GP preference tasks.

For every unique GP hyperparameter set discovered in the non-vanilla PFN
configs, this script generates GP benchmark functions with those hyperparameters
and evaluates every baseline plus every non-vanilla PFN checkpoint.

The plotted quantity is:

    log10(max_x f(x) - f(x_hat_t))

where x_hat_t is the method recommendation after t pairwise comparisons.
Regret is computed on the fixed candidate grid using the noiseless latent GP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gpytorch
import torch

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="fontconfig-")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agents import (
    FixedHyperparamQEUBOAgent,
    GPPBOAgent,
    QEUBOAgent,
    RandomAgent,
)
from evaluation.agents.base import Comparison, PBOAgent
from evaluation.benchmarks_1d import BENCHMARKS_1D, make_benchmark_1d
from evaluation.grid_designs import make_1d_grid
from evaluation.loop import run_bo_loop, _candidate_value
from pfns.run_training_cli import load_config_from_python


DEFAULT_SKIP_CHECKPOINTS = ("pfn_vanilla_gp_1d_10M.pt",)
MULTIDIM_CHECKPOINT_RE = re.compile(r"^pref_gp_(\d+)d_(.+)$")
BASELINE_METHODS = ("random", "gp_pbo", "qeubo", "fixed_qeubo")


@dataclass(frozen=True)
class GPHyperparameters:
    lengthscale: float
    outputscale: float
    noise_std: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "lengthscale": float(self.lengthscale),
            "outputscale": float(self.outputscale),
            "noise_std": float(self.noise_std),
        }

    @property
    def signature(self) -> str:
        return (
            f"lengthscale={self.lengthscale:g}_"
            f"outputscale={self.outputscale:g}_"
            f"noise_std={self.noise_std:g}"
        )


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_path: Path
    config_path: Path
    checkpoint_name: str
    method_name: str
    kind: str
    train_hparams: GPHyperparameters
    prior_class: str
    input_dim: int = 1
    config_template_name: Optional[str] = None

    @property
    def is_ranking_baseline(self) -> bool:
        return self.kind == "compare"


@dataclass(frozen=True)
class EvalSuiteSpec:
    name: str
    functions: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    oracle_noise_std: float
    baseline_hparams: GPHyperparameters
    eval_hparams: Optional[GPHyperparameters]
    benchmark: Optional[Dict]

class GaussianPreferenceOracle:
    """Fixed-grid oracle with Gaussian comparison noise."""

    def __init__(
        self,
        x_grid: torch.Tensor,
        f_grid: torch.Tensor,
        noise_std: float,
        seed: int,
    ) -> None:
        self.x_grid = x_grid.detach().cpu().float()
        self.f_grid = f_grid.detach().cpu().float()
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

        if self.x_grid.ndim == 1:
            assert self.x_grid.shape[0] == self.f_grid.shape[0]
        else:
            assert self.x_grid.shape[0] == self.f_grid.shape[0]

    @property
    def f_opt(self) -> float:
        return self.f_grid.max().item()

    @property
    def x_opt(self):
        # returns tuple of coordinates for the optimal point (x1, x2, ..., xd)
        return self._format_point(self.x_grid[self.f_grid.argmax()])

    def _point_tensor(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.x_grid.dtype)
        # Если oracle 1D, то x_grid.shape == (M,), и точка должна быть scalar tensor
        if self.x_grid.ndim == 1:
            return x.reshape(())
        # А если multidim, то точка приводится к вектору
        return x.reshape(-1)

    def _nearest_idx(self, x) -> int:
        x = self._point_tensor(x)
        if self.x_grid.ndim == 1:
            return int((self.x_grid - float(x)).abs().argmin().item())

        distances = torch.linalg.norm(self.x_grid - x, dim=-1)
        return int(distances.argmin().item())

    def _format_point(self, x: torch.Tensor):
        if self.x_grid.ndim == 1:
            return float(x.item())
        return tuple(x.detach().cpu().tolist())

    def f_at(self, x) -> float:
        idx = self._nearest_idx(x)
        return self.f_grid[idx].item()

    def compare(self, x1, x2):
        f1 = self.f_at(x1)
        f2 = self.f_at(x2)

        if self.noise_std > 0:
            f1 = f1 + self.noise_std * torch.randn((), generator=self._rng).item()
            f2 = f2 + self.noise_std * torch.randn((), generator=self._rng).item()

        if f1 >= f2:
            return self._format_point(self._point_tensor(x1)), self._format_point(self._point_tensor(x2))
        return self._format_point(self._point_tensor(x2)), self._format_point(self._point_tensor(x1))

    def simple_regret(self, x_recommended) -> float:
        return max(self.f_opt - self.f_at(x_recommended), 0.0)


def sample_gp_function(
    *,
    n_grid: int,
    input_dim: int = 1,
    hparams: GPHyperparameters,
    seed: int,
    grid_design: str = "uniform",
    grid_seed: int = 0,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a noiseless latent GP path on a fixed candidate set."""
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if input_dim == 1:
        x_eval = make_1d_grid(n_grid, design=grid_design, seed=grid_seed, dtype=torch.double)
        x_kernel = x_eval.unsqueeze(-1)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(grid_seed))
        if grid_design == "lhs":
            base = torch.arange(n_grid, dtype=torch.double).unsqueeze(-1)
            offsets = torch.rand(n_grid, input_dim, generator=generator, dtype=torch.double)
            x_kernel = (base + offsets) / float(n_grid)
            for dim in range(input_dim):
                perm = torch.randperm(n_grid, generator=generator)
                x_kernel[:, dim] = x_kernel[perm, dim]
        else:
            x_kernel = torch.rand(n_grid, input_dim, generator=generator, dtype=torch.double)
        x_eval = x_kernel

    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    base_kernel.lengthscale = hparams.lengthscale
    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    covar_module.outputscale = hparams.outputscale

    mean = torch.zeros(n_grid, dtype=torch.double)
    covar = covar_module(x_kernel).add_jitter(jitter)
    cov = covar.to_dense() if hasattr(covar, "to_dense") else covar.evaluate()
    dist = torch.distributions.MultivariateNormal(mean, cov)

    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            f = dist.sample()
    finally:
        torch.random.set_rng_state(old_state)

    return x_eval.float(), f.float()


def _build_pref_context(
    comparisons: list[Comparison],
    *,
    dtype: torch.dtype,
    device: torch.device,
    input_dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not comparisons:
        x_ctx = torch.zeros(1, 0, input_dims, dtype=dtype, device=device)
        y_ctx = torch.zeros(1, 0, dtype=dtype, device=device)
        return x_ctx, y_ctx

    pairs = torch.as_tensor(comparisons, dtype=dtype, device=device)
    if pairs.ndim == 3:
        pairs = pairs.reshape(pairs.shape[0], -1)
    x_ctx = pairs.unsqueeze(0)
    y_ctx = torch.zeros(1, len(comparisons), dtype=dtype, device=device)
    return x_ctx, y_ctx


def _safe_random_challenger(
    candidate_pool: torch.Tensor,
    chosen: float,
) -> float:
    pool = candidate_pool.tolist()
    if len(pool) <= 1:
        return chosen
    for _ in range(20):
        challenger = pool[torch.randint(len(pool), (1,)).item()]
        if challenger != chosen:
            return challenger
    return pool[0] if pool[0] != chosen else pool[-1]

# TODO: make fixes for multidim
class UtilityPFNAgent(PBOAgent):
    """PFN agent for utility or negative-regret scalar predictions."""

    def __init__(
        self,
        model,
        *,
        device: str,
        n_ts_samples: int,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.criterion = model.criterion
        self.n_ts_samples = int(n_ts_samples)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _logits(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        x_ctx, y_ctx = _build_pref_context(
            comparisons,
            dtype=self.dtype,
            device=self.device,
        )
        x_query = candidate_pool.to(dtype=self.dtype, device=self.device)
        test_x = torch.zeros(1, x_query.numel(), 2, dtype=self.dtype, device=self.device)
        test_x[0, :, 0] = x_query
        with torch.no_grad():
            logits = self.model(x_ctx, y_ctx, test_x=test_x)
        return logits[0] # (M, num_bins)

    def _score(self, comparisons: list[Comparison], candidate_pool: torch.Tensor) -> torch.Tensor:
        logits = self._logits(comparisons, candidate_pool)
        return self.criterion.mean(logits).detach().cpu()

    def _sample_score(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._logits(comparisons, candidate_pool) # (M, B)
        return self.criterion.sample(logits).detach().cpu() # (M,) один sampled scalar score на каждую candidate point

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[float, float]:
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_pool[idx[0]].item(), candidate_pool[idx[1]].item()

        argmaxes: List[float] = []
        for _ in range(max(1, self.n_ts_samples)): # n_ts_samples = 2
            # 1. sample one possible score function over grid
            # 2. take its argmax
            # 3. sample another possible score function over grid
            # 4. take its argmax
            # 5. use these two argmaxes as comparison pair
            scores = self._sample_score(comparisons, candidate_pool)
            argmaxes.append(candidate_pool[scores.argmax()].item()) # [a, b]

        if len(set(argmaxes)) == 1:
            return argmaxes[0], _safe_random_challenger(candidate_pool, argmaxes[0])
        return argmaxes[0], argmaxes[1]

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        if not comparisons:
            return candidate_pool[candidate_pool.shape[0] // 2].item()
        scores = self._score(comparisons, candidate_pool)
        return candidate_pool[scores.argmax()].item()


class PairScorePFNAgent(PBOAgent):
    """PFN agent for pair-valued qEUBO or qEUBO-negative-regret predictions."""

    def __init__(
        self,
        model,
        *,
        device: str,
        pair_batch_size: int,
        input_dim: int,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.criterion = model.criterion
        self.pair_batch_size = int(pair_batch_size)
        self.input_dim = int(input_dim)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _pair_scores(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        x = candidate_pool.to(dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            M = x.numel()
            x1, x2 = torch.meshgrid(x, x, indexing="ij")
            pairs = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
        else:
            M = x.shape[0]
            x1 = x.repeat_interleave(M, dim=0)
            x2 = x.repeat(M, 1)
            pairs = torch.cat([x1, x2], dim=-1)

        x_ctx, y_ctx = _build_pref_context(
            comparisons,
            dtype=self.dtype,
            device=self.device,
            input_dims=self.input_dim*2
        )
        # 1D x (4,)      x1 (4, 4)   x2 (4, 4)   pairs (16, 2)
        # 2D x (4, 2)   x1 (16, 2)  x2 (16, 2)  pairs (16, 4)

        # Определяет, сколько пар отправлять в PFN за один forward pass.
        batch_size = pairs.shape[0] if self.pair_batch_size <= 0 else self.pair_batch_size

        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, pairs.shape[0], batch_size):
                test_x = pairs[start : start + batch_size].unsqueeze(0)
                logits = self.model(x_ctx, y_ctx, test_x=test_x)
                chunks.append(self.criterion.mean(logits)[0].detach().cpu())

        return torch.cat(chunks).reshape(M, M)

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[float, float]:
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return _candidate_value(candidate_pool[idx[0]]), _candidate_value(candidate_pool[idx[1]])

        scores = self._pair_scores(comparisons, candidate_pool)
        # scores[i, j] ≈ E[max(f(x_i), f(x_j)) | comparisons]
        # scores[i, j] ≈ E[max(f(x_i), f(x_j)) - f* | comparisons]
        idx = torch.arange(scores.shape[0])
        scores[idx, idx] = -torch.inf
        # Диагональ зануляется 
        flat_idx = torch.argmax(scores)
        # Потом flat index переводится в пару индексов
        i = int(flat_idx // scores.shape[1])
        j = int(flat_idx % scores.shape[1])
        # argmax_{i != j} scores[i, j]
        return _candidate_value(candidate_pool[i]), _candidate_value(candidate_pool[j])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        if not comparisons:
            return _candidate_value(candidate_pool[candidate_pool.shape[0] // 2])
        scores = self._pair_scores(comparisons, candidate_pool)
        diag = torch.diagonal(scores)
        return _candidate_value(candidate_pool[diag.argmax()])


# TODO: make fixes for multidim
class CompareCopelandPFNAgent(PairScorePFNAgent):
    """Ranking-only PFN baseline for compare checkpoints."""

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[float, float]:
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_pool[idx[0]].item(), candidate_pool[idx[1]].item()
        scores = self._copeland_scores(comparisons, candidate_pool)
        top2 = torch.topk(scores, k=2).indices
        return candidate_pool[top2[0]].item(), candidate_pool[top2[1]].item()

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> float:
        if not comparisons:
            return candidate_pool[candidate_pool.shape[0] // 2].item()
        scores = self._copeland_scores(comparisons, candidate_pool)
        return candidate_pool[scores.argmax()].item()

    def _copeland_scores(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        # pair_scores[i, j] is model E[target] for pair [x_i, x_j].
        # In compare training target=1 iff the second candidate is better.
        pair_scores = self._pair_scores(comparisons, candidate_pool).clamp(0.0, 1.0)
        M = pair_scores.shape[0]
        mask = ~torch.eye(M, dtype=torch.bool)
        column_scores = pair_scores.masked_fill(~mask, float("nan")).nanmean(dim=0)
        return column_scores.nan_to_num(0.0)


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PFN checkpoints and baselines on GP log-regret suites."
    )
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        default=["checkpoints", "checkpoints2", "checkpoints3", "checkpoints4"],
    )
    parser.add_argument("--config-dirs", nargs="+", default=["my_configs", "my_configs2", "my_configs3"])
    parser.add_argument("--out", type=Path, default=Path("results/gp_log_regret/results.pt"))
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-gp-functions", type=int, default=5)
    parser.add_argument("--n-bo-seeds", type=int, default=10)
    parser.add_argument("--n-grid", type=int, default=500)
    parser.add_argument("--grid-design", choices=("uniform", "lhs"), default="uniform")
    parser.add_argument("--grid-seed-offset", type=int, default=20_000)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pfn-ts-samples", type=int, default=2)
    parser.add_argument("--pfn-pair-batch-size", type=int, default=4096)
    parser.add_argument("--qeubo-num-acqf-samples", type=int, default=512)
    parser.add_argument("--qeubo-max-fit-iter", type=int, default=100)
    parser.add_argument("--qeubo-fit-hyperparams", action="store_true")
    parser.add_argument("--fixed-qeubo-xtol", type=float, default=1e-6)
    parser.add_argument("--fixed-qeubo-maxfev", type=int, default=100)
    parser.add_argument("--fixed-qeubo-mc-samples", type=int, default=512)
    parser.add_argument("--fixed-qeubo-batch-eval-size", type=int, default=2048)
    parser.add_argument("--fixed-qeubo-jitter", type=float, default=1e-6)
    parser.add_argument("--fixed-qeubo-mean-constant", type=float, default=0.0)
    parser.add_argument("--gp-seed-offset", type=int, default=0)
    parser.add_argument("--oracle-seed-offset", type=int, default=10_000)
    parser.add_argument(
        "--benchmark-mode",
        choices=("gp_only", "deterministic_only", "all"),
        default="gp_only",
        help="Which benchmark suites to run. Default preserves the original GP-prior behavior.",
    )
    parser.add_argument(
        "--deterministic-benchmarks",
        nargs="+",
        default=list(BENCHMARKS_1D.keys()),
        help="Deterministic 1D benchmark names to run when benchmark mode includes them.",
    )
    parser.add_argument(
        "--deterministic-normalizations",
        nargs="+",
        choices=("raw", "std1"),
        default=["raw", "std1"],
        help="Utility scaling modes for deterministic benchmark suites.",
    )
    parser.add_argument(
        "--deterministic-noise-std",
        type=float,
        default=0.05,
        help="Gaussian comparison noise used for deterministic benchmark suites.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Method names to run, or 'all'. Baselines: random gp_pbo qeubo fixed_qeubo.",
    )
    parser.add_argument("--exclude-methods", nargs="*", default=[])
    parser.add_argument("--only-checkpoints", nargs="*", default=[])
    parser.add_argument("--skip-checkpoints", nargs="*", default=list(DEFAULT_SKIP_CHECKPOINTS))
    parser.add_argument("--save-every-method", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def find_config_paths(config_dirs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for config_dir in config_dirs:
        root = Path(config_dir)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("train_*.py")):
            name = path.stem.removeprefix("train_")
            out[name] = path
    return out


def infer_checkpoint_input_dim(name: str) -> int:
    match = MULTIDIM_CHECKPOINT_RE.match(name)
    if match is None:
        return 1
    return int(match.group(1))


def resolve_config_path_for_checkpoint(
    name: str,
    config_paths: Mapping[str, Path],
) -> tuple[Optional[Path], Optional[str]]:
    if name in config_paths:
        return config_paths[name], None

    match = MULTIDIM_CHECKPOINT_RE.match(name)
    if match is None:
        return None, None

    dim = int(match.group(1))
    suffix = match.group(2)
    if dim == 1:
        return None, None

    template_name = f"pref_gp_1d_{suffix}"
    if template_name in config_paths:
        return config_paths[template_name], template_name
    return None, template_name


def classify_checkpoint(name: str) -> str:
    if "vanilla" in name:
        return "vanilla"
    if "compare" in name:
        return "compare"
    if "qeubo_regret" in name:
        return "qeubo_regret"
    if "qeubo" in name:
        return "qeubo"
    if "regret" in name:
        return "regret"
    return "utility"


def load_hparams_from_config(config_path: Path) -> tuple[GPHyperparameters, str]:
    config = load_config_from_python(str(config_path), 0)
    prior = config.priors[0]
    hparams = GPHyperparameters(
        lengthscale=float(getattr(prior, "lengthscale")),
        outputscale=float(getattr(prior, "outputscale")),
        noise_std=float(getattr(prior, "noise_std")),
    )
    return hparams, type(prior).__name__


def discover_checkpoints(args: argparse.Namespace) -> tuple[List[CheckpointSpec], Dict[str, str]]:
    config_paths = find_config_paths(args.config_dirs)
    only = set(args.only_checkpoints)
    skip = set(args.skip_checkpoints)
    skipped: Dict[str, str] = {}
    specs: List[CheckpointSpec] = []

    for checkpoint_dir in args.checkpoint_dirs:
        root = Path(checkpoint_dir)
        if not root.is_dir():
            continue
        for checkpoint_path in sorted(root.glob("pfn_*.pt")):
            checkpoint_name = checkpoint_path.name
            name = checkpoint_path.stem.removeprefix("pfn_")
            if only and checkpoint_name not in only and name not in only:
                continue
            if checkpoint_name in skip or name in skip:
                skipped[checkpoint_name] = "listed in skip checkpoints"
                continue
            kind = classify_checkpoint(name)
            if kind == "vanilla":
                skipped[checkpoint_name] = "vanilla GP checkpoint requires direct utility observations"
                continue
            config_path, config_template_name = resolve_config_path_for_checkpoint(name, config_paths)
            if config_path is None:
                if config_template_name is None:
                    skipped[checkpoint_name] = f"missing config train_{name}.py"
                else:
                    skipped[checkpoint_name] = (
                        f"missing config train_{name}.py and template "
                        f"train_{config_template_name}.py"
                    )
                continue

            hparams, prior_class = load_hparams_from_config(config_path)
            method_name = f"{name}_copeland" if kind == "compare" else name
            specs.append(
                CheckpointSpec(
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    checkpoint_name=checkpoint_name,
                    method_name=method_name,
                    kind=kind,
                    train_hparams=hparams,
                    prior_class=prior_class,
                    input_dim=infer_checkpoint_input_dim(name),
                    config_template_name=config_template_name,
                )
            )

    return specs, skipped


def _replace_config_obj(obj, **updates):
    updates = {key: value for key, value in updates.items() if hasattr(obj, key)}
    if not updates:
        return obj
    if is_dataclass(obj):
        return replace(obj, **updates)
    for key, value in updates.items():
        setattr(obj, key, value)
    return obj


def apply_checkpoint_shape_overrides(config, spec: CheckpointSpec):
    if spec.input_dim <= 1:
        return config
    num_features = 2 * spec.input_dim
    model = _replace_config_obj(config.model, features_per_group=num_features)
    batch_shape_sampler = _replace_config_obj(
        config.batch_shape_sampler,
        min_num_features=num_features,
        max_num_features=num_features,
    )
    if is_dataclass(config):
        return replace(config, model=model, batch_shape_sampler=batch_shape_sampler)
    config.model = model
    config.batch_shape_sampler = batch_shape_sampler
    return config


def load_pfn_model(spec: CheckpointSpec, device: str):
    config = load_config_from_python(str(spec.config_path), 0)
    config = apply_checkpoint_shape_overrides(config, spec)
    model = config.model.create_model().to(device)
    model.eval()
    state = torch.load(spec.checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model


def requested_methods(args: argparse.Namespace, checkpoint_specs: Sequence[CheckpointSpec]) -> List[str]:
    pfn_names = [spec.method_name for spec in checkpoint_specs]
    all_names = list(BASELINE_METHODS) + pfn_names
    if args.methods == ["all"]:
        selected = list(BASELINE_METHODS) + [
            spec.method_name for spec in checkpoint_specs if spec.input_dim == 1
        ]
    else:
        requested = set(args.methods)
        selected = [name for name in all_names if name in requested]
    excluded = set(args.exclude_methods)
    return [name for name in selected if name not in excluded]


def make_pfn_agent_factory(
    spec: CheckpointSpec,
    model,
    args: argparse.Namespace,
) -> Callable[[], PBOAgent]:
    if spec.kind in {"qeubo", "qeubo_regret"}:
        return lambda: PairScorePFNAgent(
            model,
            device=args.device,
            pair_batch_size=args.pfn_pair_batch_size,
            input_dim=spec.input_dim,
        )
    if spec.kind == "compare":
        return lambda: CompareCopelandPFNAgent(
            model,
            device=args.device,
            pair_batch_size=args.pfn_pair_batch_size,
            input_dim=spec.input_dim,
        )
    return lambda: UtilityPFNAgent(
        model,
        device=args.device,
        n_ts_samples=args.pfn_ts_samples,
    )


def build_eval_suite_specs(
    args: argparse.Namespace,
    eval_hparams: Sequence[GPHyperparameters],
) -> List[EvalSuiteSpec]:
    suites: List[EvalSuiteSpec] = []

    input_dims = set()
    for value in list(getattr(args, "methods", [])) + list(getattr(args, "only_checkpoints", [])):
        name = Path(str(value)).stem.removeprefix("pfn_")
        match = MULTIDIM_CHECKPOINT_RE.match(name)
        if match is not None:
            input_dims.add(int(match.group(1)))
    if not input_dims:
        input_dims.add(1)

    if args.benchmark_mode in {"gp_only", "all"}:
        for hparams in eval_hparams:
            for input_dim in sorted(input_dims):
                gp_functions = tuple(
                    sample_gp_function(
                        n_grid=args.n_grid,
                        input_dim=input_dim,
                        hparams=hparams,
                        seed=args.gp_seed_offset + gp_idx,
                        grid_design=args.grid_design,
                        grid_seed=args.grid_seed_offset + gp_idx,
                    )
                    for gp_idx in range(args.n_gp_functions)
                )
                suite_name = hparams.signature if input_dim == 1 else f"{hparams.signature}_d{input_dim}"
                benchmark = None if input_dim == 1 else {
                    "kind": "gp",
                    "input_dim": int(input_dim),
                    "grid_design": args.grid_design,
                    "grid_seed_offset": int(args.grid_seed_offset),
                }
                suites.append(
                    EvalSuiteSpec(
                        name=suite_name,
                        functions=gp_functions,
                        oracle_noise_std=hparams.noise_std,
                        baseline_hparams=hparams,
                        eval_hparams=hparams,
                        benchmark=benchmark,
                    )
                )

    if args.benchmark_mode in {"deterministic_only", "all"}:
        reference_hparams = eval_hparams[0]
        baseline_hparams = GPHyperparameters(
            lengthscale=reference_hparams.lengthscale,
            outputscale=reference_hparams.outputscale,
            noise_std=args.deterministic_noise_std,
        )
        for benchmark_name in args.deterministic_benchmarks:
            for normalization in args.deterministic_normalizations:
                x_grid, f_grid = make_benchmark_1d(
                    benchmark_name,
                    n_grid=args.n_grid,
                    normalization=normalization,
                    device="cpu",
                    grid_design=args.grid_design,
                    grid_seed=args.grid_seed_offset,
                )
                suite_name = f"{benchmark_name}_{normalization}"
                suites.append(
                    EvalSuiteSpec(
                        name=suite_name,
                        functions=((x_grid, f_grid),),
                        oracle_noise_std=args.deterministic_noise_std,
                        baseline_hparams=baseline_hparams,
                        eval_hparams=None,
                        benchmark={
                            "kind": "deterministic",
                            "name": benchmark_name,
                            "normalization": normalization,
                            "noise_std": float(args.deterministic_noise_std),
                            "reference_hparams": reference_hparams.as_dict(),
                            "grid_design": args.grid_design,
                            "grid_seed": int(args.grid_seed_offset),
                        },
                    )
                )

    return suites


def make_agent_for_method(
    method_name: str,
    *,
    hparams: GPHyperparameters,
    args: argparse.Namespace,
    pfn_factories: Mapping[str, Callable[[], PBOAgent]],
    bo_seed: int,
) -> PBOAgent:
    if method_name == "random":
        return RandomAgent(seed=bo_seed)
    if method_name == "gp_pbo":
        return GPPBOAgent(
            lengthscale=hparams.lengthscale,
            outputscale=hparams.outputscale,
        )
    if method_name == "qeubo":
        return QEUBOAgent(
            fit_hyperparams=args.qeubo_fit_hyperparams,
            max_fit_iter=args.qeubo_max_fit_iter,
            num_acqf_samples=args.qeubo_num_acqf_samples,
        )
    if method_name == "fixed_qeubo":
        return FixedHyperparamQEUBOAgent(
            lengthscale=hparams.lengthscale,
            outputscale=hparams.outputscale,
            noise_std=hparams.noise_std,
            mean_constant=args.fixed_qeubo_mean_constant,
            jitter=args.fixed_qeubo_jitter,
            xtol=args.fixed_qeubo_xtol,
            maxfev=args.fixed_qeubo_maxfev,
            num_acqf_samples=args.fixed_qeubo_mc_samples,
            batch_eval_size=args.fixed_qeubo_batch_eval_size,
        )
    return pfn_factories[method_name]()


def hparams_equal(a: GPHyperparameters, b: GPHyperparameters) -> bool:
    return (
        math.isclose(a.lengthscale, b.lengthscale)
        and math.isclose(a.outputscale, b.outputscale)
        and math.isclose(a.noise_std, b.noise_std)
    )


def method_metadata(
    method_name: str,
    spec_by_method: Mapping[str, CheckpointSpec],
    eval_hparams: Optional[GPHyperparameters],
    benchmark: Optional[Mapping],
) -> Dict:
    eval_hparams_dict = eval_hparams.as_dict() if eval_hparams is not None else None
    if method_name in BASELINE_METHODS:
        return {
            "method_name": method_name,
            "kind": "baseline",
            "checkpoint": None,
            "config": None,
            "train_hparams": None,
            "eval_hparams": eval_hparams_dict,
            "benchmark": dict(benchmark) if benchmark is not None else None,
            "is_in_domain": None,
            "is_ranking_baseline": False,
        }

    spec = spec_by_method[method_name]
    is_in_domain = False if eval_hparams is None else hparams_equal(spec.train_hparams, eval_hparams)
    return {
        "method_name": method_name,
        "kind": spec.kind,
        "checkpoint": str(spec.checkpoint_path),
        "config": str(spec.config_path),
        "config_template_name": spec.config_template_name,
        "prior_class": spec.prior_class,
        "input_dim": spec.input_dim,
        "train_hparams": spec.train_hparams.as_dict(),
        "eval_hparams": eval_hparams_dict,
        "benchmark": dict(benchmark) if benchmark is not None else None,
        "is_in_domain": is_in_domain,
        "is_ranking_baseline": spec.is_ranking_baseline,
    }


def main() -> None:
    args = parse_args()
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    checkpoint_specs, skipped = discover_checkpoints(args)
    spec_by_method = {spec.method_name: spec for spec in checkpoint_specs}
    selected_methods = requested_methods(args, checkpoint_specs)
    print(f"selected_methods: {selected_methods}")
    if not selected_methods:
        raise RuntimeError("No methods selected for evaluation.")

    eval_hparams = sorted(
        {spec.train_hparams for spec in checkpoint_specs},
        key=lambda h: (h.lengthscale, h.outputscale, h.noise_std),
    )
    if not eval_hparams:
        raise RuntimeError("No non-skipped PFN checkpoint configs found.")

    suite_specs = build_eval_suite_specs(args, eval_hparams)
    if not suite_specs:
        raise RuntimeError("No benchmark suites selected for evaluation.")

    print("[discovery] selected methods:", ", ".join(selected_methods))
    print("[discovery] eval hparams:", ", ".join(h.signature for h in eval_hparams))
    print("[discovery] benchmark suites:", ", ".join(suite.name for suite in suite_specs))
    if skipped:
        print("[discovery] skipped:", json.dumps(skipped, indent=2))

    pfn_models = {}
    for spec in checkpoint_specs:
        if spec.method_name in selected_methods:
            print(f"[load] {spec.method_name} <- {spec.checkpoint_path}")
            pfn_models[spec.method_name] = load_pfn_model(spec, args.device)

    pfn_factories = {
        spec.method_name: make_pfn_agent_factory(spec, pfn_models[spec.method_name], args)
        for spec in checkpoint_specs
        if spec.method_name in pfn_models
    }

    result = {
        "metadata": {
            "n_grid": args.n_grid,
            "budget": args.budget,
            "n_init": args.n_init,
            "n_gp_functions": args.n_gp_functions,
            "n_bo_seeds": args.n_bo_seeds,
            "grid_design": args.grid_design,
            "grid_seed_offset": args.grid_seed_offset,
            "eps": args.eps,
            "device": args.device,
            "selected_methods": selected_methods,
            "skipped_checkpoints": skipped,
            "benchmark_mode": args.benchmark_mode,
            "deterministic_benchmarks": args.deterministic_benchmarks,
            "deterministic_normalizations": args.deterministic_normalizations,
            "deterministic_noise_std": args.deterministic_noise_std,
        },
        "suites": {},
    }

    for suite_spec in suite_specs:
        suite_name = suite_spec.name
        print(f"\n=== suite: {suite_name} ===")
        suite = {"methods": {}}
        if suite_spec.eval_hparams is not None:
            suite["eval_hparams"] = suite_spec.eval_hparams.as_dict()
        if suite_spec.benchmark is not None:
            suite["benchmark"] = suite_spec.benchmark

        for method_name in selected_methods:
            print(f"[suite={suite_name}] method={method_name}")
            n_functions = len(suite_spec.functions)
            simple_regret = torch.empty(n_functions, args.n_bo_seeds, args.budget)
            utility_at_recommendation = torch.empty_like(simple_regret)

            for function_idx, (x_grid, f_grid) in enumerate(suite_spec.functions):
                for bo_seed in range(args.n_bo_seeds):
                    seed = args.oracle_seed_offset + function_idx * 100_000 + bo_seed
                    oracle = GaussianPreferenceOracle(
                        x_grid=x_grid,
                        f_grid=f_grid,
                        noise_std=suite_spec.oracle_noise_std,
                        seed=seed,
                    )

                    agent = make_agent_for_method(
                        method_name,
                        hparams=suite_spec.baseline_hparams,
                        args=args,
                        pfn_factories=pfn_factories,
                        bo_seed=bo_seed,
                    )

                    run = run_bo_loop(
                        agent=agent,
                        oracle=oracle,
                        budget=args.budget,
                        n_init=args.n_init,
                        seed=bo_seed,
                        verbose=args.verbose,
                    )
                    sr = torch.tensor(run["simple_regret"], dtype=torch.float32)
                    utility = torch.tensor(
                        [oracle.f_at(x_hat) for x_hat in run["recommendations"]],
                        dtype=torch.float32,
                    )
                    simple_regret[function_idx, bo_seed] = sr
                    utility_at_recommendation[function_idx, bo_seed] = utility

                    print(
                        f"  function={function_idx:03d} seed={bo_seed:03d} "
                        f"final_regret={sr[-1].item():.6g}"
                    )

            log10_regret = torch.log10(torch.clamp(simple_regret, min=args.eps))
            suite["methods"][method_name] = {
                "simple_regret": simple_regret,
                "log10_regret": log10_regret,
                "utility_at_recommendation": utility_at_recommendation,
                "metadata": method_metadata(
                    method_name,
                    spec_by_method,
                    suite_spec.eval_hparams,
                    suite_spec.benchmark,
                ),
            }

            if args.save_every_method:
                result["suites"][suite_name] = suite
                atomic_torch_save(result, args.out)
                print(f"[save] {args.out}")

        result["suites"][suite_name] = suite
        atomic_torch_save(result, args.out)
        print(f"[save] {args.out}")

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

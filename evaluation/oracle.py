"""
Oracle and test functions for preference-based BO evaluation.

The oracle knows the true f(x) but only reveals pairwise comparisons.
All functions operate on x in [0, 1]^d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import gpytorch
import torch

from evaluation.grid_designs import make_1d_grid


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
class EvalSuiteSpec:
    name: str
    functions: tuple[tuple[torch.Tensor, torch.Tensor] | "SampledGPFunction", ...]
    oracle_noise_std: float
    baseline_hparams: GPHyperparameters
    eval_hparams: Optional[GPHyperparameters]
    benchmark: Optional[Dict]


@dataclass(frozen=True)
class SampledGPFunction:
    support: str
    input_dim: int
    hparams: GPHyperparameters
    seed: int
    x_grid: Optional[torch.Tensor] = None
    f_grid: Optional[torch.Tensor] = None
    rff_weights: Optional[torch.Tensor] = None
    rff_phases: Optional[torch.Tensor] = None
    rff_coeffs: Optional[torch.Tensor] = None
    rff_scale: float = 1.0
    x_opt: Optional[object] = None
    f_opt: Optional[float] = None

    def evaluate(self, x, *, batch_size: int = 2048) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.input_dim)

        if self.support == "grid":
            if self.x_grid is None or self.f_grid is None:
                raise ValueError("Grid GP function requires x_grid and f_grid.")
            grid = self.x_grid.reshape(-1, self.input_dim)
            nearest = torch.cdist(x, grid).argmin(dim=1)
            return self.f_grid[nearest]

        if self.support != "continuous_rff":
            raise ValueError(f"Unknown GP function support {self.support!r}.")
        if self.rff_weights is None or self.rff_phases is None or self.rff_coeffs is None:
            raise ValueError("Continuous RFF GP function is missing RFF parameters.")

        values: List[torch.Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size]
            proj = chunk @ self.rff_weights + self.rff_phases
            values.append(self.rff_scale * (torch.cos(proj) @ self.rff_coeffs))
        return torch.cat(values)


class GaussianPreferenceOracle:
    """Oracle with Gaussian comparison noise for grid or continuous GP functions."""

    def __init__(
        self,
        x_grid: Optional[torch.Tensor] = None,
        f_grid: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        seed: int = 0,
        *,
        gp_function: Optional[SampledGPFunction] = None,
    ) -> None:
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)
        self.gp_function = gp_function

        if gp_function is not None:
            if x_grid is not None or f_grid is not None:
                raise ValueError("Pass either gp_function or x_grid/f_grid, not both.")
            if gp_function.x_opt is None or gp_function.f_opt is None:
                raise ValueError("SampledGPFunction must contain x_opt and f_opt.")

            self.support = gp_function.support
            self.input_dim = int(gp_function.input_dim)
            self.x_grid = gp_function.x_grid
            self.f_grid = gp_function.f_grid
            self._x_opt = gp_function.x_opt
            self._f_opt = float(gp_function.f_opt)
            return

        if x_grid is None or f_grid is None:
            raise ValueError("Grid oracle requires x_grid and f_grid.")

        self.support = "grid"
        self.x_grid = x_grid.detach().cpu().float()
        self.f_grid = f_grid.detach().cpu().float()
        self.input_dim = 1 if self.x_grid.ndim == 1 else int(self.x_grid.shape[1])

        if self.x_grid.shape[0] != self.f_grid.shape[0]:
            raise ValueError("x_grid and f_grid must have the same first dimension.")

        best_idx = int(self.f_grid.argmax().item())
        self._x_opt = self._format_point(self.x_grid[best_idx])
        self._f_opt = float(self.f_grid[best_idx].item())

    @property
    def f_opt(self) -> float:
        return self._f_opt

    @property
    def x_opt(self):
        return self._x_opt

    def _point_tensor(self, x) -> torch.Tensor:
        point = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        if point.numel() != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}D point, got shape {tuple(point.shape)}.")
        return point

    def _nearest_idx(self, x) -> int:
        if self.support != "grid":
            raise RuntimeError("_nearest_idx is only valid for grid support.")
        point = self._point_tensor(x)
        if self.input_dim == 1 and self.x_grid.ndim == 1:
            return int((self.x_grid - point[0]).abs().argmin().item())

        grid = self.x_grid.reshape(-1, self.input_dim)
        distances = torch.linalg.norm(grid - point, dim=-1)
        return int(distances.argmin().item())

    def _format_point(self, x: torch.Tensor):
        point = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        if self.input_dim == 1:
            return float(point[0].item())
        return tuple(float(v) for v in point.tolist())

    def f_at(self, x) -> float:
        point = self._point_tensor(x)

        if self.gp_function is not None:
            value = self.gp_function.evaluate(point.reshape(1, self.input_dim))
            return float(value.reshape(-1)[0].item())

        idx = self._nearest_idx(point)
        return float(self.f_grid[idx].item())

    def compare(self, x1, x2):
        f1 = self.f_at(x1)
        f2 = self.f_at(x2)

        if self.noise_std > 0:
            f1 = f1 + self.noise_std * torch.randn((), generator=self._rng).item()
            f2 = f2 + self.noise_std * torch.randn((), generator=self._rng).item()

        if f1 >= f2:
            return self._format_point(self._point_tensor(x1)), self._format_point(
                self._point_tensor(x2)
            )
        return self._format_point(self._point_tensor(x2)), self._format_point(
            self._point_tensor(x1)
        )

    def simple_regret(self, x_recommended) -> float:
        return max(self.f_opt - self.f_at(x_recommended), 0.0)


def _make_grid_points(
    *,
    n_grid: int,
    input_dim: int,
    grid_design: str,
    grid_seed: int,
) -> torch.Tensor:
    if input_dim == 1:
        return make_1d_grid(n_grid, design=grid_design, seed=grid_seed, dtype=torch.double)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(grid_seed))
    if grid_design == "lhs":
        base = torch.arange(n_grid, dtype=torch.double).unsqueeze(-1)
        offsets = torch.rand(n_grid, input_dim, generator=generator, dtype=torch.double)
        x_kernel = (base + offsets) / float(n_grid)
        for dim in range(input_dim):
            perm = torch.randperm(n_grid, generator=generator)
            x_kernel[:, dim] = x_kernel[perm, dim]
        return x_kernel

    if grid_design == "uniform":
        return torch.rand(n_grid, input_dim, generator=generator, dtype=torch.double)

    raise ValueError(f"Unknown grid design {grid_design!r}.")


def sample_gp_function(
    *,
    n_grid: int = 500,
    input_dim: int = 1,
    hparams: GPHyperparameters,
    seed: int,
    grid_design: str = "uniform",
    grid_seed: int = 0,
    jitter: float = 1e-6,
    support: str = "grid",
    rff_num_features: int = 4096,
    opt_reference_size: int = 65536,
    opt_reference_seed: Optional[int] = None,
    rff_eval_batch_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor] | SampledGPFunction:
    """Sample a noiseless latent GP benchmark."""
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}")

    if support == "grid":
        x_eval = _make_grid_points(
            n_grid=n_grid,
            input_dim=input_dim,
            grid_design=grid_design,
            grid_seed=grid_seed,
        )
        x_kernel = x_eval.reshape(n_grid, input_dim)

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

    if support != "continuous_rff":
        raise ValueError(f"Unknown GP function support {support!r}.")

    if rff_num_features < 1:
        raise ValueError("rff_num_features must be positive.")
    if opt_reference_size < 1:
        raise ValueError("opt_reference_size must be positive.")

    path_rng = torch.Generator(device="cpu")
    path_rng.manual_seed(int(seed))

    rff_weights = (
        torch.randn(input_dim, rff_num_features, generator=path_rng)
        / float(hparams.lengthscale)
    )
    rff_phases = 2.0 * math.pi * torch.rand(rff_num_features, generator=path_rng)
    rff_coeffs = torch.randn(rff_num_features, generator=path_rng)
    rff_scale = math.sqrt(2.0 * float(hparams.outputscale) / float(rff_num_features))

    sampled = SampledGPFunction(
        support="continuous_rff",
        input_dim=input_dim,
        hparams=hparams,
        seed=int(seed),
        rff_weights=rff_weights.float(),
        rff_phases=rff_phases.float(),
        rff_coeffs=rff_coeffs.float(),
        rff_scale=rff_scale,
    )

    if opt_reference_seed is None:
        opt_reference_seed = int(seed) + 1_000_003

    opt_rng = torch.Generator(device="cpu")
    opt_rng.manual_seed(int(opt_reference_seed))

    n_restarts = min(int(opt_reference_size), 64)
    starts = torch.rand(n_restarts, input_dim, generator=opt_rng).clamp(1e-6, 1 - 1e-6)
    starts[0].fill_(0.5)

    raw_x = torch.logit(starts).detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [raw_x],
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        x = raw_x.sigmoid()
        values = sampled.evaluate(x, batch_size=rff_eval_batch_size)
        loss = -values.sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        x_reference = raw_x.sigmoid().clamp(0.0, 1.0)
        f_reference = sampled.evaluate(x_reference, batch_size=rff_eval_batch_size)
        best_idx = int(f_reference.argmax().item())
        best_x = x_reference[best_idx]

    x_opt = (
        float(best_x[0].item())
        if input_dim == 1
        else tuple(float(v) for v in best_x.tolist())
    )

    return replace(
        sampled,
        x_opt=x_opt,
        f_opt=float(f_reference[best_idx].item()),
    )

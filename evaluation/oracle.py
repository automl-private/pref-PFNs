"""
Oracle and test functions for preference-based BO evaluation.

The oracle knows the true f(x) but only reveals pairwise comparisons.
All functions operate on x in [0, 1]^d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, dataclass, replace
from typing import Callable

import gpytorch
import torch

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from evaluation.grid_designs import make_1d_grid


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def gp_sample(
    n_grid: int = 500,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a 1D GP function on a fixed grid. Returns (x_grid, f_grid).
    x_grid: (n_grid,), f_grid: (n_grid,)
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    x = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=torch.double)

    base_kernel = gpytorch.kernels.RBFKernel()
    base_kernel.lengthscale = lengthscale
    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    covar_module.outputscale = outputscale

    mean = torch.zeros(n_grid, dtype=torch.double, device=device)
    cov = covar_module(x.unsqueeze(-1)).add_jitter(1e-6).evaluate()
    dist = torch.distributions.MultivariateNormal(mean, cov)

    with torch.no_grad():
        f = dist.sample()
        if noise_std > 0:
            f = f + noise_std * torch.randn(n_grid, generator=rng, dtype=torch.double, device=device)

    return x.float(), f.float()


def branin_1d(n_grid: int = 500, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D slice of Branin at x2=0.5, rescaled to [0,1].
    """
    x = torch.linspace(0.0, 1.0, n_grid, device=device)
    # Branin domain: x1 in [-5, 10]
    x1 = x * 15.0 - 5.0
    x2 = torch.full_like(x1, 7.5)  # midpoint of [0, 15]
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
    # flip sign (Branin is a minimization problem, we maximize)
    f = -f
    return x, f


def sinusoidal_1d(n_grid: int = 500, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Simple multimodal 1D function."""
    x = torch.linspace(0.0, 1.0, n_grid, device=device)
    f = torch.sin(6 * math.pi * x) + 0.5 * torch.sin(2 * math.pi * x) + 0.1 * x
    return x, f


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

@dataclass
class Oracle:
    """
    Wraps a discrete 1D function defined on a fixed candidate grid.

    comparisons are noisy: the oracle flips the answer with probability
    `noise_prob` (simulates human preference noise).
    """
    x_grid: torch.Tensor       # (n_grid,)
    f_grid: torch.Tensor       # (n_grid,) — true latent utility
    noise_prob: float = 0.0    # probability of flipping the comparison
    seed: int = 0
    _rng: torch.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = torch.Generator()
        self._rng.manual_seed(self.seed)
        assert self.x_grid.shape == self.f_grid.shape

    # ------------------------------------------------------------------
    # ground truth
    # ------------------------------------------------------------------

    @property
    def x_opt(self) -> float:
        return self.x_grid[self.f_grid.argmax()].item()

    @property
    def f_opt(self) -> float:
        return self.f_grid.max().item()

    def f_at(self, x: float) -> float:
        """Nearest-neighbour lookup of f(x) on the grid."""
        idx = (self.x_grid - x).abs().argmin()
        return self.f_grid[idx].item()

    def snap_to_grid(self, x: float) -> float:
        """Return the nearest grid point to x."""
        idx = (self.x_grid - x).abs().argmin()
        return self.x_grid[idx].item()

    # ------------------------------------------------------------------
    # comparison
    # ------------------------------------------------------------------

    def compare(self, x1: float, x2: float) -> tuple[float, float]:
        """
        Returns (winner, loser).  May flip with probability noise_prob.
        """
        f1, f2 = self.f_at(x1), self.f_at(x2)
        if f1 >= f2:
            winner, loser = x1, x2
        else:
            winner, loser = x2, x1

        if self.noise_prob > 0:
            flip = torch.rand(1, generator=self._rng).item() < self.noise_prob
            if flip:
                winner, loser = loser, winner

        return winner, loser

    def simple_regret(self, x_recommended: float) -> float:
        return self.f_opt - self.f_at(x_recommended)


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def make_benchmarks(
    n_gp_seeds: int = 5,
    n_grid: int = 500,
    noise_prob: float = 0.05,
    gp_kwargs: dict | None = None,
    device: str = "cpu",
) -> list[dict]:
    """
    Returns a list of benchmark dicts:
        {"name": str, "oracle": Oracle}
    """
    gp_kwargs = gp_kwargs or {}
    benchmarks = []

    for seed in range(n_gp_seeds):
        x, f = gp_sample(n_grid=n_grid, seed=seed, device=device, **gp_kwargs)
        benchmarks.append({
            "name": f"gp_seed{seed}",
            "oracle": Oracle(x_grid=x, f_grid=f, noise_prob=noise_prob, seed=seed),
        })

    for name, fn in [("branin_1d", branin_1d), ("sinusoidal_1d", sinusoidal_1d)]:
        x, f = fn(n_grid=n_grid, device=device)
        benchmarks.append({
            "name": name,
            "oracle": Oracle(x_grid=x, f_grid=f, noise_prob=noise_prob),
        })

    return benchmarks

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
    functions: tuple[Union[tuple[torch.Tensor, torch.Tensor], "SampledGPFunction"], ...]
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
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.input_dim) # [N, input_dim]

        if self.support == "grid":
            if self.x_grid is None or self.f_grid is None:
                raise ValueError("Grid GP function requires x_grid and f_grid.")
            grid = self.x_grid.reshape(-1, self.input_dim)
            # ближайший сосед
            nearest = torch.cdist(x, grid).argmin(dim=1) # [N, len(grid)], для каждого x находит индекс ближайшей точки сетки
            return self.f_grid[nearest]

        if self.support != "continuous_rff":
            raise ValueError(f"Unknown GP function support {self.support!r}.")
        if self.rff_weights is None or self.rff_phases is None or self.rff_coeffs is None:
            raise ValueError("Continuous RFF GP function is missing RFF parameters.")

        values: List[torch.Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            # Разбивает x на порции размера batch_size (кроме последней)
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
    support: str = "grid",
    rff_num_features: int = 4096,
    opt_reference_size: int = 65536,
    opt_reference_seed: Optional[int] = None,
    rff_eval_batch_size: int = 2048,
) -> Union[Tuple[torch.Tensor, torch.Tensor], SampledGPFunction]:
    """Sample a noiseless latent GP benchmark.

    support="grid" preserves the old exact finite-grid GP sample.
    support="continuous_rff" returns an approximate continuous RBF GP path.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if support == "grid":
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
    
    if support != "continuous_rff":
        raise ValueError(f"Unknown GP function support {support!r}.")

    if rff_num_features < 1:
        # количество базисных функций
        raise ValueError("rff_num_features must be positive.")
    if opt_reference_size < 1:
        # размер опорной выборки для поиска оптимума
        raise ValueError("opt_reference_size must be positive.")

    path_rng = torch.Generator(device="cpu")
    path_rng.manual_seed(int(seed))

    # Для RBF-ядра спектральная плотность – гауссовская
    # Генерируем матрицу [input_dim, rff_num_features], каждый элемент ~ N(0,1), затем делим на длину масштаба lengthscale
    # спектральная плотность RBF – тоже гауссовская с обратной ковариацией
    # Спектральная плотность RBF — это преобразование Фурье экспоненциально-квадратичной ковариационной функции.
    rff_weights = (
        torch.randn(input_dim, rff_num_features, generator=path_rng)
        / float(hparams.lengthscale)
    ) # [input_dim, rff_num_features]
    # Генерируем случайные сдвиги равномерно на [0,2π].
    rff_phases = 2.0 * math.pi * torch.rand(rff_num_features, generator=path_rng) # [rff_num_features]
    # независимая стандартная нормальная случайная величина
    rff_coeffs = torch.randn(rff_num_features, generator=path_rng)
    # sqrt(2 / GP variance)
    rff_scale = math.sqrt(2.0 * float(hparams.outputscale) / float(rff_num_features))

    # x_grid, f_grid остаются None
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

    # последовательность Соболя для 
    sobol = torch.quasirandom.SobolEngine(
        dimension=input_dim,
        scramble=True,
        seed=int(opt_reference_seed),
    )
    x_reference = sobol.draw(int(opt_reference_size)).float()
    # Вычисление значений GP на этих точках
    f_reference = sampled.evaluate(x_reference, batch_size=rff_eval_batch_size)

    # Поиск точки с максимальным значением
    best_idx = int(f_reference.argmax().item())
    best_x = x_reference[best_idx]
    # Для одномерного случая возвращается число с плавающей точкой; для многомерного – кортеж чисел
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

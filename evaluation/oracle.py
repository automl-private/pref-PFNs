"""
Oracle and test functions for preference-based BO evaluation.

The oracle knows the true f(x) but only reveals pairwise comparisons.
All functions operate on x in [0, 1]^d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from typing import Dict, List, Optional

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
    functions: tuple["SampledGPFunction", ...]
    oracle_noise_std: float
    baseline_hparams: GPHyperparameters
    eval_hparams: Optional[GPHyperparameters]
    benchmark: Optional[Dict]

@dataclass(frozen=True)
class SampledGPFunction:
    input_dim: int
    hparams: GPHyperparameters
    seed: int
    rff_weights: Optional[torch.Tensor] = None
    rff_phases: Optional[torch.Tensor] = None
    rff_coeffs: Optional[torch.Tensor] = None
    rff_scale: float = 1.0
    x_opt: Optional[object] = None
    f_opt: Optional[float] = None

    def evaluate(self, x, *, batch_size: int = 2048) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.input_dim) # [N, input_dim]

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
    """Oracle with Gaussian comparison noise for continuous RFF GP functions."""

    def __init__(
        self,
        noise_std: float = 0.0,
        seed: int = 0,
        *,
        gp_function: SampledGPFunction,
    ) -> None:
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)
        self.gp_function = gp_function

        if gp_function.x_opt is None or gp_function.f_opt is None:
            raise ValueError("SampledGPFunction must contain x_opt and f_opt.")

        self.support = "continuous_rff"
        self.input_dim = int(gp_function.input_dim)
        self._x_opt = gp_function.x_opt
        self._f_opt = float(gp_function.f_opt)

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

    def _format_point(self, x: torch.Tensor):
        point = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        if self.input_dim == 1:
            return float(point[0].item())
        return tuple(float(v) for v in point.tolist())

    def f_at(self, x) -> float:
        point = self._point_tensor(x)
        value = self.gp_function.evaluate(point.reshape(1, self.input_dim))
        return float(value.reshape(-1)[0].item())

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
    input_dim: int = 1,
    hparams: GPHyperparameters,
    seed: int,
    rff_num_features: int = 4096,
    opt_reference_size: int = 65536,
    opt_reference_seed: Optional[int] = None,
    rff_eval_batch_size: int = 2048,
) -> SampledGPFunction:
    """Sample a noiseless continuous RFF approximation to an RBF GP path."""
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}")

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

    sampled = SampledGPFunction(
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

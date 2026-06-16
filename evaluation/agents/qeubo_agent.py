"""
qEUBO-агент (Astudillo & Frazier 2023).

Модель: PairwiseGP с RBF-ядром и probit likelihood из BoTorch.
Выбор пары: qExpectedUtilityOfBestOption, оптимизированная либо по дискретному
candidate_pool, либо непрерывно на [0, 1]^d.
Рекомендация: максимум posterior mean по candidate_pool или непрерывной области.

Требуется: pip install botorch
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import PBOAgent, Comparison, candidate_matrix, candidate_value

try:
    from botorch.models.pairwise_gp import (
        PairwiseGP,
        PairwiseLaplaceMarginalLogLikelihood,
    )
    from botorch.acquisition.preference import qExpectedUtilityOfBestOption
    from botorch.acquisition.analytic import PosteriorMean
    from botorch.optim import optimize_acqf, optimize_acqf_discrete
    from botorch.fit import fit_gpytorch_mll

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


def _require_botorch():
    """Проверяет, что BoTorch доступен, и дает понятную ошибку иначе."""
    if not _BOTORCH_AVAILABLE:
        raise ImportError(
            "botorch is required for qEUBO agent.\n"
            "Install with:  pip install botorch"
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_pairwise_tensors(
    comparisons: list[Comparison],
    dtype=torch.float64,
) -> tuple[Tensor, Tensor]:
    """
    Преобразует историю сравнений в формат `PairwiseGP`.

    На входе список пар `(winner_x, loser_x)`, где точки могут быть скалярами
    или многомерными tuple/list/tensor. Функция собирает все уникальные точки
    в порядке первого появления и кодирует каждое сравнение индексами этих
    точек.

    Возвращает:
        datapoints: shape `(n_unique, d)`, все уникальные наблюденные точки.
        comp_idx: shape `(m, 2)`, индексы `[winner_idx, loser_idx]`.
    """
    def key(point):
        """Делает из точки hashable tuple-ключ с фиксированным dtype."""
        tensor = torch.as_tensor(point, dtype=dtype).reshape(-1)
        return tuple(float(v) for v in tensor.tolist())

    seen: dict[tuple[float, ...], int] = {}
    for w, l in comparisons:
        wk = key(w)
        lk = key(l)
        if wk not in seen:
            seen[wk] = len(seen)
        if lk not in seen:
            seen[lk] = len(seen)

    datapoints = torch.tensor(
        sorted(seen.keys(), key=lambda x: seen[x]),
        dtype=dtype,
    )  # (n, d)

    comp_idx = torch.tensor(
        [[seen[key(w)], seen[key(l)]] for w, l in comparisons],
        dtype=torch.long,
    )  # (m, 2)

    return datapoints, comp_idx


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QEUBOAgent(PBOAgent):
    """
    qEUBO baseline поверх BoTorch `PairwiseGP`.

    Агент поддерживает два режима:
    - `support="grid"`: старая логика, qEUBO оптимизируется по `candidate_pool`.
    - `support="continuous_rff"`: qEUBO и posterior mean оптимизируются
      непрерывно на области `[0, 1]^d`.

    Args:
        fit_hyperparams: Оптимизировать ли GP hyperparameters на каждом шаге.
        max_fit_iter: Максимум итераций для fit hyperparameters.
        num_acqf_samples: Число MC-сэмплов для оценки qEUBO.
        dtype: dtype для GP-тензоров; для GP обычно устойчивее `float64`.
        support: Режим поиска: `"grid"` или `"continuous_rff"`.
        continuous_num_restarts: Число рестартов для `optimize_acqf`.
        continuous_raw_samples: Число raw samples для выбора стартов optimizer.
        continuous_maxiter: Максимум итераций continuous optimizer.
        min_pair_distance: Минимальная допустимая дистанция между двумя точками
            в предложенной паре.
    """

    def __init__(
        self,
        fit_hyperparams: bool = True,
        max_fit_iter: int = 100,
        num_acqf_samples: int = 512,
        dtype=torch.float64,
        support: str = "grid", # выбирает старую grid-логику или continuous optimizer
        continuous_num_restarts: int = 10,  # число стартов для optimize_acqf
        continuous_raw_samples: int = 256, # число raw samples для выбора стартовых точек
        continuous_maxiter: int = 100, # лимит итераций optimizer.
        min_pair_distance: float = 1e-6, # защита от пары (x, x)
    ):
        """Инициализирует qEUBO-агента и параметры grid/continuous оптимизации."""
        _require_botorch()
        self.fit_hyperparams = fit_hyperparams
        self.max_fit_iter = max_fit_iter
        self.num_acqf_samples = num_acqf_samples
        self.dtype = dtype
        self.support = support
        self.continuous_num_restarts = int(continuous_num_restarts)
        self.continuous_raw_samples = int(continuous_raw_samples)
        self.continuous_maxiter = int(continuous_maxiter)
        self.min_pair_distance = float(min_pair_distance)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_model(self, comparisons: list[Comparison]) -> "PairwiseGP":
        """
        Строит и обучает `PairwiseGP` на уже наблюденных сравнениях.

        Важный момент для continuous режима: модель фитится только на реальных
        точках из `comparisons`, а не на текущем `candidate_pool`.
        """
        datapoints, comp_idx = _build_pairwise_tensors(comparisons, dtype=self.dtype)

        model = PairwiseGP(datapoints, comp_idx)
        model.train()

        if self.fit_hyperparams:
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, max_attempts=1, options={"maxiter": self.max_fit_iter})

        model.eval()
        return model

    def _posterior_mean(self, model: "PairwiseGP", candidate_pool: Tensor) -> Tensor:
        """
        Считает posterior mean `f(x)` для каждой точки из `candidate_pool`.

        Используется в grid-режиме `recommend`, где рекомендация остается
        argmax по дискретному candidate pool.

        Args:
            model: Обученная preference GP модель.
            candidate_pool: Тензор shape `(n,)` или `(n, d)`.

        Returns:
            Tensor shape `(n,)` со средним posterior utility.
        """
        X = candidate_matrix(candidate_pool, dtype=self.dtype)
        with torch.no_grad():
            posterior = model.posterior(X)
        return posterior.mean.squeeze(-1).squeeze(-1)  # (n,)

    def _bounds_from_pool(self, candidate_pool: Tensor) -> Tensor:
        """
        Строит bounds `[0, 1]^d` для continuous optimizer.

        Размерность `d` берется из формы `candidate_pool`, чтобы не добавлять
        отдельный параметр `input_dim` в интерфейс агента.
        """
        X = candidate_matrix(candidate_pool, dtype=self.dtype)
        input_dim = X.shape[-1]
        return torch.stack(
            [
                torch.zeros(input_dim, dtype=self.dtype, device=X.device),
                torch.ones(input_dim, dtype=self.dtype, device=X.device),
            ]
        )

    def _make_qeubo_acqf(self, model: "PairwiseGP"):
        """
        Создает qEUBO acquisition function для обученной `PairwiseGP`.

        Один и тот же acquisition используется в grid-ветке
        `optimize_acqf_discrete` и в continuous-ветке `optimize_acqf`.
        """
        from botorch.sampling.normal import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))
        return qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

    def _optimize_qeubo_continuous(
        self,
        model: "PairwiseGP",
        candidate_pool: Tensor,
    ) -> Tensor:
        """
        Непрерывно оптимизирует qEUBO для пары точек.

        Решает задачу `argmax_{x1, x2 in [0, 1]^d} qEUBO(x1, x2)` через
        BoTorch `optimize_acqf` с `q=2`. `candidate_pool` здесь нужен только
        для определения размерности области поиска.
        """
        acqf = self._make_qeubo_acqf(model)
        bounds = self._bounds_from_pool(candidate_pool)

        X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=2,
            num_restarts=self.continuous_num_restarts,
            raw_samples=self.continuous_raw_samples,
            options={"maxiter": self.continuous_maxiter},
        )
        return X_next.detach()

    def _optimize_mean_continuous(
        self,
        model: "PairwiseGP",
        candidate_pool: Tensor,
    ) -> Tensor:
        """
        Непрерывно оптимизирует posterior mean для рекомендации.

        Решает `argmax_{x in [0, 1]^d} E[f(x) | data]` через аналитический
        BoTorch acquisition `PosteriorMean`.
        """
        acqf = PosteriorMean(model)
        bounds = self._bounds_from_pool(candidate_pool)

        X_best, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=self.continuous_num_restarts,
            raw_samples=self.continuous_raw_samples,
            options={"maxiter": self.continuous_maxiter},
        )
        return X_best.detach()[0]

    def _ensure_distinct_pair(
        self,
        X_pair: Tensor,
        candidate_pool: Tensor,
    ) -> Tensor:
        """
        Гарантирует, что continuous optimizer не вернул пару почти одинаковых точек.

        Если расстояние между точками меньше `min_pair_distance`, вторая точка
        заменяется случайным challenger из `[0, 1]^d`.
        """
        if torch.linalg.norm(X_pair[0] - X_pair[1]) >= self.min_pair_distance:
            return X_pair

        bounds = self._bounds_from_pool(candidate_pool)
        challenger = bounds[0] + torch.rand_like(bounds[0]) * (bounds[1] - bounds[0])
        X_pair = X_pair.clone()
        X_pair[1] = challenger
        return X_pair

    # ------------------------------------------------------------------
    # PBOAgent interface
    # ------------------------------------------------------------------

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple:
        """
        Возвращает следующую пару точек для сравнения.

        В grid-режиме оптимизирует qEUBO по `candidate_pool`. В continuous
        режиме оптимизирует qEUBO напрямую на `[0, 1]^d`. Если истории
        сравнений еще нет, возвращает случайную пару из `candidate_pool`.
        """
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)

        if self.support == "grid":
            # Старое поведение: qEUBO ищется только среди точек candidate_pool.
            acqf = self._make_qeubo_acqf(model)
            choices = candidate_matrix(candidate_pool, dtype=self.dtype)

            X_next, _ = optimize_acqf_discrete(
                acq_function=acqf,
                q=2,
                choices=choices,
                unique=True,
            )  # X_next: (2, d)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        if self.support == "continuous_rff":
            # Новое поведение: candidate_pool задает только размерность, не сетку поиска.
            X_next = self._optimize_qeubo_continuous(model, candidate_pool)
            X_next = self._ensure_distinct_pair(X_next, candidate_pool)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        raise ValueError(f"Unknown QEUBOAgent support {self.support!r}.")

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ):
        """
        Возвращает текущую рекомендацию лучшей точки.

        В grid-режиме выбирает argmax posterior mean среди `candidate_pool`.
        В continuous режиме оптимизирует posterior mean на `[0, 1]^d`.
        """
        if not comparisons:
            return candidate_value(candidate_pool[len(candidate_pool) // 2])

        model = self._fit_model(comparisons)

        if self.support == "grid":
            # Старое поведение: рекомендация выбирается только из candidate_pool.
            mean = self._posterior_mean(model, candidate_pool)
            return candidate_value(candidate_pool[mean.argmax()])

        if self.support == "continuous_rff":
            # Новое поведение: рекомендация ищется непрерывно на [0, 1]^d.
            X_best = self._optimize_mean_continuous(model, candidate_pool)
            return candidate_value(X_best)

        raise ValueError(f"Unknown QEUBOAgent support {self.support!r}.")

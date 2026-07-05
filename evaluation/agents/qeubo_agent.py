"""
qEUBO-агент (Astudillo & Frazier 2023).

Модель: PairwiseGP с RBF-ядром и probit likelihood из BoTorch.
Выбор пары: qExpectedUtilityOfBestOption, оптимизированная по дискретному
candidate_pool или непрерывно на [0, 1]^d.
Рекомендация: максимум posterior mean по candidate_pool или непрерывной области.

Требуется: pip install botorch
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import (
    PBOAgent,
    Comparison,
    Point,
    _build_pairwise_tensors,
    candidate_matrix,
    candidate_value,
)

try:
    from botorch.models.pairwise_gp import (
        PairwiseGP,
        PairwiseLaplaceMarginalLogLikelihood,
    )
    from botorch.acquisition.preference import qExpectedUtilityOfBestOption
    from botorch.acquisition.analytic import PosteriorMean
    from botorch.acquisition import qNoisyExpectedImprovement
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.generation import MaxPosteriorSampling
    from botorch.optim import optimize_acqf, optimize_acqf_discrete
    from botorch.fit import fit_gpytorch_mll
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.sampling import draw_sobol_samples

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
# Agent
# ---------------------------------------------------------------------------

class QEUBOAgent(PBOAgent):
    """
    qEUBO baseline поверх BoTorch `PairwiseGP`.

    qEUBO и posterior mean оптимизируются либо на дискретном `candidate_pool`,
    либо непрерывно на области `[0, 1]^d`.

    Args:
        fit_hyperparams: Оптимизировать ли GP hyperparameters на каждом шаге.
        max_fit_iter: Максимум итераций для fit hyperparameters.
        num_acqf_samples: Число MC-сэмплов для оценки qEUBO.
        dtype: dtype для GP-тензоров; для GP обычно устойчивее `float64`.
        device: Устройство для BoTorch/GPyTorch тензоров.
        support: `"grid"` или `"continuous_rff"`.
        continuous_num_restarts: Число рестартов для `optimize_acqf`.
        continuous_raw_samples: Число raw samples для выбора стартов optimizer.
        continuous_maxiter: Максимум итераций continuous optimizer.
        min_pair_distance: Минимальная допустимая дистанция между двумя точками
            в предложенной паре.
        gp_lengthscale: RBF lengthscale для случая `fit_hyperparams=False`.
        gp_outputscale: Kernel outputscale для случая `fit_hyperparams=False`.
    """

    def __init__(
        self,
        fit_hyperparams: bool = True,
        max_fit_iter: int = 100,
        num_acqf_samples: int = 512,
        dtype=torch.float64,
        device: str | torch.device = "cpu",
        support: str = "grid",
        continuous_num_restarts: int = 10,  # число стартов для optimize_acqf
        continuous_raw_samples: int = 256, # число raw samples для выбора стартовых точек
        continuous_maxiter: int = 100, # лимит итераций optimizer.
        min_pair_distance: float = 1e-6, # защита от пары (x, x)
        max_fit_attempts: int | None = None,
        gp_lengthscale: float | None = None,
        gp_outputscale: float | None = None,
    ):
        """Инициализирует qEUBO-агента и параметры grid/continuous оптимизации."""
        _require_botorch()
        self.fit_hyperparams = fit_hyperparams
        self.max_fit_iter = max_fit_iter
        if not self.fit_hyperparams and (gp_lengthscale is None or gp_outputscale is None):
            raise ValueError(
                "QEUBOAgent requires gp_lengthscale and gp_outputscale "
                "when fit_hyperparams=False."
            )
        self.gp_lengthscale = None if gp_lengthscale is None else float(gp_lengthscale)
        self.gp_outputscale = None if gp_outputscale is None else float(gp_outputscale)
        self.num_acqf_samples = num_acqf_samples
        self.dtype = dtype
        self.device = torch.device(device)
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
        datapoints, comp_idx = _build_pairwise_tensors(
            comparisons,
            dtype=self.dtype,
            device=self.device,
        )

        model = PairwiseGP(datapoints, comp_idx).to(self.device)
        model.train()

        if self.fit_hyperparams:
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, max_attempts=1, options={"maxiter": self.max_fit_iter})
        else:
            self._apply_kernel_hparams(model)

        model.eval()
        return model

    def _apply_kernel_hparams(self, model: "PairwiseGP") -> None:
        """
        Вставляет fixed kernel hyperparameters в `PairwiseGP`.

        Эта ветка используется только когда `fit_hyperparams=False`: тогда GP
        не должен оставаться на BoTorch defaults, а получает параметры из config.
        После смены kernel parameters нужно обновить cached quantities модели,
        потому что `PairwiseGP.__init__` уже построил их на прежних defaults.
        """
        lengthscale = torch.full_like(
            model.covar_module.base_kernel.lengthscale,
            self.gp_lengthscale,
        )
        model.covar_module.base_kernel.lengthscale = lengthscale

        outputscale = torch.as_tensor(
            self.gp_outputscale,
            dtype=self.dtype,
            device=model.datapoints.device,
        )
        model.covar_module.outputscale = outputscale

        transformed_dp = model.transform_inputs(model.datapoints)
        model._update(transformed_dp)

    def _posterior_mean(self, model: "PairwiseGP", candidate_pool: Tensor) -> Tensor:
        """
        Считает posterior mean в точках из `candidate_pool`.

        Для grid и continuous support используется один и тот же путь:
        сначала точки приводятся к матрице `(M, d)`, затем явно вызывается
        `model.posterior(X)`.
        """
        X = candidate_matrix(candidate_pool, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            posterior = model.posterior(X)
        return posterior.mean.squeeze(-1).squeeze(-1)

    def _bounds_from_pool(self, candidate_pool: Tensor) -> Tensor:
        """
        Строит bounds `[0, 1]^d` для continuous optimizer.

        Размерность `d` берется из формы `candidate_pool`, чтобы не добавлять
        отдельный параметр `input_dim` в интерфейс агента.
        """
        X = candidate_matrix(candidate_pool, dtype=self.dtype, device=self.device)
        input_dim = X.shape[-1]
        return torch.stack(
            [
                torch.zeros(input_dim, dtype=self.dtype, device=X.device),
                torch.ones(input_dim, dtype=self.dtype, device=X.device),
            ]
        )

    def _make_sampler(self):
        """Создает Sobol QMC sampler для subclasses в этом файле."""
        return SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))

    def _make_qeubo_acqf(self, model: "PairwiseGP"):
        """
        Создает qEUBO acquisition function для обученной `PairwiseGP`.
        """
        from botorch.sampling.normal import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_acqf_samples]))
        return qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

    def _optimize_acqf_continuous(
        self,
        acqf,
        candidate_pool: Tensor,
        *,
        q: int,
    ) -> Tensor:
        """
        Оптимизирует BoTorch acquisition function на `[0, 1]^d`.

        Этот helper повторяет continuous ветку PABBO `pbo.py`: acquisition
        оптимизируется через `optimize_acqf` с заданными raw samples и restarts.
        """
        bounds = self._bounds_from_pool(candidate_pool)

        X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=self.continuous_num_restarts,
            raw_samples=self.continuous_raw_samples,
            options={"maxiter": self.continuous_maxiter},
        )
        return X_next.detach()

    def _optimize_acqf_grid(
        self,
        acqf,
        candidate_pool: Tensor,
        *,
        q: int,
    ) -> Tensor:
        """Optimizes an acquisition function over the finite candidate pool."""
        choices = candidate_matrix(candidate_pool, dtype=self.dtype, device=self.device)
        X_next, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=q,
            choices=choices,
            unique=True,
        )
        return X_next.detach()

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
    ) -> tuple[Point, Point]:
        """
        Возвращает следующую пару точек для сравнения.

        В grid режиме оптимизирует qEUBO по `candidate_pool`. В continuous
        режиме оптимизирует qEUBO напрямую на `[0, 1]^d`.
        """
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)

        if self.support == "grid":
            acqf = self._make_qeubo_acqf(model)
            X_next = self._optimize_acqf_grid(acqf, candidate_pool, q=2)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        if self.support == "continuous_rff":
            X_next = self._optimize_qeubo_continuous(model, candidate_pool)
            X_next = self._ensure_distinct_pair(X_next, candidate_pool)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        raise ValueError(f"Unknown QEUBOAgent support {self.support!r}.")

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> Point:
        """
        Возвращает текущую рекомендацию лучшей точки.

        В обоих режимах выбирает argmax posterior mean среди текущего
        `candidate_pool`. Для continuous support этот pool является свежей
        случайной аппроксимацией непрерывной области.
        """
        if not comparisons:
            return candidate_value(candidate_pool[len(candidate_pool) // 2])

        model = self._fit_model(comparisons)

        if self.support not in {"grid", "continuous_rff"}:
            raise ValueError(f"Unknown QEUBOAgent support {self.support!r}.")

        posterior_mean = self._posterior_mean(model, candidate_pool)
        best_idx = int(posterior_mean.argmax().item())
        return candidate_value(candidate_pool[best_idx])


class QEIAgent(QEUBOAgent):
    """
    PABBO-style qEI baseline on top of BoTorch `PairwiseGP`.

    The acquisition is `qExpectedImprovement` with `best_f` equal to the maximum
    posterior mean over currently observed datapoints, matching `pbo.py`.
    """

    def _make_qei_acqf(self, model: "PairwiseGP"):
        posterior = model.posterior(model.datapoints)
        best_f = posterior.mean.max().item()
        return qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=self._make_sampler(),
        )

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Point, Point]:
        """Возвращает пару, выбранную qEI acquisition."""
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)
        acqf = self._make_qei_acqf(model)
        if self.support == "grid":
            X_next = self._optimize_acqf_grid(acqf, candidate_pool, q=2)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        if self.support == "continuous_rff":
            X_next = self._optimize_acqf_continuous(acqf, candidate_pool, q=2)
            X_next = self._ensure_distinct_pair(X_next, candidate_pool)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        raise ValueError(f"Unknown QEIAgent support {self.support!r}.")


class QNEIAgent(QEUBOAgent):
    """
    PABBO-style qNEI baseline on top of BoTorch `PairwiseGP`.

    The baseline set is the observed datapoints from the current preference GP.
    """

    def _make_qnei_acqf(self, model: "PairwiseGP"):
        return qNoisyExpectedImprovement(
            model=model,
            X_baseline=model.datapoints,
            sampler=self._make_sampler(),
            prune_baseline=True,
        )

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Point, Point]:
        """Возвращает пару, выбранную qNEI acquisition."""
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)
        acqf = self._make_qnei_acqf(model)
        if self.support == "grid":
            X_next = self._optimize_acqf_grid(acqf, candidate_pool, q=2)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        if self.support == "continuous_rff":
            X_next = self._optimize_acqf_continuous(acqf, candidate_pool, q=2)
            X_next = self._ensure_distinct_pair(X_next, candidate_pool)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        raise ValueError(f"Unknown QNEIAgent support {self.support!r}.")


class QTSAgent(QEUBOAgent):
    """
    PABBO-style qTS baseline on top of BoTorch `PairwiseGP`.

    Each point in the pair is selected by posterior Thompson sampling over a
    fresh Sobol candidate set from `[0, 1]^d`.
    """

    def _thompson_point(self, model: "PairwiseGP", bounds: Tensor) -> Tensor:
        """Samples one posterior maximizer from a Sobol candidate set."""
        choices = draw_sobol_samples(
            bounds=bounds,
            n=self.continuous_raw_samples,
            q=1,
        ).squeeze(-2)
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        return thompson_sampling(choices, num_samples=1).reshape(1, -1)

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Point, Point]:
        """Возвращает пару, выбранную Thompson sampling."""
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        model = self._fit_model(comparisons)
        if self.support == "grid":
            choices = candidate_matrix(candidate_pool, dtype=self.dtype, device=self.device)
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            X_next = thompson_sampling(choices, num_samples=2)
            return candidate_value(X_next[0]), candidate_value(X_next[1])

        if self.support != "continuous_rff":
            raise ValueError(f"Unknown QTSAgent support {self.support!r}.")

        bounds = self._bounds_from_pool(candidate_pool)
        X_next = torch.cat(
            [
                self._thompson_point(model, bounds),
                self._thompson_point(model, bounds),
            ],
            dim=0,
        )
        X_next = self._ensure_distinct_pair(X_next, candidate_pool)
        return candidate_value(X_next[0]), candidate_value(X_next[1])

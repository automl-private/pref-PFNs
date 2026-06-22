"""
PFN-based preference BO agent.

Uses the trained preference PFN model to:
  - recommend: argmax of diagonal pair-score over a continuous candidate pool
  - suggest_pair: shortlist/rerank pair scores on a continuous candidate pool
"""

from __future__ import annotations

import torch

from .base import PBOAgent, Comparison, Point, candidate_value

try:
    from botorch.acquisition.acquisition import AcquisitionFunction as _PBOAcquisitionFunction
    from botorch.optim import optimize_acqf

    _BOTORCH_AVAILABLE = True
except ImportError:
    _PBOAcquisitionFunction = torch.nn.Module
    optimize_acqf = None
    _BOTORCH_AVAILABLE = False


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


class PairScorePFNAgent(PBOAgent):
    """
    PFN-агент для checkpoint'ов, которые напрямую скорят пару точек.

    Работает как batched candidate-set optimizer: сначала скорит диагональ
    `(x, x)` на большом continuous pool, затем rerank'ит полный набор пар
    только на shortlist из top-k и exploration-точек.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        pair_batch_size: int,
        input_dim: int,
        continuous_top_k: int = 64,
        continuous_explore_k: int = 64,
    ) -> None:
        """
        Инициализирует pair-score PFN и параметры continuous shortlist-поиска.

        Args:
            model: Обученная PFN-модель с `criterion`.
            device: Устройство для forward pass'ов PFN.
            pair_batch_size: Сколько пар отправлять в PFN за один batch.
            input_dim: Размерность одной точки; вход пары имеет размер
                `2 * input_dim`.
            continuous_top_k: Сколько лучших точек по диагональному score
                брать в shortlist.
            continuous_explore_k: Сколько случайных exploration-точек добавлять
                в shortlist.
        """
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.criterion = model.criterion
        self.pair_batch_size = int(pair_batch_size)
        self.input_dim = int(input_dim)
        self.continuous_top_k = int(continuous_top_k)
        self.continuous_explore_k = int(continuous_explore_k)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _candidate_matrix(self, candidate_pool: torch.Tensor) -> torch.Tensor:
        """
        Приводит candidate pool к матрице shape `(M, input_dim)`.

        Для одномерного pool shape `(M,)` добавляет последнюю размерность, а для
        многомерного pool сохраняет первую размерность как число кандидатов.
        """
        x = candidate_pool.to(dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            return x.unsqueeze(-1)
        return x.reshape(x.shape[0], -1)

    def _score_pair_tensor(
        self,
        comparisons: list[Comparison],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Скорит произвольный список пар через PFN.

        Args:
            comparisons: История наблюденных сравнений `(winner, loser)`.
            pairs: Tensor shape `(N, 2 * input_dim)`, где каждая строка это
                конкатенация `[x_1, x_2]`.

        Returns:
            Tensor shape `(N,)` с mean score из PFN criterion для каждой пары.
        """
        pairs = pairs.to(dtype=self.dtype, device=self.device)
        expected_dim = self.input_dim * 2
        if pairs.ndim != 2 or pairs.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected pairs with shape (N, {expected_dim}), got {tuple(pairs.shape)}."
            )

        x_ctx, y_ctx = _build_pref_context(
            comparisons,
            dtype=self.dtype,
            device=self.device,
            input_dims=expected_dim,
        )

        batch_size = pairs.shape[0] if self.pair_batch_size <= 0 else self.pair_batch_size
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, pairs.shape[0], batch_size):
                test_x = pairs[start : start + batch_size].unsqueeze(0)
                logits = self.model(x_ctx, y_ctx, test_x=test_x)
                chunks.append(self.criterion.mean(logits)[0].detach().cpu())
        return torch.cat(chunks)

    def _diag_scores(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Считает score для диагональных пар `(x, x)` по всем кандидатам.

        В continuous режиме это дешевый proxy для качества одиночной точки:
        `recommend` берет argmax этих score'ов, а `suggest_pair` использует их
        для отбора shortlist перед полным pairwise reranking.
        """
        x = self._candidate_matrix(candidate_pool)
        pairs = torch.cat([x, x], dim=-1)
        return self._score_pair_tensor(comparisons, pairs)

    def _pair_scores(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Строит и скорит полную матрицу пар из одного candidate pool.

        Возвращает `scores` shape `(M, M)`, где `scores[i, j]` соответствует
        PFN-score пары `(candidate_pool[i], candidate_pool[j])`.
        """
        x = self._candidate_matrix(candidate_pool)
        M = x.shape[0]
        x1 = x.repeat_interleave(M, dim=0)
        x2 = x.repeat(M, 1)
        pairs = torch.cat([x1, x2], dim=-1)
        # 1D x (4,)      x1 (4, 4)   x2 (4, 4)   pairs (16, 2)
        # 2D x (4, 2)   x1 (16, 2)  x2 (16, 2)  pairs (16, 4)
        return self._score_pair_tensor(comparisons, pairs).reshape(M, M)

    def _continuous_shortlist(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Формирует shortlist для continuous pair-search.

        Сначала берет `continuous_top_k` лучших точек по диагональному score
        `(x, x)`, затем добавляет `continuous_explore_k` случайных точек из
        исходного pool и удаляет дубликаты. Это снижает стоимость pair search
        с полного `M x M` до `K x K`.
        """
        M = len(candidate_pool)
        diag_scores = self._diag_scores(comparisons, candidate_pool)
        top_k = min(M, max(1, self.continuous_top_k))
        explore_k = min(M, max(0, self.continuous_explore_k))

        selected = [int(i) for i in torch.topk(diag_scores, k=top_k).indices.tolist()]
        if explore_k > 0:
            selected.extend(int(i) for i in torch.randperm(M)[:explore_k].tolist())

        unique_indices: list[int] = []
        seen = set()
        for idx in selected:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        if len(unique_indices) < min(2, M):
            for idx in torch.randperm(M).tolist():
                idx = int(idx)
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
                if len(unique_indices) >= min(2, M):
                    break

        shortlist_idx = torch.tensor(unique_indices, dtype=torch.long)
        return candidate_pool[shortlist_idx]

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        """
        Выбирает следующую пару для oracle comparison.

        Сначала строит shortlist, затем скорит все пары внутри него и
        возвращает лучшую недиагональную пару.
        """
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        pool = self._continuous_shortlist(comparisons, candidate_pool)
        scores = self._pair_scores(comparisons, pool)

        # scores[i, j] ≈ E[max(f(x_i), f(x_j)) | comparisons]
        # scores[i, j] ≈ E[max(f(x_i), f(x_j)) - f* | comparisons]
        idx = torch.arange(scores.shape[0])
        scores[idx, idx] = -torch.inf
        # Диагональ зануляется
        flat_idx = int(torch.argmax(scores).item())
        # Потом flat index переводится в пару индексов
        i = int(flat_idx // scores.shape[1])
        j = int(flat_idx % scores.shape[1])
        # argmax_{i != j} scores[i, j]
        return candidate_value(pool[i]), candidate_value(pool[j])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        """
        Возвращает текущую рекомендацию лучшей точки.

        Скорит только диагональ `(x, x)` на большом `candidate_pool` и
        возвращает argmax.
        """
        if not comparisons:
            return candidate_value(candidate_pool[candidate_pool.shape[0] // 2])

        diag = self._diag_scores(comparisons, candidate_pool)
        return candidate_value(candidate_pool[diag.argmax()])


class PBOPFN(_PBOAcquisitionFunction):
    """
    Минимальная BoTorch acquisition-функция для pair-score PFN.

    `.fit(...)` только кеширует preference context; веса PFN не меняются.
    `forward(...)` принимает BoTorch tensor shape `batch_shape x q x d`:
    при `q=2` скорит пару `(x1, x2)`, при `q=1` скорит диагональ `(x, x)`.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        pair_batch_size: int,
        input_dim: int,
    ) -> None:
        if _BOTORCH_AVAILABLE:
            super().__init__(model=model)
        else:
            super().__init__()
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.criterion = model.criterion
        self.pair_batch_size = int(pair_batch_size)
        self.input_dim = int(input_dim)
        self._x_ctx: torch.Tensor | None = None
        self._y_ctx: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def fit(self, comparisons: list[Comparison]) -> "PBOPFN":
        """
        Запоминает историю сравнений как in-context данные для PFN.

        Подготавливает контекст, но не обучает модель и не обновляет параметры checkpoint-а.
        """
        self._x_ctx, self._y_ctx = _build_pref_context(
            comparisons,
            dtype=self.dtype,
            device=self.device,
            input_dims=2 * self.input_dim,
        )
        return self

    def score_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Скорит плоский список пар shape `(N, 2 * input_dim)`.

        Возвращает tensor shape `(N,)`. Градиент по `pairs` сохраняется, чтобы
        `optimize_acqf` мог оптимизировать acquisition-функцию.
        """
        if self._x_ctx is None or self._y_ctx is None:
            raise RuntimeError("call .fit(comparisons) before scoring PBOPFN")

        pairs = pairs.to(dtype=self.dtype, device=self.device)
        expected_dim = 2 * self.input_dim
        if pairs.ndim != 2 or pairs.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected pairs with shape (N, {expected_dim}), got {tuple(pairs.shape)}."
            )

        batch_size = pairs.shape[0] if self.pair_batch_size <= 0 else self.pair_batch_size
        chunks: list[torch.Tensor] = []
        for start in range(0, pairs.shape[0], batch_size):
            test_x = pairs[start : start + batch_size].unsqueeze(0) # (1, batch_size, 2d)
            logits = self.model(self._x_ctx, self._y_ctx, test_x=test_x)
            chunks.append(self.criterion.mean(logits).reshape(-1)) # средний predicted score, (batch_size,)
        return torch.cat(chunks, dim=0) # (N,)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Возвращает acquisition score для BoTorch optimizer-а.

        `X` должен иметь shape `batch_shape x q x d`. Для `q=2` две точки
        конкатенируются в одну пару, а для `q=1` точка дублируется в `(x, x)`.
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)
        if X.ndim < 3:
            raise ValueError(f"Expected X with shape (..., q, d), got {tuple(X.shape)}.")
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got X with shape {tuple(X.shape)}."
            )
        if X.shape[-2] not in (1, 2):
            raise ValueError(
                f"PBOPFN supports q=1 or q=2, got X with shape {tuple(X.shape)}."
            )

        X = X.to(dtype=self.dtype, device=self.device)
        batch_shape = X.shape[:-2] # возьми все batch dimensions, кроме последних двух q и d

        # переводит BoTorch-формат X в формат, который понимает pair-score PFN
        # PFN ждёт не две отдельные точки, а одну строку-пару
        # А BoTorch передаёт точки как: batch_shape x q x d
        if X.shape[-2] == 1:
            x = X.squeeze(-2)
            pairs = torch.cat([x, x], dim=-1) # batch_shape x 2d
        else:
            pairs = X.reshape(*batch_shape, 2 * self.input_dim) # batch_shape1 x batch_shape2 x 2d

        flat_pairs = pairs.reshape(-1, 2 * self.input_dim) # batch_shape1*batch_shape2 x 2d
        scores = self.score_pairs(flat_pairs)
        return scores.reshape(batch_shape)


class BoTorchPairPFN(PBOAgent):
    """
    Continuous PBOAgent поверх `PBOPFN`.

    Класс только адаптирует BoTorch-оптимизацию обратно к интерфейсу repo:
    `suggest_pair` возвращает две точки, `recommend` возвращает одну точку.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        pair_batch_size: int,
        input_dim: int,
        continuous_num_restarts: int = 10,
        continuous_raw_samples: int = 128,
        continuous_maxiter: int = 100,
    ) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "botorch is required for BoTorchPairPFN continuous acquisition optimization.\n"
                "Install with:  pip install botorch"
            )
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.pair_batch_size = int(pair_batch_size)
        self.input_dim = int(input_dim)
        self.continuous_num_restarts = int(continuous_num_restarts)
        self.continuous_raw_samples = int(continuous_raw_samples)
        self.continuous_maxiter = int(continuous_maxiter)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _make_acqf(self, comparisons: list[Comparison]) -> PBOPFN:
        """Создает `PBOPFN` и кладет в него текущую историю сравнений."""
        return PBOPFN(
            self.model,
            device=str(self.device),
            pair_batch_size=self.pair_batch_size,
            input_dim=self.input_dim,
        ).fit(comparisons)

    def _bounds(self) -> torch.Tensor:
        """Возвращает continuous границы поиска `[0, 1]^d`."""
        return torch.stack(
            [
                torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
                torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
            ]
        )

    def _optimize_continuous(self, acqf: PBOPFN, *, q: int) -> torch.Tensor:
        """Оптимизирует `PBOPFN` на `[0, 1]^d` для `q=1` или `q=2`."""
        X_best, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self._bounds(),
            q=q,
            num_restarts=self.continuous_num_restarts,
            raw_samples=self.continuous_raw_samples,
            options={"maxiter": self.continuous_maxiter},
        )
        return X_best.detach()

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        """Возвращает следующую пару из continuous оптимизации pair-score PFN."""
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        acqf = self._make_acqf(comparisons)
        X_next = self._optimize_continuous(acqf, q=2)
        return candidate_value(X_next[0]), candidate_value(X_next[1])

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        """Возвращает текущую лучшую точку через continuous оптимизацию `(x, x)`."""
        if not comparisons:
            return candidate_value(candidate_pool[len(candidate_pool) // 2])

        acqf = self._make_acqf(comparisons)
        X_best = self._optimize_continuous(acqf, q=1)[0]
        return candidate_value(X_best)

"""
PFN-based preference BO agent.

Uses the trained preference PFN model to:
  - recommend: argmax of mean posterior E[f(x)] over candidate pool
  - suggest_pair: Thompson Sampling — sample two f draws, compare argmax
"""

from __future__ import annotations

from typing import List

import torch

from .base import PBOAgent, Comparison, Point, candidate_value


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

    В `support="grid"` сохраняет старое поведение: считает полную матрицу
    pair-score'ов на переданном `candidate_pool`. В `support="continuous_rff"`
    работает как batched candidate-set optimizer: сначала скорит диагональ
    `(x, x)` на большом continuous pool, затем rerank'ит полный набор пар только
    на shortlist из top-k и exploration-точек.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        pair_batch_size: int,
        input_dim: int,
        support: str = "grid",
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
            support: `"grid"` для старого полного перебора или
                `"continuous_rff"` для shortlist/reranking по continuous pool.
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
        self.support = support
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
        chunks: List[torch.Tensor] = []
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

        unique_indices: List[int] = []
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

        В grid режиме делает старый полный argmax по матрице pair-score'ов на
        `candidate_pool`. В continuous режиме сначала строит shortlist, затем
        скорит все пары внутри него и возвращает лучшую недиагональную пару.
        """
        if not comparisons:
            idx = torch.randperm(len(candidate_pool))[:2]
            return candidate_value(candidate_pool[idx[0]]), candidate_value(candidate_pool[idx[1]])

        if self.support == "grid":
            scores = self._pair_scores(comparisons, candidate_pool)
            pool = candidate_pool
        elif self.support == "continuous_rff":
            pool = self._continuous_shortlist(comparisons, candidate_pool)
            scores = self._pair_scores(comparisons, pool)
        else:
            raise ValueError(f"Unknown PairScorePFNAgent support {self.support!r}.")

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

        В grid режиме сохраняет старую логику: берет диагональ полной pair-score
        матрицы. В continuous режиме не строит `M x M`; скорит только диагональ
        `(x, x)` на большом `candidate_pool` и возвращает argmax.
        """
        if not comparisons:
            return candidate_value(candidate_pool[candidate_pool.shape[0] // 2])
        if self.support == "grid":
            scores = self._pair_scores(comparisons, candidate_pool)
            diag = torch.diagonal(scores)
            return candidate_value(candidate_pool[diag.argmax()])

        if self.support == "continuous_rff":
            diag = self._diag_scores(comparisons, candidate_pool)
            return candidate_value(candidate_pool[diag.argmax()])

        raise ValueError(f"Unknown PairScorePFNAgent support {self.support!r}.")

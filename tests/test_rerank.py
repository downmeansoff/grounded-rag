"""Rerank: порядок задаёт cross-encoder, а не порядок кандидатов.

Модель не грузится: она весит около гигабайта, а проверять надо не качество
её оценок, а то, что оценки применяются правильно. Скорер подставляется.
"""

from __future__ import annotations

from grounded_rag.rerank.cross_encoder import Reranker
from grounded_rag.store.postgres import SearchHit


class FakeScorer:
    """Оценка = позиция текста в списке приоритетов, задаваемом тестом."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen = pairs
        return self.scores


def _hit(text: str, distance: float = 0.5) -> SearchHit:
    return SearchHit(
        reg_number="0312100006326000036",
        title="Оказание услуг по гардеробному обслуживанию",
        attachment_name="Описание объекта закупки",
        chunk_index=0,
        text=text,
        distance=distance,
    )


def test_empty_candidates_need_no_model():
    # Реранкер без модели и без имени: если бы он полез грузить, тест бы упал.
    assert Reranker("несуществующая-модель").rerank("запрос", [], top_k=5) == []


def test_order_follows_scores_not_input_order():
    hits = [_hit("первый"), _hit("второй"), _hit("третий")]
    reranker = Reranker("fake", model=FakeScorer([0.1, 0.9, 0.5]))
    assert [h.text for h in reranker.rerank("запрос", hits, top_k=3)] == [
        "второй",
        "третий",
        "первый",
    ]


def test_top_k_truncates_after_reordering():
    hits = [_hit("первый"), _hit("второй"), _hit("третий")]
    reranker = Reranker("fake", model=FakeScorer([0.1, 0.9, 0.5]))
    assert [h.text for h in reranker.rerank("запрос", hits, top_k=1)] == ["второй"]


def test_pairs_are_query_plus_chunk_text():
    scorer = FakeScorer([0.5, 0.4])
    Reranker("fake", model=scorer).rerank("какой график работы?", [_hit("а"), _hit("б")])
    assert scorer.seen == [("какой график работы?", "а"), ("какой график работы?", "б")]


def test_rerank_score_is_recorded_on_hits():
    reranker = Reranker("fake", model=FakeScorer([0.25, 0.75]))
    top = reranker.rerank("запрос", [_hit("а"), _hit("б")], top_k=2)
    assert [h.rerank_score for h in top] == [0.75, 0.25]


def test_distance_survives_rerank_untouched():
    # distance остаётся косинусным расстоянием, а не подменяется скором реранкера.
    hits = [_hit("а", distance=0.11), _hit("б", distance=0.42)]
    top = Reranker("fake", model=FakeScorer([0.1, 0.9])).rerank("запрос", hits, top_k=2)
    assert [h.distance for h in top] == [0.42, 0.11]


def test_search_hit_without_rerank_has_no_score():
    assert _hit("а").rerank_score is None

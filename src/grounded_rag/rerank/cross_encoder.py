"""Rerank кандидатов cross-encoder'ом.

Bi-encoder (то, чем ищем) кодирует запрос и чанк независимо друг от друга,
поэтому близость считается между двумя сжатыми до вектора смыслами. Этого
хватает, чтобы вытащить из корпуса три десятка правдоподобных кандидатов,
но не хватает, чтобы отличить первый из них от пятого.

Cross-encoder читает пару «запрос + чанк» вместе, одним прогоном, и выдаёт
одно число: насколько этот чанк отвечает именно на этот запрос. Он в разы
дороже на документ, поэтому применяется не к корпусу, а к коротком списку
кандидатов, который уже отобрал гибридный поиск.

Порядок при этом меняется, а состав контекста для генерации становится
плотнее: в топ-5 попадает то, что реально отвечает, а не то, что похоже.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from grounded_rag.store.postgres import SearchHit


class Scorer(Protocol):
    """Минимум от CrossEncoder: оценить пары (запрос, текст)."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class Reranker:
    def __init__(self, model_name: str, model: Scorer | None = None) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Scorer:
        # Ленивая загрузка: модель весит около гигабайта, и держать её в памяти
        # имеет смысл только если до rerank реально дошло.
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        if not hits:
            return []

        scores = self.model.predict([(query, hit.text) for hit in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(hit, rerank_score=float(score)) for hit, score in ranked[:top_k]]

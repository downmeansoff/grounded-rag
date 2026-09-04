"""Эмбеддер на GigaChat и выбор бэкенда: без сети и без ключей.

Подставляется клиент SDK, а не готовый эмбеддер, поэтому проверяется то, что
эмбеддер действительно делает сам: режет корпус на батчи, собирает ответ в
правильном порядке и не пропускает вектор чужой длины в индекс.
"""

from __future__ import annotations

import pytest

from grounded_rag.config import Settings
from grounded_rag.embed.factory import make_embedder
from grounded_rag.embed.gigachat import GigaEmbedder, normalize
from grounded_rag.errors import DimensionMismatch
from grounded_rag.llm import QuotaExhausted

DIM = 4


class Item:
    def __init__(self, embedding: list[float], index: int) -> None:
        self.embedding = embedding
        self.index = index


class Response:
    def __init__(self, data: list[Item]) -> None:
        self.data = data


class StubClient:
    """Клиент SDK, отвечающий вектором по номеру текста в батче."""

    def __init__(self, dim: int = DIM, fail: Exception | None = None) -> None:
        self.dim = dim
        self.fail = fail
        self.batches: list[list[str]] = []

    def embeddings(self, texts: list[str], model: str = "Embeddings"):
        if self.fail is not None:
            raise self.fail
        self.batches.append(list(texts))
        return Response([Item([float(i + 1)] * self.dim, i) for i in range(len(texts))])


def embedder(client: StubClient, batch: int = 32) -> GigaEmbedder:
    return GigaEmbedder(credentials="x", scope="y", dim=DIM, batch=batch, client=client)


def test_corpus_goes_out_in_batches():
    # Весь корпус одним запросом нельзя: отказ на таком запросе теряет прогон
    # целиком, а не последний батч.
    client = StubClient()
    vectors = embedder(client, batch=2).embed_passages(["a", "b", "c", "d", "e"])

    assert [len(b) for b in client.batches] == [2, 2, 1]
    assert len(vectors) == 5


def test_order_follows_index_not_arrival():
    # Порядок ответа не обещан, а перепутанные векторы не дают ошибки: чанки
    # просто молча получают чужие эмбеддинги, и найти это потом нечем.
    class Shuffled(StubClient):
        def embeddings(self, texts, model="Embeddings"):
            return Response([Item([2.0] * DIM, 1), Item([1.0] * DIM, 0)])

    vectors = embedder(Shuffled()).embed_passages(["первый", "второй"])

    assert vectors[0] == normalize([1.0] * DIM)
    assert vectors[1] == normalize([2.0] * DIM)


def test_vector_of_another_length_does_not_reach_the_index():
    client = StubClient(dim=DIM + 1)

    with pytest.raises(DimensionMismatch) as exc:
        embedder(client).embed_query("что угодно")

    assert str(DIM + 1) in str(exc.value)


def test_vectors_come_out_unit_length():
    # Косинусной метрике длина безразлична, а сравнению бэкендов нет: distance
    # в выдаче должен значить одно и то же у обеих моделей.
    vectors = embedder(StubClient()).embed_passages(["a"])
    assert sum(x * x for x in vectors[0]) == pytest.approx(1.0)


def test_normalize_leaves_a_zero_vector_alone():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_query_returns_one_vector():
    vector = embedder(StubClient()).embed_query("сколько мест в гардеробе")
    assert len(vector) == DIM


def test_exhausted_quota_is_not_retried():
    # Тот же разбор отказов, что и у чат-модели: 402 повтором не лечится, и
    # пятьсот обречённых запросов подряд это только потерянное время.
    client = StubClient(fail=RuntimeError("402 Payment Required"))

    with pytest.raises(QuotaExhausted):
        embedder(client).embed_query("вопрос")


def test_factory_gives_gigachat_without_touching_the_network():
    settings = Settings(
        embedding_backend="gigachat", embedding_dim=1024, gigachat_credentials="ключ"
    )
    assert isinstance(make_embedder(settings), GigaEmbedder)


def test_factory_refuses_a_backend_that_does_not_fit_the_index():
    # Самая дорогая ошибка при смене бэкенда: 768 в конфигурации и 1024 у
    # модели. Без этой проверки она всплывает отказом вставки посреди ingest.
    settings = Settings(
        embedding_backend="gigachat", embedding_dim=768, gigachat_credentials="ключ"
    )

    with pytest.raises(DimensionMismatch) as exc:
        make_embedder(settings)

    assert "1024" in str(exc.value)


def test_factory_refuses_gigachat_without_credentials():
    settings = Settings(
        embedding_backend="gigachat", embedding_dim=1024, gigachat_credentials=""
    )

    with pytest.raises(RuntimeError, match="GIGACHAT_CREDENTIALS"):
        make_embedder(settings)


def test_factory_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match="local, gigachat"):
        make_embedder(Settings(embedding_backend="openai"))

"""Эмбеддер на GigaChat под тем же протоколом, что и локальная модель.

Второй бэкенд нужен не чтобы заменить первый, а чтобы разницу между ними
можно было померить размеченным набором из `eval/`, а не заявить. Обе
реализации закрывают протокол `Embedder`, переключение стоит одной строки
конфигурации, и остальной конвейер о выборе не знает.

Две вещи отличаются от локального эмбеддера, и обе не косметические.

Префиксов `query:` и `passage:` здесь нет. Это соглашение семейства e5, на
котором его обучали, а не общее правило: модели GigaChat такой префикс ничего
не сообщает, он для неё просто лишнее слово в начале текста.

Размерность другая, 1024 против 768, а колонка `embedding` в базе объявлена
под конкретное число. Поэтому сменить бэкенд на готовом индексе нельзя, его
надо пересобрать, и об этом лучше узнать из внятной ошибки, чем из отказа
вставки на пятисотом чанке.
"""

from __future__ import annotations

import math

from grounded_rag.errors import DimensionMismatch
from grounded_rag.llm import call_with_retries

# Размерность модели GigaChat Embeddings. Это не настройка, а факт про модель:
# другое число здесь не изменит ответ сервера, а только сломает вставку.
GIGA_EMBEDDING_DIM = 1024

# Сколько текстов уходит в один запрос. Весь корпус одним запросом слать нельзя:
# 550 чанков это около мегабайта текста, и любой отказ на таком запросе теряет
# весь прогон целиком, а не один батч.
BATCH = 32


def normalize(vector: list[float]) -> list[float]:
    """Приводит вектор к единичной длине.

    Для самого поиска это не обязательно: в базе стоит косинусная метрика, а
    она длину вектора игнорирует. Обязательно это для сравнения бэкендов, ради
    которого второй бэкенд и заводился: число distance в выдаче должно значить
    одно и то же независимо от того, какая модель считала вектор.
    """
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return vector
    return [x / length for x in vector]


class GigaEmbedder:
    def __init__(
        self,
        credentials: str,
        scope: str,
        dim: int = GIGA_EMBEDDING_DIM,
        model: str = "Embeddings",
        batch: int = BATCH,
        client=None,
    ) -> None:
        self.credentials = credentials
        self.scope = scope
        self.dim = dim
        self.model = model
        self.batch = batch
        # Готовый клиент приходит из тестов: так проверяется сам эмбеддер,
        # батчи и разбор ответа, а не заглушка вместо него.
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from gigachat import GigaChat

            # verify_ssl_certs=False по той же причине, что и в llm.py: цепочка
            # Сбера подписана «Минцифры России», корневого сертификата которого
            # нет в системном хранилище Windows.
            self._client = GigaChat(
                credentials=self.credentials,
                scope=self.scope,
                verify_ssl_certs=False,
            )
        return self._client

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            vectors.extend(self._embed(texts[start : start + self.batch]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = call_with_retries(lambda: self.client.embeddings(texts, model=self.model))
        # Порядок ответа не обещан, поле index в нём есть именно поэтому.
        # Перепутанные местами векторы дадут молча неправильный индекс:
        # ошибки не будет, просто чанки получат чужие эмбеддинги.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [normalize(self._checked(item.embedding)) for item in ordered]

    def _checked(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dim:
            raise DimensionMismatch(
                f"модель вернула вектор длины {len(vector)}, а индекс заведён под {self.dim}. "
                f"Поставьте EMBEDDING_DIM={len(vector)} и пересоберите индекс: "
                f"размерность колонки задаётся при создании таблицы и не меняется на лету"
            )
        return vector

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GigaEmbedder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

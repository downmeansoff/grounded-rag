"""Локальный эмбеддер на sentence-transformers. Без ключей, работает офлайн после скачивания модели.

Позже заменяется на GigaEmbeddings (RU) под тем же интерфейсом Embedder —
дневная замена бэкенда, не переписывание пайплайна.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# Модели семейства e5 обучены с явными префиксами: без них качество заметно проседает.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class LocalEmbedder:
    def __init__(self, model_name: str, dim: int):
        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_E5_PASSAGE_PREFIX + t for t in texts]
        vecs = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            _E5_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()

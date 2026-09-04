"""Выбор бэкенда эмбеддингов по конфигурации.

Скрипты не должны знать, какая модель считает векторы: ingest, query, ask и
замер спрашивают эмбеддер здесь и работают с протоколом `Embedder`. Иначе
смена бэкенда это правка пяти файлов, а не одной переменной окружения.

Здесь же ловится единственная несостыковка, которая иначе всплывёт поздно и
непонятно: размерность. Колонка в базе заведена под конкретное число, у
моделей оно разное, и рассказать об этом надо до того, как прогон потратит
двадцать минут и упрётся в отказ вставки.
"""

from __future__ import annotations

from grounded_rag.config import Settings
from grounded_rag.embed.base import Embedder
from grounded_rag.embed.gigachat import GIGA_EMBEDDING_DIM
from grounded_rag.errors import DimensionMismatch

BACKENDS = ("local", "gigachat")


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend not in BACKENDS:
        raise ValueError(
            f"неизвестный EMBEDDING_BACKEND={settings.embedding_backend!r}, "
            f"доступны: {', '.join(BACKENDS)}"
        )

    if settings.embedding_backend == "gigachat":
        if not settings.gigachat_credentials:
            raise RuntimeError(
                "EMBEDDING_BACKEND=gigachat, но GIGACHAT_CREDENTIALS пуст: "
                "этот бэкенд ходит в сеть и без ключа не работает"
            )
        if settings.embedding_dim != GIGA_EMBEDDING_DIM:
            raise DimensionMismatch(
                f"EMBEDDING_BACKEND=gigachat даёт векторы длины {GIGA_EMBEDDING_DIM}, "
                f"а EMBEDDING_DIM={settings.embedding_dim}. Смена бэкенда требует "
                f"пересбора индекса: поставьте EMBEDDING_DIM={GIGA_EMBEDDING_DIM}, "
                f"удалите таблицу chunks и запустите ingest заново"
            )
        from grounded_rag.embed.gigachat import GigaEmbedder

        return GigaEmbedder(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            dim=settings.embedding_dim,
            model=settings.gigachat_embedding_model,
            batch=settings.gigachat_embedding_batch,
        )

    # Импорт внутри ветки: он тянет sentence-transformers и torch, а это
    # несколько секунд на старте. Платить их, выбрав другой бэкенд, незачем.
    from grounded_rag.embed.local import LocalEmbedder

    return LocalEmbedder(settings.embedding_model, settings.embedding_dim)

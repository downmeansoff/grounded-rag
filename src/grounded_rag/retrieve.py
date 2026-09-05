"""Retrieval одной функцией: гибридный поиск и, если включён, rerank поверх него.

Политика поиска живёт здесь, а не в скриптах: иначе `query.py` и `ask.py`
незаметно расходятся, и отладочная выдача перестаёт показывать то, что
реально уходит в генерацию.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import psycopg

from grounded_rag.config import Settings
from grounded_rag.config import settings as default_settings
from grounded_rag.rerank.cross_encoder import Reranker
from grounded_rag.store import postgres as store
from grounded_rag.store.postgres import SearchHit

_reranker: Reranker | None = None


def parse_filters(pairs: Sequence[str] | None) -> dict[str, str]:
    """Аргументы вида "Ключ=значение" в словарь фильтров.

    Живёт рядом с retrieve, потому что синтаксис фильтра должен быть один во всех
    скриптах: разойдись он, поиск в query.py и поиск в ask.py стали бы разными
    поисками при одинаковой на вид команде.
    """
    filters: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ValueError(f'Фильтр задаётся как "Ключ=значение", получено: {pair!r}')
        filters[key] = value
    return filters


def _get_reranker(config: Settings) -> Reranker:
    global _reranker
    if _reranker is None or _reranker.model_name != config.rerank_model:
        _reranker = Reranker(config.rerank_model)
    return _reranker


def retrieve(
    conn: psycopg.Connection,
    query_embedding: list[float],
    query_text: str,
    k: int = 5,
    config: Settings | None = None,
    filters: Mapping[str, str] | None = None,
) -> list[SearchHit]:
    config = config or default_settings

    if not config.use_rerank:
        return store.search_hybrid(conn, query_embedding, query_text, k=k, filters=filters)

    # Cross-encoder'у нужен запас кандидатов: смысл rerank в том, чтобы поднять
    # наверх чанк, который гибридный поиск поставил, скажем, двадцатым.
    candidates = store.search_hybrid(
        conn,
        query_embedding,
        query_text,
        k=config.rerank_candidates,
        candidates=config.rerank_candidates,
        filters=filters,
    )
    return _get_reranker(config).rerank(query_text, candidates, top_k=k)

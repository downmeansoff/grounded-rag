"""Retrieval одной функцией: гибридный поиск и, если включён, rerank поверх него.

Политика поиска живёт здесь, а не в скриптах: иначе `query.py` и `ask.py`
незаметно расходятся, и отладочная выдача перестаёт показывать то, что
реально уходит в генерацию.
"""

from __future__ import annotations

import psycopg

from grounded_rag.config import Settings, settings as default_settings
from grounded_rag.rerank.cross_encoder import Reranker
from grounded_rag.store import postgres as store
from grounded_rag.store.postgres import SearchHit

_reranker: Reranker | None = None


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
) -> list[SearchHit]:
    config = config or default_settings

    if not config.use_rerank:
        return store.search_hybrid(conn, query_embedding, query_text, k=k)

    # Cross-encoder'у нужен запас кандидатов: смысл rerank в том, чтобы поднять
    # наверх чанк, который гибридный поиск поставил, скажем, двадцатым.
    candidates = store.search_hybrid(
        conn,
        query_embedding,
        query_text,
        k=config.rerank_candidates,
        candidates=config.rerank_candidates,
    )
    return _get_reranker(config).rerank(query_text, candidates, top_k=k)

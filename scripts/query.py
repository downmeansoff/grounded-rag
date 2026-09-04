"""Простой поиск по проиндексированному корпусу.

Использование:
    python scripts/query.py "текст запроса"
"""

from __future__ import annotations

import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from grounded_rag.config import settings
from grounded_rag.embed.factory import make_embedder
from grounded_rag.retrieve import retrieve
from grounded_rag.store import postgres as store


def main(query: str, k: int = 5) -> None:
    embedder = make_embedder(settings)
    conn = store.connect(settings.dsn)

    query_vec = embedder.embed_query(query)
    hits = retrieve(conn, query_vec, query, k=k)

    if not hits:
        print("Ничего не найдено.")
        return

    for i, hit in enumerate(hits, 1):
        score = "" if hit.rerank_score is None else f" rerank={hit.rerank_score:.4f}"
        print(f"\n[{i}] distance={hit.distance:.4f}{score} doc={hit.doc_id} part={hit.part_name}#{hit.chunk_index}")
        print(f"    {hit.title}")
        snippet = hit.text.strip().replace("\n", " ")
        print(f"    {snippet[:300]}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python scripts/query.py "текст запроса"')
        sys.exit(1)
    main(sys.argv[1])

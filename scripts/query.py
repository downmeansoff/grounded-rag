"""Простой поиск по проиндексированному корпусу.

Использование:
    python scripts/query.py "текст запроса"
"""

from __future__ import annotations

import sys

from grounded_rag.config import settings
from grounded_rag.embed.local import LocalEmbedder
from grounded_rag.store import postgres as store


def main(query: str, k: int = 5) -> None:
    embedder = LocalEmbedder(settings.embedding_model, settings.embedding_dim)
    conn = store.connect(settings.dsn)

    query_vec = embedder.embed_query(query)
    hits = store.search(conn, query_vec, k=k)

    if not hits:
        print("Ничего не найдено.")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] distance={hit.distance:.4f} reg={hit.reg_number} attachment={hit.attachment_name}#{hit.chunk_index}")
        print(f"    {hit.title}")
        snippet = hit.text.strip().replace("\n", " ")
        print(f"    {snippet[:300]}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python scripts/query.py "текст запроса"')
        sys.exit(1)
    main(sys.argv[1])

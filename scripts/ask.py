"""Ответ с цитатами по проиндексированному корпусу (retrieval + генерация).

Использование:
    python scripts/ask.py "какой график работы?"
"""

from __future__ import annotations

import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from grounded_rag.config import settings
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.generate.gigachat_llm import answer
from grounded_rag.retrieve import retrieve
from grounded_rag.store import postgres as store


def main(query: str, k: int = 5) -> None:
    profile = make_domain(settings)
    embedder = make_embedder(settings)
    conn = store.connect(settings.dsn)

    query_vec = embedder.embed_query(query)
    hits = retrieve(conn, query_vec, query, k=k)
    conn.close()

    if not hits:
        print("Ничего не найдено.")
        return

    text = answer(
        query,
        hits,
        profile,
        settings.gigachat_credentials,
        settings.gigachat_scope,
        settings.gigachat_model,
    )
    print(text)

    print("\nИсточники:")
    for i, hit in enumerate(hits, 1):
        citation = profile.citation(hit.doc_id, hit.part_name, hit.chunk_index)
        print(f"  [{i}] {citation} - {hit.title}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python scripts/ask.py "вопрос"')
        sys.exit(1)
    main(sys.argv[1])

"""Ответ с цитатами по тендерной документации (retrieval + генерация).

Использование:
    python scripts/ask.py "какой график работы?"
"""

from __future__ import annotations

import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from grounded_rag.config import settings
from grounded_rag.embed.local import LocalEmbedder
from grounded_rag.generate.gigachat_llm import answer
from grounded_rag.retrieve import retrieve
from grounded_rag.store import postgres as store


def main(query: str, k: int = 5) -> None:
    embedder = LocalEmbedder(settings.embedding_model, settings.embedding_dim)
    conn = store.connect(settings.dsn)

    query_vec = embedder.embed_query(query)
    hits = retrieve(conn, query_vec, query, k=k)
    conn.close()

    if not hits:
        print("Ничего не найдено.")
        return

    text = answer(
        query, hits, settings.gigachat_credentials, settings.gigachat_scope, settings.gigachat_model
    )
    print(text)

    print("\nИсточники:")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] тендер {hit.reg_number} — {hit.title} ({hit.attachment_name}#{hit.chunk_index})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python scripts/ask.py "вопрос"')
        sys.exit(1)
    main(sys.argv[1])

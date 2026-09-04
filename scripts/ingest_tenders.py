"""Прогоняет корпус tenderhunt (output/docs/*.txt) через ingestion -> chunking -> embedding -> Postgres.

Использование:
    python scripts/ingest_tenders.py <путь_к_output/docs>
"""

from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

from grounded_rag.chunk.recursive import chunk_text
from grounded_rag.config import settings
from grounded_rag.embed.local import LocalEmbedder
from grounded_rag.ingest.loader import load_corpus
from grounded_rag.store import postgres as store


def main(docs_dir: Path) -> None:
    docs = load_corpus(docs_dir)
    print(f"Загружено документов: {len(docs)}")

    embedder = LocalEmbedder(settings.embedding_model, settings.embedding_dim)

    conn = store.connect(settings.dsn)
    store.ensure_schema(conn, embedder.dim)

    total_chunks = 0
    for doc in tqdm(docs, desc="ingest"):
        store.upsert_document(conn, doc)
        store.delete_chunks_for_document(conn, doc.reg_number)

        for att in doc.attachments:
            chunks = chunk_text(att.text)
            if not chunks:
                continue
            vectors = embedder.embed_passages([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                store.insert_chunk(conn, doc.reg_number, att.name, chunk.index, chunk.text, vector)
            total_chunks += len(chunks)

    conn.close()
    print(f"Готово. Чанков записано: {total_chunks}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python scripts/ingest_tenders.py <путь_к_output/docs>")
        sys.exit(1)
    main(Path(sys.argv[1]))

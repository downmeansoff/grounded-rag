"""Прогоняет корпус tenderhunt (output/docs/*.txt) через ingestion -> chunking -> embedding -> Postgres.

Использование:
    python scripts/ingest_tenders.py <путь_к_output/docs> [номер закупки ...]

Номера закупки в конце сужают прогон до этих тендеров. Нужно это из-за
Contextual Retrieval: он делает вызов LLM на каждый чанк, бесплатный тариф
GigaChat конечен, и разумно сначала обогатить один тендер, посмотреть на
выдачу и только потом платить за весь корпус.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tqdm import tqdm

from grounded_rag.chunk.recursive import chunk_text
from grounded_rag.config import settings
from grounded_rag.contextual.contextualizer import Contextualizer, enrich
from grounded_rag.embed.factory import make_embedder
from grounded_rag.ingest.loader import load_corpus
from grounded_rag.llm import GigaChatModel
from grounded_rag.store import postgres as store


def build_contextualizer() -> Contextualizer | None:
    if not settings.use_contextual:
        return None
    if not settings.gigachat_credentials:
        print("USE_CONTEXTUAL=true, но GIGACHAT_CREDENTIALS пуст: контекст не генерируется")
        return None

    model = GigaChatModel(
        settings.gigachat_credentials,
        settings.gigachat_scope,
        settings.gigachat_model,
    )
    return Contextualizer(
        model,
        cache_path=Path(settings.contextual_cache_path),
        head_chars=settings.contextual_head_chars,
    )


def main(docs_dir: Path, only: list[str] | None = None) -> None:
    docs = load_corpus(docs_dir)
    if only:
        docs = [doc for doc in docs if doc.reg_number in only]
    print(f"Загружено документов: {len(docs)}")

    embedder = make_embedder(settings)
    contextualizer = build_contextualizer()
    if contextualizer is not None:
        cached = len(contextualizer.cache) if contextualizer.cache else 0
        print(f"Contextual Retrieval включён, в кэше уже есть контекстов: {cached}")

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

            texts = [c.text for c in chunks]
            if contextualizer is None:
                contexts = [""] * len(texts)
            else:
                contexts = contextualizer.contexts_for(doc, att.name, texts)

            # Эмбеддинг считается по склейке «контекст плюс текст», в базу
            # ложится оригинальный текст: цитата должна остаться документом.
            vectors = embedder.embed_passages(
                [enrich(ctx, text) for ctx, text in zip(contexts, texts)]
            )
            for chunk, vector, context in zip(chunks, vectors, contexts):
                store.insert_chunk(
                    conn,
                    doc.reg_number,
                    att.name,
                    chunk.index,
                    chunk.text,
                    vector,
                    context=context,
                )
            total_chunks += len(chunks)

    conn.close()
    print(f"Готово. Чанков записано: {total_chunks}")

    if contextualizer is not None:
        print(f"Вызовов LLM за прогон: {contextualizer.calls}, остальное из кэша")
        if contextualizer.failures:
            print(
                f"Не удалось получить контекст для чанков: {contextualizer.failures}. "
                "Они проиндексированы без него, повторный прогон доберёт их."
            )
        if contextualizer.exhausted:
            print(
                f"Тариф GigaChat исчерпан, оставшиеся чанки пропущены: {contextualizer.skipped}. "
                "Уже полученные контексты лежат в кэше, прогон после пополнения "
                "продолжит с места обрыва и не заплатит за них второй раз."
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/ingest_tenders.py <путь_к_output/docs> [номер ...]")
        sys.exit(1)
    main(Path(sys.argv[1]), sys.argv[2:])

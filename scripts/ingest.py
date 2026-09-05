"""Гоняет корпус через ingestion -> chunking -> embedding -> Postgres.

Использование:
    python scripts/ingest.py <путь_к_папке_с_документами> [идентификатор ...]

Что именно лежит в папке и чем оно разбирается, решает профиль предметной
области из DOMAIN: tenders ждёт выгрузку tenderhunt, plain берёт любые .txt и
.md. Сам прогон про это не знает и одинаков для обоих.

Идентификаторы в конце сужают прогон до этих документов. Нужно это из-за
Contextual Retrieval: он делает вызов LLM на каждый чанк, бесплатный тариф
GigaChat конечен, и разумно сначала обогатить один документ, посмотреть на
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
from grounded_rag.domain.base import DomainProfile
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.llm import GigaChatModel
from grounded_rag.store import postgres as store


def build_contextualizer(profile: DomainProfile) -> Contextualizer | None:
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
        profile,
        cache_path=Path(settings.contextual_cache_path),
        head_chars=settings.contextual_head_chars,
    )


def main(docs_dir: Path, only: list[str] | None = None) -> None:
    profile = make_domain(settings)
    docs = profile.load(docs_dir)
    if only:
        docs = [doc for doc in docs if doc.doc_id in only]
    print(f"Профиль: {profile.name}. Загружено документов: {len(docs)}")

    embedder = make_embedder(settings)
    contextualizer = build_contextualizer(profile)
    if contextualizer is not None:
        cached = len(contextualizer.cache) if contextualizer.cache else 0
        print(f"Contextual Retrieval включён, в кэше уже есть контекстов: {cached}")

    conn = store.connect(settings.dsn)
    store.ensure_schema(conn, embedder.dim)

    total_chunks = 0
    for doc in tqdm(docs, desc="ingest"):
        store.upsert_document(conn, doc)
        store.delete_chunks_for_document(conn, doc.doc_id)

        for part in doc.parts:
            chunks = chunk_text(part.text)
            if not chunks:
                continue

            texts = [c.text for c in chunks]
            if contextualizer is None:
                contexts = [""] * len(texts)
            else:
                contexts = contextualizer.contexts_for(doc, part.name, texts)

            # Эмбеддинг считается по склейке «контекст плюс текст», в базу
            # ложится оригинальный текст: цитата должна остаться документом.
            vectors = embedder.embed_passages(
                [enrich(ctx, text) for ctx, text in zip(contexts, texts, strict=True)]
            )
            for chunk, vector, context in zip(chunks, vectors, contexts, strict=True):
                store.insert_chunk(
                    conn,
                    doc.doc_id,
                    part.name,
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
        print("Использование: python scripts/ingest.py <путь_к_документам> [идентификатор ...]")
        sys.exit(1)
    main(Path(sys.argv[1]), sys.argv[2:])

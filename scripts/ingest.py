"""Гоняет корпус через ingestion -> chunking -> embedding -> Postgres.

Использование:
    python scripts/ingest.py <путь_к_папке_с_документами> [идентификатор ...]

Что именно лежит в папке и чем оно разбирается, решает профиль предметной
области из DOMAIN: tenders ждёт выгрузку tenderhunt, plain берёт любые .txt и
.md. Сам прогон про это не знает и одинаков для обоих.

    python scripts/ingest.py --force <путь_к_папке_с_документами>

Идентификаторы в конце сужают прогон до этих документов. Нужно это из-за
Contextual Retrieval: он делает вызов LLM на каждый чанк, бесплатный тариф
GigaChat конечен, и разумно сначала обогатить один документ, посмотреть на
выдачу и только потом платить за весь корпус.

Документ, отпечаток которого совпал с уже записанным, пропускается: на архиве
закупок повторный прогон обычно добавляет несколько новых документов к тысяче
старых, и пересчитывать эмбеддинги для всей тысячи ради этого незачем. Что
входит в отпечаток, описано в `index_key.py`; смена модели, размера чанка или
профиля меняет его у всех документов сразу. `--force` переиндексирует всё,
не спрашивая отпечаток.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tqdm import tqdm

from grounded_rag.chunk.recursive import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_text
from grounded_rag.config import settings
from grounded_rag.contextual.contextualizer import Contextualizer, enrich, unpaid_chunks
from grounded_rag.domain.base import Document, DomainProfile
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.index_key import index_key
from grounded_rag.llm import GigaChatModel
from grounded_rag.store import postgres as store

# Сколько платных вызовов модели прогон делает молча. Порог, а не запрет:
# обогатить десяток новых закупок это обычная работа, а полторы тысячи
# чанков это уже счёт, который стоит увидеть до, а не после.
DEFAULT_MAX_CALLS = 200


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


def fingerprint(doc: Document, profile: DomainProfile, dim: int) -> str:
    return index_key(
        doc,
        profile_name=profile.name,
        prompt_version=profile.prompt_version,
        embedding_model=settings.embedding_model,
        embedding_dim=dim,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_OVERLAP,
        contextual=settings.use_contextual,
    )


def main(
    docs_dir: Path,
    only: list[str] | None = None,
    force: bool = False,
    drop_contexts: bool = False,
    max_calls: int = DEFAULT_MAX_CALLS,
) -> None:
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

    # Индексация без контекста поверх индекса, собранного с контекстом, тихо
    # выбрасывает описания чанков и роняет поиск: эмбеддинги считаются заново,
    # уже без них. Заметить это по выдаче нельзя, поэтому прогон обрывается.
    # Контексты при этом никуда не деваются, они лежат в кэше, и включение
    # USE_CONTEXTUAL обратно ничего не стоит.
    if not settings.use_contextual and not drop_contexts:
        enriched = store.contextual_chunks(conn)
        if enriched:
            conn.close()
            print(
                f"В индексе {enriched} чанков собраны с Contextual Retrieval, а сейчас "
                f"USE_CONTEXTUAL=false: этот прогон перезапишет их без контекста и ухудшит "
                f"поиск. Верните USE_CONTEXTUAL=true (контексты лежат в кэше, повторная "
                f"генерация ничего не стоит) либо подтвердите потерю флагом --drop-contexts."
            )
            sys.exit(1)

    # Отпечатки читаются до цикла: на архиве закупок это одно чтение вместо
    # тысячи, а сравнивать всё равно надо каждый документ.
    known = {} if force else store.index_keys(conn)
    keys = {doc.doc_id: fingerprint(doc, profile, embedder.dim) for doc in docs}
    fresh = [doc for doc in docs if known.get(doc.doc_id) != keys[doc.doc_id]]
    if len(fresh) != len(docs):
        print(f"Не изменились и пропущены: {len(docs) - len(fresh)}")
    docs = fresh

    # Сколько чанков придётся отдать модели, посчитано до первого вызова.
    # Внутри цикла предупреждать поздно: к моменту, когда счёт заметят, деньги
    # уже потрачены. Проверено на себе, замером на тысяче сгенерированных
    # документов: полторы тысячи вызовов ушло прежде, чем это стало заметно.
    if contextualizer is not None and docs:
        planned = {
            doc.doc_id: [(part.name, chunk.text) for part in doc.parts for chunk in chunk_text(part.text)]
            for doc in docs
        }
        unpaid = unpaid_chunks(contextualizer.cache, planned, profile)
        if unpaid > max_calls:
            conn.close()
            print(
                f"Контекст придётся сгенерировать для {unpaid} чанков, и это платные вызовы "
                f"модели. Порог {max_calls} задан затем, чтобы случайный прогон по большому "
                f"корпусу не потратил тариф молча. Если расход намеренный, повторите с "
                f"--max-calls {unpaid}; если контекст не нужен, поставьте USE_CONTEXTUAL=false."
            )
            sys.exit(1)
        if unpaid:
            print(f"Новых контекстов будет сгенерировано: {unpaid}, остальное из кэша")

    total_chunks = 0
    for doc in tqdm(docs, desc="ingest"):
        store.upsert_document(conn, doc, keys[doc.doc_id])
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
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(
            "Использование: python scripts/ingest.py [--force] [--drop-contexts] "
            "[--max-calls N] <путь_к_документам> [идентификатор ...]"
        )
        sys.exit(1)
    limit = DEFAULT_MAX_CALLS
    for flag in flags:
        if flag.startswith("--max-calls"):
            _, _, value = flag.partition("=")
            limit = int(value) if value.isdigit() else limit
    if "--max-calls" in sys.argv:
        position = sys.argv.index("--max-calls")
        if position + 1 < len(sys.argv) and sys.argv[position + 1].isdigit():
            limit = int(sys.argv[position + 1])
            args = [a for a in args if a != sys.argv[position + 1]]
    main(Path(args[0]), args[1:], "--force" in flags, "--drop-contexts" in flags, limit)

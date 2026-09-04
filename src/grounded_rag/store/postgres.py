from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

from grounded_rag.domain.base import Document
from grounded_rag.errors import DimensionMismatch, SchemaOutdated


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


# Имя таблицы разрешается строго в текущей схеме. Голый to_regclass('chunks')
# идёт по search_path целиком и, не найдя таблицу в первой схеме, берёт её из
# public. Тесты работают в отдельной схеме поверх той же базы, и такая проверка
# отвечала бы им про боевые таблицы: пустая тестовая схема выглядела бы как уже
# собранный индекс.
_IN_CURRENT_SCHEMA = "to_regclass(current_schema() || '.' || quote_ident(%s))"


def chunks_dim(conn: psycopg.Connection) -> int | None:
    """Размерность вектора в уже существующей таблице, None если её нет.

    pgvector держит размерность в atttypmod колонки, и это единственный способ
    узнать её, не пытаясь вставить вектор.
    """
    row = conn.execute(
        f"""
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = {_IN_CURRENT_SCHEMA} AND attname = 'embedding'
        """,
        ("chunks",),
    ).fetchone()
    return row[0] if row else None


def _has_column(conn: psycopg.Connection, table: str, column: str) -> bool:
    row = conn.execute(
        f"""
        SELECT 1 FROM pg_attribute
        WHERE attrelid = {_IN_CURRENT_SCHEMA} AND attname = %s AND NOT attisdropped
        """,
        (table, column),
    ).fetchone()
    return row is not None


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    return conn.execute(f"SELECT {_IN_CURRENT_SCHEMA} IS NOT NULL", (table,)).fetchone()[0]


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    # То же, что с размерностью, но про имена колонок: индекс, собранный до
    # появления профилей предметной области, называет документ reg_number и
    # знает про заказчика с НМЦК. CREATE TABLE IF NOT EXISTS такую таблицу не
    # трогает, и прогон дошёл бы до вставки и упал там на «column doc_id does
    # not exist», из чего не видно, что делать дальше.
    if _table_exists(conn, "documents") and not _has_column(conn, "documents", "doc_id"):
        raise SchemaOutdated(
            "таблица documents собрана прошлой схемой: документ в ней называется "
            "reg_number, а метаданные разложены по колонкам под тендеры. Колонки "
            "переименовать нельзя, не потеряв генерируемую колонку tsv, поэтому "
            "удалите таблицы chunks и documents и запустите ingest заново. "
            "Контексты лежат в кэше, повторная индексация за них не платит"
        )

    # CREATE TABLE IF NOT EXISTS готовую таблицу не трогает, поэтому смена
    # бэкенда эмбеддингов на собранном индексе иначе прошла бы молча, а упала
    # бы на вставке первого чанка ошибкой про несовпадение размерности вектора,
    # из которой не видно ни причины, ни что делать.
    existing = chunks_dim(conn)
    if existing is not None and existing != dim:
        raise DimensionMismatch(
            f"индекс собран под векторы длины {existing}, а модель даёт {dim}. "
            f"Размерность колонки задаётся при создании таблицы: чтобы сменить "
            f"бэкенд эмбеддингов, удалите таблицу chunks и запустите ingest заново"
        )
    # meta это JSONB, а не колонки: у тендера метаданные это заказчик и НМЦК,
    # у статьи автор и журнал, и общего набора полей между профилями нет.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_path TEXT
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
            part_name TEXT,
            chunk_index INT,
            text TEXT,
            context TEXT,
            embedding vector({dim})
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    ensure_fulltext(conn)


def ensure_fulltext(conn: psycopg.Connection) -> None:
    """Полнотекстовый индекс рядом с векторным: половина гибридного поиска.

    Вектор находит перефразировки, но проваливает точные редкие токены -
    номер закупки, статью закона, номер приложения. Полнотекст ловит ровно их.

    В индекс идёт не только text, но и doc_id с part_name: идентификатор
    документа внутри текста чанка обычно не встречается, он живёт в
    метаданных, и без него запрос «0312100006326000036» не находит ничего.

    Остальные метаданные в tsvector не идут. Ключи в meta у каждого профиля
    свои, а колонка объявлена GENERATED: включить их значило бы пересобирать
    её при каждой смене профиля.

    Туда же идёт context: сгенерированное описание места чанка в документе
    приносит слова, которых в самом фрагменте нет («гардероб», «штрафы»,
    «порядок оплаты»), и полнотекст начинает находить его по ним. В оригинальной
    статье про Contextual Retrieval это называется contextual BM25.

    Конфигурация 'russian' задана явно: двухаргументный to_tsvector IMMUTABLE,
    поэтому колонку можно объявить GENERATED и не поддерживать руками.
    """
    conn.execute("""
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector(
                'russian',
                coalesce(doc_id, '') || ' '
                || coalesce(part_name, '') || ' '
                || coalesce(context, '') || ' '
                || coalesce(text, '')
            )
        ) STORED
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)")


def upsert_document(conn: psycopg.Connection, doc: Document) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, title, meta, source_path)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            title = EXCLUDED.title,
            meta = EXCLUDED.meta,
            source_path = EXCLUDED.source_path
        """,
        (doc.doc_id, doc.title, Json(doc.meta), doc.source_path),
    )


def delete_chunks_for_document(conn: psycopg.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


def insert_chunk(
    conn: psycopg.Connection,
    doc_id: str,
    part_name: str,
    chunk_index: int,
    text: str,
    embedding: list[float],
    context: str = "",
) -> None:
    """text хранится оригинальным всегда.

    Контекст влияет на то, как чанк ищется (эмбеддинг считается по склейке,
    tsv включает context), но не на то, что попадает в цитату: пользователю
    показывается документ, а не пересказ документа моделью.
    """
    conn.execute(
        """
        INSERT INTO chunks (doc_id, part_name, chunk_index, text, context, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (doc_id, part_name, chunk_index, text, context, embedding),
    )


def all_chunk_texts(conn: psycopg.Connection) -> list[tuple[str, int, str]]:
    """Весь индекс как (идентификатор документа, номер чанка, текст).

    Нужно замеру: проверить, что эталонный фрагмент вообще достижим поиском.
    Фрагмент может лежать в документе, но развалиться по границе чанков, и
    тогда ни один чанк его не содержит, а метрика показывает ноль.
    """
    rows = conn.execute("SELECT doc_id, chunk_index, text FROM chunks").fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


@dataclass
class SearchHit:
    doc_id: str
    title: str
    part_name: str
    chunk_index: int
    text: str
    distance: float
    # Заполняется только на выходе rerank: у чанка, пришедшего прямо из поиска,
    # оценки cross-encoder'а нет, и притворяться, что она есть, нельзя.
    rerank_score: float | None = None


def search(conn: psycopg.Connection, query_embedding: list[float], k: int = 5) -> list[SearchHit]:
    rows = conn.execute(
        """
        SELECT c.doc_id, d.title, c.part_name, c.chunk_index, c.text,
               c.embedding <=> %s::vector AS distance
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY distance
        LIMIT %s
        """,
        (query_embedding, k),
    ).fetchall()
    return [SearchHit(*row) for row in rows]


RRF_K = 60  # сглаживание из оригинальной статьи про Reciprocal Rank Fusion


def search_hybrid(
    conn: psycopg.Connection,
    query_embedding: list[float],
    query_text: str,
    k: int = 5,
    candidates: int = 30,
) -> list[SearchHit]:
    """Вектор и полнотекст, слитые через RRF.

    Скоры двух поисков несопоставимы напрямую: косинусное расстояние и ts_rank_cd
    живут в разных шкалах, а нормализация их шкал зависит от выборки и плывёт от
    запроса к запросу. RRF складывает не скоры, а обратные ранги, поэтому шкалы
    вообще не нужны: важно лишь, насколько высоко документ встал в каждом списке.

    Поле distance остаётся настоящим косинусным расстоянием и считается для всех
    отобранных чанков, включая найденные только полнотекстом. Порядок при этом
    задаёт RRF, а не distance.
    """
    rows = conn.execute(
        """
        WITH vector_hits AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
            FROM (
                SELECT id, embedding <=> %(vec)s::vector AS distance
                FROM chunks
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(candidates)s
            ) v
        ),
        text_hits AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY rank_score DESC) AS rank
            FROM (
                SELECT c.id, ts_rank_cd(c.tsv, q) AS rank_score
                FROM chunks c, plainto_tsquery('russian', %(query)s) q
                WHERE c.tsv @@ q
                ORDER BY ts_rank_cd(c.tsv, q) DESC
                LIMIT %(candidates)s
            ) t
        ),
        fused AS (
            SELECT id, SUM(1.0 / (%(rrf)s + rank)) AS score
            FROM (
                SELECT id, rank FROM vector_hits
                UNION ALL
                SELECT id, rank FROM text_hits
            ) ranked
            GROUP BY id
            ORDER BY score DESC
            LIMIT %(k)s
        )
        SELECT c.doc_id, d.title, c.part_name, c.chunk_index, c.text,
               c.embedding <=> %(vec)s::vector AS distance
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY f.score DESC
        """,
        {
            "vec": query_embedding,
            "query": query_text,
            "candidates": candidates,
            "k": k,
            "rrf": RRF_K,
        },
    ).fetchall()
    return [SearchHit(*row) for row in rows]

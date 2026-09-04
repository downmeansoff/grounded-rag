from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from grounded_rag.ingest.loader import TenderDoc


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            reg_number TEXT PRIMARY KEY,
            title TEXT,
            customer TEXT,
            price TEXT,
            source_path TEXT
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            reg_number TEXT REFERENCES documents(reg_number) ON DELETE CASCADE,
            attachment_name TEXT,
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
    # Миграция для баз, залитых до Contextual Retrieval: колонка добавляется
    # пустой, старые чанки просто остаются без контекста.
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT")
    ensure_fulltext(conn)


def ensure_fulltext(conn: psycopg.Connection) -> None:
    """Полнотекстовый индекс рядом с векторным: половина гибридного поиска.

    Вектор находит перефразировки, но проваливает точные редкие токены -
    номер закупки, статью закона, номер приложения. Полнотекст ловит ровно их.

    В индекс идёт не только text, но и reg_number с attachment_name: номер
    закупки внутри текста чанка обычно не встречается, он живёт в метаданных,
    и без них запрос «0312100006326000036» не находит вообще ничего.

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
                coalesce(reg_number, '') || ' '
                || coalesce(attachment_name, '') || ' '
                || coalesce(context, '') || ' '
                || coalesce(text, '')
            )
        ) STORED
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)")


def upsert_document(conn: psycopg.Connection, doc: TenderDoc) -> None:
    conn.execute(
        """
        INSERT INTO documents (reg_number, title, customer, price, source_path)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (reg_number) DO UPDATE SET
            title = EXCLUDED.title,
            customer = EXCLUDED.customer,
            price = EXCLUDED.price,
            source_path = EXCLUDED.source_path
        """,
        (doc.reg_number, doc.title, doc.customer, doc.price, doc.source_path),
    )


def delete_chunks_for_document(conn: psycopg.Connection, reg_number: str) -> None:
    conn.execute("DELETE FROM chunks WHERE reg_number = %s", (reg_number,))


def insert_chunk(
    conn: psycopg.Connection,
    reg_number: str,
    attachment_name: str,
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
        INSERT INTO chunks (reg_number, attachment_name, chunk_index, text, context, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (reg_number, attachment_name, chunk_index, text, context, embedding),
    )


@dataclass
class SearchHit:
    reg_number: str
    title: str
    attachment_name: str
    chunk_index: int
    text: str
    distance: float
    # Заполняется только на выходе rerank: у чанка, пришедшего прямо из поиска,
    # оценки cross-encoder'а нет, и притворяться, что она есть, нельзя.
    rerank_score: float | None = None


def search(conn: psycopg.Connection, query_embedding: list[float], k: int = 5) -> list[SearchHit]:
    rows = conn.execute(
        """
        SELECT c.reg_number, d.title, c.attachment_name, c.chunk_index, c.text,
               c.embedding <=> %s::vector AS distance
        FROM chunks c
        JOIN documents d ON d.reg_number = c.reg_number
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
        SELECT c.reg_number, d.title, c.attachment_name, c.chunk_index, c.text,
               c.embedding <=> %(vec)s::vector AS distance
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.reg_number = c.reg_number
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

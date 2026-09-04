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
            embedding vector({dim})
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


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
) -> None:
    conn.execute(
        """
        INSERT INTO chunks (reg_number, attachment_name, chunk_index, text, embedding)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (reg_number, attachment_name, chunk_index, text, embedding),
    )


@dataclass
class SearchHit:
    reg_number: str
    title: str
    attachment_name: str
    chunk_index: int
    text: str
    distance: float


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

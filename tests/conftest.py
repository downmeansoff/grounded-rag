"""Общая для SQL-тестов база: настоящий Postgres в отдельной схеме.

Тесты поиска идут в настоящую базу, потому что проверяемое поведение живёт в
SQL: генерируемая колонка tsv, plainto_tsquery('russian', ...), слияние рангов
через RRF и условия по JSONB. Мок вокруг них проверял бы только сам мок.

Изоляция от рабочего корпуса: отдельная схема в незакоммиченной транзакции.
Откат в конце теста сносит и таблицы, и схему, поэтому боевые чанки тесты не
видят и не трогают.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from grounded_rag.config import settings
from grounded_rag.store import postgres as store


@pytest.fixture
def conn():
    try:
        connection = store.connect(settings.dsn)
    except psycopg.OperationalError:
        # Локально база опциональна: без docker compose эти тесты пропускаются,
        # остальные идут. В CI пропуск означал бы зелёный прогон, не проверивший
        # ни одного SQL, поэтому там отсутствие базы переводится в падение.
        if os.getenv("RAG_REQUIRE_POSTGRES"):
            raise
        pytest.skip("Postgres недоступен")

    connection.autocommit = False
    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS rag_test")
        connection.execute("SET search_path TO rag_test, public")
        # Подготовка внутри try: упади ensure_schema снаружи, соединение с
        # незакоммиченным CREATE SCHEMA осталось бы открытым, и следующий тест
        # висел бы на блокировке схемы вместо того, чтобы показать ошибку.
        store.ensure_schema(connection, settings.embedding_dim)
        yield connection
    finally:
        connection.rollback()
        connection.close()

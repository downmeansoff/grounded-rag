"""Гибридный поиск: полнотекст ловит то, что вектор теряет.

Тесты идут в настоящий Postgres, потому что проверяемое поведение живёт
в SQL: генерируемая колонка tsv, plainto_tsquery('russian', ...) и слияние
рангов через RRF. Мок вокруг них проверял бы только сам мок.

Изоляция от рабочего корпуса: отдельная схема в незакоммиченной транзакции.
Откат в конце теста сносит и таблицы, и схему, поэтому боевые 550 чанков
тесты не видят и не трогают.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from grounded_rag.config import settings
from grounded_rag.domain.base import Document
from grounded_rag.errors import DimensionMismatch
from grounded_rag.store import postgres as store

DIM = settings.embedding_dim


def _axis(i: int) -> list[float]:
    """Единичный вектор по оси i: расстояния между разными осями предсказуемы."""
    vec = [0.0] * DIM
    vec[i] = 1.0
    return vec


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
        store.ensure_schema(connection, DIM)
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _seed(conn, doc_id: str, chunks: list[tuple[str, list[float]]]) -> None:
    store.upsert_document(
        conn,
        Document(
            doc_id=doc_id,
            title=f"Тендер {doc_id}",
            source_path=f"/tmp/{doc_id}",
            meta={"Заказчик": "Заказчик", "НМЦК": "100000"},
        ),
    )
    for index, (text, embedding) in enumerate(chunks):
        store.insert_chunk(conn, doc_id, "Описание объекта закупки", index, text, embedding)


def test_fulltext_finds_rare_token_that_vector_ranks_last(conn):
    # Чанк с редким токеном намеренно дальше всех по вектору, причём строго:
    # при равных расстояниях порядок выбирал бы планировщик, и тест бы плавал.
    near = [0.0] * DIM
    near[0], near[1] = 0.6, 0.8  # косинусное расстояние до _axis(0) ровно 0.4
    _seed(
        conn,
        "1000000000000000001",
        [
            ("Услуги гардеробного обслуживания в рабочие дни.", _axis(0)),
            ("Требования к персоналу и форменной одежде.", near),
            ("Работы ведутся согласно СанПиН 2.1.3684-21.", _axis(2)),
        ],
    )
    query = _axis(0)

    vector_texts = [h.text for h in store.search(conn, query, k=2)]
    assert not any("СанПиН" in t for t in vector_texts)

    hybrid_texts = [h.text for h in store.search_hybrid(conn, query, "СанПиН", k=3)]
    assert any("СанПиН" in t for t in hybrid_texts)


def test_doc_id_is_searchable_though_absent_from_chunk_text(conn):
    # Идентификатор документа в тексте чанка не встречается: он живёт только в
    # метаданных, и без него в tsvector такой запрос не находит ничего.
    _seed(conn, "1000000000000000002", [("Уборка помещений по графику.", _axis(5))])
    hits = store.search_hybrid(conn, _axis(9), "1000000000000000002", k=3)
    assert [h.doc_id for h in hits] == ["1000000000000000002"]


def test_hit_found_by_both_searches_outranks_hit_found_by_one(conn):
    _seed(
        conn,
        "1000000000000000003",
        [
            # Ближайший по вектору, но запросных слов в тексте нет.
            ("Общие положения и порядок расчётов.", _axis(0)),
            # Второй по вектору, зато совпадает с запросом текстуально.
            ("Гардеробное обслуживание посетителей музея.", _axis(1)),
        ],
    )
    hits = store.search_hybrid(conn, _axis(0), "гардеробное обслуживание", k=2)
    assert hits[0].text.startswith("Гардеробное обслуживание")


def test_hybrid_respects_k(conn):
    _seed(
        conn,
        "1000000000000000004",
        [(f"Пункт {i} про оказание услуг.", _axis(i)) for i in range(10)],
    )
    assert len(store.search_hybrid(conn, _axis(0), "оказание услуг", k=4)) == 4


def test_hybrid_works_when_fulltext_matches_nothing(conn):
    # Запрос без единого общего токена не должен ронять поиск: остаётся вектор.
    _seed(conn, "1000000000000000005", [("Обслуживание гардероба.", _axis(0))])
    hits = store.search_hybrid(conn, _axis(0), "zzzqqq несуществующее", k=3)
    assert len(hits) == 1
    assert hits[0].doc_id == "1000000000000000005"


def test_generated_context_is_searchable_but_stays_out_of_the_citation(conn):
    # Contextual Retrieval со стороны базы. Слова «гардеробное обслуживание» нет
    # в тексте ни одного из чанков, оно есть только в сгенерированном контексте
    # второго. Первый при этом ближе по вектору, то есть без context в tsvector
    # он бы и остался наверху.
    store.upsert_document(
        conn,
        Document(
            doc_id="1000000000000000007",
            title="Тендер 1000000000000000007",
            source_path="/tmp/1000000000000000007",
            meta={"Заказчик": "Заказчик", "НМЦК": "100000"},
        ),
    )
    store.insert_chunk(
        conn, "1000000000000000007", "Контракт", 0,
        "Общие положения и порядок расчётов.", _axis(0),
    )
    store.insert_chunk(
        conn, "1000000000000000007", "Контракт", 1,
        "Оплата производится в течение 15 рабочих дней с даты подписания акта.", _axis(1),
        context="Фрагмент описывает порядок расчётов по контракту на гардеробное обслуживание.",
    )

    hits = store.search_hybrid(conn, _axis(0), "гардеробное обслуживание", k=2)

    assert hits[0].chunk_index == 1
    # Контекст влияет на поиск, но в цитату уходит документ, а не пересказ.
    assert "гардероб" not in hits[0].text


def test_distance_stays_real_cosine_distance(conn):
    _seed(conn, "1000000000000000006", [("Оказание услуг вахтера.", _axis(0))])
    hit = store.search_hybrid(conn, _axis(0), "оказание услуг", k=1)[0]
    assert hit.distance == pytest.approx(0.0, abs=1e-6)


def test_schema_refuses_an_embedder_of_another_dimension(conn):
    # Смена бэкенда эмбеддингов на уже собранном индексе. CREATE TABLE IF NOT
    # EXISTS готовую таблицу не трогает, поэтому без этой проверки ingest
    # доходил бы до вставки первого чанка и падал там ошибкой про длину
    # вектора, из которой не видно ни причины, ни что делать дальше.
    assert store.chunks_dim(conn) == DIM

    with pytest.raises(DimensionMismatch):
        store.ensure_schema(conn, DIM + 256)

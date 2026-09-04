"""Гибридный поиск: полнотекст ловит то, что вектор теряет.

Тесты идут в настоящий Postgres, потому что проверяемое поведение живёт
в SQL: генерируемая колонка tsv, plainto_tsquery('russian', ...) и слияние
рангов через RRF. Мок вокруг них проверял бы только сам мок.

Изоляция от рабочего корпуса: отдельная схема в незакоммиченной транзакции.
Откат в конце теста сносит и таблицы, и схему, поэтому боевые 550 чанков
тесты не видят и не трогают.
"""

from __future__ import annotations

import psycopg
import pytest

from grounded_rag.config import settings
from grounded_rag.ingest.loader import TenderDoc
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
        pytest.skip("Postgres недоступен")

    connection.autocommit = False
    connection.execute("CREATE SCHEMA IF NOT EXISTS rag_test")
    connection.execute("SET search_path TO rag_test, public")
    store.ensure_schema(connection, DIM)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _seed(conn, reg_number: str, chunks: list[tuple[str, list[float]]]) -> None:
    store.upsert_document(
        conn,
        TenderDoc(
            reg_number=reg_number,
            title=f"Тендер {reg_number}",
            customer="Заказчик",
            price="100000",
            source_path=f"/tmp/{reg_number}",
            attachments=[],
        ),
    )
    for index, (text, embedding) in enumerate(chunks):
        store.insert_chunk(conn, reg_number, "Описание объекта закупки", index, text, embedding)


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


def test_reg_number_is_searchable_though_absent_from_chunk_text(conn):
    # Номер закупки в тексте чанка не встречается: он живёт только в метаданных.
    _seed(conn, "1000000000000000002", [("Уборка помещений по графику.", _axis(5))])
    hits = store.search_hybrid(conn, _axis(9), "1000000000000000002", k=3)
    assert [h.reg_number for h in hits] == ["1000000000000000002"]


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
    assert hits[0].reg_number == "1000000000000000005"


def test_distance_stays_real_cosine_distance(conn):
    _seed(conn, "1000000000000000006", [("Оказание услуг вахтера.", _axis(0))])
    hit = store.search_hybrid(conn, _axis(0), "оказание услуг", k=1)[0]
    assert hit.distance == pytest.approx(0.0, abs=1e-6)

"""Фильтр по метаданным документа поверх поиска.

Метаданные лежат в JSONB и до сих пор в поиске не участвовали: движок умел
искать по смыслу и по словам, но не умел ответить на "только по этому
заказчику". Проверяется здесь не SQL сам по себе, а два места, где такой фильтр
обычно и ломается: он должен отсекать документы до отбора кандидатов, а не
после слияния, и он должен молчать про непопадание, а не подменять его выдачей
по всему корпусу.
"""

from __future__ import annotations

import pytest

from grounded_rag.config import settings
from grounded_rag.domain.base import Document
from grounded_rag.retrieve import parse_filters, retrieve
from grounded_rag.store import postgres as store

DIM = settings.embedding_dim
MUSEUM = "ГБУК Музей имени Алабина"
SCHOOL = "МБОУ Школа 12"


def _axis(i: int) -> list[float]:
    vec = [0.0] * DIM
    vec[i] = 1.0
    return vec


def _seed(conn, doc_id: str, meta: dict[str, str], chunks: list[tuple[str, list[float]]]) -> None:
    store.upsert_document(
        conn,
        Document(doc_id=doc_id, title=f"Тендер {doc_id}", source_path=f"/tmp/{doc_id}", meta=meta),
    )
    for index, (text, embedding) in enumerate(chunks):
        store.insert_chunk(conn, doc_id, "Описание объекта закупки", index, text, embedding)


def _two_customers(conn) -> None:
    _seed(conn, "2000000000000000001", {"Заказчик": MUSEUM}, [("Гардеробное обслуживание музея.", _axis(0))])
    _seed(conn, "2000000000000000002", {"Заказчик": SCHOOL}, [("Гардеробное обслуживание школы.", _axis(1))])


def test_filter_leaves_only_documents_of_that_customer(conn):
    _two_customers(conn)

    found = store.search_hybrid(conn, _axis(0), "гардеробное обслуживание", k=5, filters={"Заказчик": SCHOOL})

    assert [hit.doc_id for hit in found] == ["2000000000000000002"]


def test_filter_runs_before_candidates_are_taken(conn):
    # Главное свойство. Кандидатов берётся candidates штук, и нужный документ
    # стоит в общем списке ниже: отсеки мы лишнее после слияния, он бы просто не
    # дожил до фильтра. Тридцать чанков чужого заказчика ближе по вектору и
    # только они содержат слово запроса, один чанк нужного заказчика дальше всех.
    _seed(
        conn,
        "2000000000000000003",
        {"Заказчик": SCHOOL},
        [(f"Уборка помещений, участок {i}.", _axis(0)) for i in range(30)],
    )
    _seed(conn, "2000000000000000004", {"Заказчик": MUSEUM}, [("Гардероб на 200 мест.", _axis(7))])

    without = store.search_hybrid(conn, _axis(0), "уборка помещений", k=5, candidates=30)
    assert "2000000000000000004" not in {hit.doc_id for hit in without}

    within = store.search_hybrid(
        conn, _axis(0), "уборка помещений", k=5, candidates=30, filters={"Заказчик": MUSEUM}
    )
    assert [hit.doc_id for hit in within] == ["2000000000000000004"]


def test_value_is_matched_case_insensitively_as_a_substring(conn):
    # Иначе фильтром нельзя пользоваться: полное юридическое имя заказчика
    # пришлось бы вбивать посимвольно.
    _two_customers(conn)

    found = store.search_hybrid(conn, _axis(0), "гардеробное обслуживание", k=5, filters={"Заказчик": "музей"})

    assert [hit.doc_id for hit in found] == ["2000000000000000001"]


def test_wildcards_inside_the_value_are_taken_literally(conn):
    # _ и % в значении это символы искомой строки, а не шаблон LIKE.
    _seed(conn, "2000000000000000005", {"Заказчик": "ГБУК А_Б"}, [("Первый.", _axis(0))])
    _seed(conn, "2000000000000000006", {"Заказчик": "ГБУК АХБ"}, [("Второй.", _axis(1))])

    found = store.search_hybrid(conn, _axis(0), "первый", k=5, filters={"Заказчик": "А_Б"})

    assert [hit.doc_id for hit in found] == ["2000000000000000005"]


def test_doc_id_is_compared_whole(conn):
    # Идентификатор закупки это адрес, а не описание: совпадение по началу
    # номера означало бы ответ по чужому тендеру.
    _two_customers(conn)

    assert store.search_hybrid(conn, _axis(0), "гардеробное", k=5, filters={"doc_id": "20000000000000000"}) == []

    found = store.search_hybrid(conn, _axis(0), "гардеробное", k=5, filters={"doc_id": "2000000000000000002"})
    assert [hit.doc_id for hit in found] == ["2000000000000000002"]


def test_unknown_key_finds_nothing_instead_of_everything(conn):
    # Ключей метаданных движок не знает заранее: они приходят из профиля. Опечатка
    # в имени ключа должна давать пустую выдачу, а не молчаливый поиск по всему
    # корпусу, будто фильтра не было.
    _two_customers(conn)

    assert store.search_hybrid(conn, _axis(0), "гардеробное обслуживание", k=5, filters={"Заказчиик": "музей"}) == []


def test_conditions_are_combined_with_and(conn):
    _seed(conn, "2000000000000000007", {"Заказчик": MUSEUM, "НМЦК": "450000"}, [("Гардероб музея.", _axis(0))])
    _seed(conn, "2000000000000000008", {"Заказчик": MUSEUM, "НМЦК": "120000"}, [("Уборка музея.", _axis(1))])

    both = {"Заказчик": "музей", "НМЦК": "450000"}
    found = store.search_hybrid(conn, _axis(0), "музей", k=5, filters=both)

    assert [hit.doc_id for hit in found] == ["2000000000000000007"]


def test_filter_works_in_plain_vector_search_too(conn):
    # Замер гоняет и режим без полнотекста: останься фильтр только в гибриде,
    # сравнение режимов шло бы на разных корпусах.
    _two_customers(conn)

    found = store.search(conn, _axis(0), k=5, filters={"Заказчик": SCHOOL})

    assert [hit.doc_id for hit in found] == ["2000000000000000002"]


def test_retrieve_passes_the_filter_down(monkeypatch):
    # Политика поиска живёт в retrieve: потеряйся фильтр здесь, query.py и ask.py
    # молча искали бы по всему корпусу, показывая фильтр в подсказке.
    seen: dict = {}

    def fake_search_hybrid(conn, vector, text, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(store, "search_hybrid", fake_search_hybrid)
    retrieve(None, _axis(0), "запрос", k=3, filters={"Заказчик": MUSEUM})

    assert seen["filters"] == {"Заказчик": MUSEUM}


def test_parse_filters_reads_pairs():
    assert parse_filters(["Заказчик=ГБУК Музей", "НМЦК=450000"]) == {"Заказчик": "ГБУК Музей", "НМЦК": "450000"}
    assert parse_filters(None) == {}


def test_parse_filters_keeps_equals_signs_inside_the_value():
    assert parse_filters(["Формула=a=b"]) == {"Формула": "a=b"}


@pytest.mark.parametrize("bad", ["Заказчик", "=музей", "Заказчик=", "  =  "])
def test_parse_filters_rejects_half_a_pair(bad):
    # Пустая половина это не пустой фильтр: тихо пропустив её, движок ответил бы
    # по всему корпусу на запрос, заданный как узкий.
    with pytest.raises(ValueError) as error:
        parse_filters([bad])

    assert "Ключ=значение" in str(error.value)

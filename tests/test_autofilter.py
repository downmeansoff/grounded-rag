"""Заказчик из текста запроса в фильтр по метаданным.

Проверяется две вещи, и обе про то, где такая подстановка обычно врёт. Первая:
имя в вопросе и имя в метаданных не совпадают буквально ("в тюменской
поликлинике" против "ТЮМЕНСКОЙ ОБЛАСТИ ... ПОЛИКЛИНИКА №5"), поэтому лексемы
берутся из того же русского словаря Postgres, что и полнотекстовая часть
поиска. Вторая: молчать надо чаще, чем угадывать. Общие слова в названиях
("государственное", "учреждение") заказчика не различают, и вопрос без имени
заказчика обязан уйти в поиск по всему корпусу, а не в чужую закупку.
"""

from __future__ import annotations

from grounded_rag.autofilter import auto_filter, common_lexemes, customer_lexemes, pick_customer
from grounded_rag.config import settings
from grounded_rag.domain.base import Document
from grounded_rag.store import postgres as store

KEY = "Заказчик"
CLINIC = "ГАУЗ ТЮМЕНСКОЙ ОБЛАСТИ «ГОРОДСКАЯ ПОЛИКЛИНИКА №5»"
MUSEUM = "ГБУК САМАРСКОЙ ОБЛАСТИ «ОБЛАСТНОЙ МУЗЕЙ ИМЕНИ АЛАБИНА»"
UNIVERSITY = "ФГБОУ ВО «УРАЛЬСКИЙ ГОСУДАРСТВЕННЫЙ МЕДИЦИНСКИЙ УНИВЕРСИТЕТ»"
MED_BOOKS = "Работники исполнителя обязаны иметь личные медицинские книжки."


def _axis(i: int) -> list[float]:
    vec = [0.0] * settings.embedding_dim
    vec[i] = 1.0
    return vec


def _seed(conn, doc_id: str, customer: str, text: str | None = None) -> None:
    store.upsert_document(
        conn,
        Document(
            doc_id=doc_id,
            title=f"Тендер {doc_id}",
            source_path=f"/tmp/{doc_id}.txt",
            meta={KEY: customer, "НМЦК": "1 000 000,00"},
        ),
    )
    if text is not None:
        store.insert_chunk(conn, doc_id, "Описание объекта закупки", 0, text, _axis(0))


def test_shared_lexemes_do_not_decide() -> None:
    """Слово, которое есть у всех, не голосует.

    Иначе выигрывал бы заказчик с самым длинным названием: у него больше слов,
    а значит больше случайных совпадений с любым вопросом.
    """
    by_customer = {
        "Государственное учреждение А": {"государствен", "учрежден", "а"},
        "Государственное учреждение Б": {"государствен", "учрежден", "б"},
    }
    assert pick_customer({"государствен", "учрежден"}, by_customer) is None
    assert pick_customer({"государствен", "б"}, by_customer) == "Государственное учреждение Б"


def test_tie_means_no_filter() -> None:
    """Ничья это отказ, а не выбор первого попавшегося.

    Сузить корпус до чужого заказчика хуже, чем не сузить: поиск по всему
    корпусу отвечает плохо, а фильтр по чужому - уверенно и не из того
    документа.
    """
    by_customer = {"Поликлиника №5": {"поликлиник", "5"}, "Школа №5": {"школ", "5"}}
    assert pick_customer({"5"}, by_customer) is None


def test_word_of_the_domain_does_not_decide() -> None:
    """Лексема из имени, которой корпус пользуется как обычным словом, не голосует.

    Одного отсева по заказчикам мало: "медицинский" стоит в названии ровно
    одного заказчика и потому различает, хотя различает не заказчика, а тему.
    """
    by_customer = {"Медицинский университет": {"медицинск", "университет", "уральск"}}
    assert pick_customer({"медицинск", "книжк"}, by_customer, common={"медицинск"}) is None
    assert pick_customer({"уральск", "университет"}, by_customer, common={"медицинск"}) == "Медицинский университет"


def test_no_match_means_no_filter() -> None:
    by_customer = {"Поликлиника №5": {"поликлиник", "5"}}
    assert pick_customer({"график", "работ"}, by_customer) is None


def test_empty_index_is_not_an_error() -> None:
    assert pick_customer({"поликлиник"}, {}) is None


def test_lexemes_come_from_the_index(conn) -> None:
    _seed(conn, "0001", CLINIC)
    _seed(conn, "0002", MUSEUM)

    by_customer = customer_lexemes(conn, KEY)

    assert set(by_customer) == {CLINIC, MUSEUM}
    assert "поликлиник" in by_customer[CLINIC]
    # Предлоги и общие слова словарь выбрасывает сам, отдельного стоп-листа нет.
    assert "област" in by_customer[CLINIC] and "област" in by_customer[MUSEUM]


def test_customer_found_in_another_case(conn) -> None:
    """Падеж в вопросе другой, и подстрокой это не ловится."""
    _seed(conn, "0001", CLINIC)
    _seed(conn, "0002", MUSEUM)

    filters, found = auto_filter(conn, "какой штраф в тюменской поликлинике", KEY)

    assert found == CLINIC
    assert filters == {KEY: CLINIC}


def test_question_without_customer_goes_unfiltered(conn) -> None:
    _seed(conn, "0001", CLINIC)
    _seed(conn, "0002", MUSEUM)

    filters, found = auto_filter(conn, "какой график работы гардероба", KEY)

    assert found is None
    assert filters == {}


def test_manual_filter_wins(conn) -> None:
    """Заданное руками условие сильнее угаданного.

    Опечатка в имени заказчика должна оставаться опечаткой с пустой выдачей, а
    не молча подменяться догадкой движка.
    """
    _seed(conn, "0001", CLINIC)
    _seed(conn, "0002", MUSEUM)

    filters, found = auto_filter(conn, "какой штраф в тюменской поликлинике", KEY, {KEY: "музей"})

    assert found is None
    assert filters == {KEY: "музей"}


def _corpus_where_med_books_are_everywhere(conn) -> None:
    """Три документа из четырёх требуют медкнижки, и один из них медицинский вуз."""
    _seed(conn, "0001", CLINIC, MED_BOOKS)
    _seed(conn, "0002", MUSEUM, MED_BOOKS)
    _seed(conn, "0003", UNIVERSITY, MED_BOOKS)
    _seed(conn, "0004", "МБОУ «ШКОЛА №12»", "Гардероб работает с 08:00 до 20:00.")


def test_frequent_word_of_the_corpus_is_not_a_name(conn) -> None:
    _corpus_where_med_books_are_everywhere(conn)

    common = common_lexemes(conn, customer_lexemes(conn, KEY))

    assert "медицинск" in common
    # Имя собственное стоит в одном документе из четырёх и порог не переходит.
    assert "уральск" not in common and "тюменск" not in common


def test_question_about_the_domain_word_goes_unfiltered(conn) -> None:
    """Вопрос про медкнижки не должен уезжать в медицинский университет.

    Ответ там есть в каждом третьем документе, а фильтр по вузу оставил бы от
    корпуса один, где вопрос вообще ни при чём.
    """
    _corpus_where_med_books_are_everywhere(conn)

    filters, found = auto_filter(conn, "нужны ли гардеробщикам медицинские книжки", KEY)

    assert found is None
    assert filters == {}


def test_name_still_works_next_to_the_domain_word(conn) -> None:
    """Отсев частых слов не должен заодно убивать имя, стоящее рядом с ними."""
    _corpus_where_med_books_are_everywhere(conn)

    _, found = auto_filter(conn, "сколько человеко-часов в уральском медицинском университете", KEY)

    assert found == UNIVERSITY


def test_profile_without_customer_key(conn) -> None:
    """Профиль без имени владельца в метаданных: фильтра просто нет."""
    _seed(conn, "0001", CLINIC)

    filters, found = auto_filter(conn, "какой штраф в тюменской поликлинике", "")

    assert found is None
    assert filters == {}

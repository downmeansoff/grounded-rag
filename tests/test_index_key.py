"""Отпечаток документа: по нему ingest решает, переиндексировать или пропустить.

Ошибка здесь тихая в обе стороны. Отпечаток, который меняется зря, стоит
получаса пересчёта на архиве в тысячу закупок. Отпечаток, который не меняется
когда должен, оставляет в индексе векторы от прошлой модели или от прошлой
версии документа, и по выдаче этого не видно вообще: поиск просто начинает
находить не то.
"""

from __future__ import annotations

from grounded_rag.domain.base import Document, Part
from grounded_rag.index_key import index_key
from grounded_rag.store import postgres as store

SETUP = {
    "profile_name": "tenders",
    "prompt_version": "1",
    "embedding_model": "intfloat/multilingual-e5-base",
    "embedding_dim": 768,
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "contextual": False,
}


def _doc(**changes) -> Document:
    fields = {
        "doc_id": "0001",
        "title": "Оказание услуг по гардеробному обслуживанию",
        "source_path": "/tmp/0001.txt",
        "meta": {"Заказчик": "ПОЛИКЛИНИКА №5", "НМЦК": "1 000 000,00"},
        "parts": [Part(name="Описание объекта закупки", text="Гардероб на 150 мест.")],
    }
    fields.update(changes)
    return Document(**fields)


def test_same_document_and_setup_give_the_same_key() -> None:
    assert index_key(_doc(), **SETUP) == index_key(_doc(), **SETUP)


def test_text_change_changes_the_key() -> None:
    other = _doc(parts=[Part(name="Описание объекта закупки", text="Гардероб на 200 мест.")])
    assert index_key(other, **SETUP) != index_key(_doc(), **SETUP)


def test_metadata_change_changes_the_key() -> None:
    """Заказчик уезжает в промпт контекстуализации и в фильтр, а не только в отчёт."""
    other = _doc(meta={"Заказчик": "ПОЛИКЛИНИКА №7", "НМЦК": "1 000 000,00"})
    assert index_key(other, **SETUP) != index_key(_doc(), **SETUP)


def test_metadata_order_does_not_change_the_key() -> None:
    """Словарь метаданных приходит из JSONB, и порядок ключей там не обещан."""
    other = _doc(meta={"НМЦК": "1 000 000,00", "Заказчик": "ПОЛИКЛИНИКА №5"})
    assert index_key(other, **SETUP) == index_key(_doc(), **SETUP)


def test_part_name_change_changes_the_key() -> None:
    """Имя части это адрес чанка: по нему собирается цитата."""
    other = _doc(parts=[Part(name="Обоснование НМЦК", text="Гардероб на 150 мест.")])
    assert index_key(other, **SETUP) != index_key(_doc(), **SETUP)


def test_field_boundary_is_not_glued() -> None:
    """Разные разбиения одних и тех же символов дают разные отпечатки.

    Без разделителя заголовок "АБ" без метаданных и заголовок "А" с
    метаданными "Б" склеились бы в одну строку и получили один отпечаток.
    """
    left = _doc(title="АБ", meta={})
    right = _doc(title="А", meta={"": "Б"})
    assert index_key(left, **SETUP) != index_key(right, **SETUP)


def test_every_setting_participates() -> None:
    """Смена любой настройки сборки обязана переиндексировать корпус целиком.

    Иначе рядом лежали бы векторы двух разных моделей, и поиск смешивал бы два
    пространства, ничего об этом не сообщая.
    """
    base = index_key(_doc(), **SETUP)
    changed = {
        "profile_name": "plain",
        "prompt_version": "2",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_dim": 384,
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "contextual": True,
    }
    for name, value in changed.items():
        assert index_key(_doc(), **{**SETUP, name: value}) != base, name


def test_key_survives_a_restart(conn) -> None:
    """Отпечаток читается из базы, а не из памяти прогона."""
    doc = _doc()
    store.upsert_document(conn, doc, "deadbeef")

    assert store.index_keys(conn) == {"0001": "deadbeef"}


def test_document_indexed_before_the_column_appeared(conn) -> None:
    """Старая строка без отпечатка читается как «неизвестно», а не падает.

    Такой документ переиндексируется один раз и дальше пропускается.
    """
    store.upsert_document(conn, _doc())

    assert store.index_keys(conn) == {"0001": ""}


def test_key_is_replaced_on_reindex(conn) -> None:
    """Иначе изменённый документ переиндексировался бы каждый прогон заново."""
    doc = _doc()
    store.upsert_document(conn, doc, "first")
    store.upsert_document(conn, doc, "second")

    assert store.index_keys(conn) == {"0001": "second"}

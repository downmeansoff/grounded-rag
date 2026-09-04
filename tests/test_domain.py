"""Профиль предметной области: под что именно подогнан движок.

Проверяется не качество разбора конкретного формата, а то, ради чего профиль
заведён: домен выбирается настройкой, неизвестное имя падает сразу и понятно, а
на смене профиля старые контексты не приклеиваются к новым документам.

Отдельно закреплена версия промпта у tenders. Ключ кэша контекстов это
prompt_version плюс идентификатор документа, имя части и текст чанка, и любая
правка любого из них означает, что за уже оплаченные контексты придётся
заплатить второй раз.
"""

from __future__ import annotations

import pytest

from grounded_rag.config import settings
from grounded_rag.contextual.cache import cache_key
from grounded_rag.domain.factory import make_domain
from grounded_rag.domain.plain import PlainProfile
from grounded_rag.domain.tenders import TendersProfile

TENDER_FILE = """НОМЕР: 0312100006326000036
НАЗВАНИЕ: Оказание услуг по гардеробному обслуживанию
ЗАКАЗЧИК: ГБУК Музей
НМЦК: 450000
====================
### Описание объекта закупки [docx]
Предметом закупки является гардеробное обслуживание посетителей музея.

### Проект контракта [pdf]
Оплата производится в течение 15 рабочих дней с даты подписания акта.
"""


def _with_domain(name: str):
    return settings.model_copy(update={"domain": name})


def test_domain_is_chosen_by_settings():
    assert isinstance(make_domain(_with_domain("tenders")), TendersProfile)
    assert isinstance(make_domain(_with_domain("plain")), PlainProfile)


def test_unknown_domain_names_the_available_ones():
    # Опечатка в DOMAIN не должна разбирать корпус чем попало по умолчанию.
    with pytest.raises(ValueError) as error:
        make_domain(_with_domain("articles"))

    assert "articles" in str(error.value)
    assert "tenders" in str(error.value)
    assert "plain" in str(error.value)


def test_tenders_profile_reads_header_and_sections(tmp_path):
    (tmp_path / "0312100006326000036.txt").write_text(TENDER_FILE, encoding="utf-8")

    doc = TendersProfile().load(tmp_path)[0]

    assert doc.doc_id == "0312100006326000036"
    assert doc.title == "Оказание услуг по гардеробному обслуживанию"
    assert doc.meta == {"Заказчик": "ГБУК Музей", "НМЦК": "450000"}
    assert [part.name for part in doc.parts] == ["Описание объекта закупки", "Проект контракта"]
    assert doc.parts[0].ext == "docx"
    assert "гардеробное обслуживание" in doc.parts[0].text


def test_raw_keeps_the_whole_file_including_the_header(tmp_path):
    # По raw сверяется разметка замера: фрагмент из шапки существует в
    # документе, даже если разбор не забрал его ни в одну часть, и списывать
    # такое на ошибку разметки нельзя.
    (tmp_path / "0312100006326000036.txt").write_text(TENDER_FILE, encoding="utf-8")

    doc = TendersProfile().load(tmp_path)[0]

    assert doc.raw == TENDER_FILE
    assert "ЗАКАЗЧИК: ГБУК Музей" not in "".join(part.text for part in doc.parts)


def test_tenders_profile_skips_a_file_without_sections(tmp_path):
    (tmp_path / "пусто.txt").write_text("НОМЕР: 123\nНАЗВАНИЕ: Ничего\n", encoding="utf-8")
    assert TendersProfile().load(tmp_path) == []


def test_plain_profile_takes_a_file_as_a_whole(tmp_path):
    (tmp_path / "regulation.md").write_text(
        "# Регламент уборки\n\nВлажная уборка проводится ежедневно.", encoding="utf-8"
    )

    doc = PlainProfile().load(tmp_path)[0]

    assert doc.doc_id == "regulation"
    assert doc.title == "Регламент уборки"
    assert [part.name for part in doc.parts] == ["regulation.md"]
    assert doc.meta == {"Файл": "regulation.md"}


def test_plain_profile_ignores_foreign_extensions_and_empty_files(tmp_path):
    (tmp_path / "таблица.xlsx").write_text("не текст", encoding="utf-8")
    (tmp_path / "пусто.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "есть.txt").write_text("Единственный настоящий документ.", encoding="utf-8")

    assert [doc.doc_id for doc in PlainProfile().load(tmp_path)] == ["есть"]


def test_profiles_word_the_answer_prompt_for_their_own_corpus():
    tenders = TendersProfile().answer_system
    plain = PlainProfile().answer_system

    assert "тендерной документации" in tenders
    assert "тендер" not in plain
    # Общее у профилей остаётся общим: отвечать только по блокам и отказываться.
    for system in (tenders, plain):
        assert "в найденных документах ответа нет" in system


def test_tenders_prompt_version_is_pinned():
    # Версия участвует в ключе кэша. Сдвинуть её значит выбросить уже
    # оплаченные контексты всего корпуса и заплатить за них заново, поэтому
    # менять её можно только вместе с самими формулировками tenders.
    assert TendersProfile().prompt_version == "2"


def test_switching_profile_misses_the_cache_of_another_profile():
    # Иначе к документу нового домена приклеилось бы описание, написанное под
    # прошлый: контекст выглядел бы настоящим и врал бы про место чанка.
    tenders = cache_key(TendersProfile().prompt_version, "doc", "part", "текст")
    plain = cache_key(PlainProfile().prompt_version, "doc", "part", "текст")
    assert tenders != plain

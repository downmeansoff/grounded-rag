"""Генерация: честный отказ и формат контекста.

Тесты не ходят в сеть. Отказ при пустом retrieval - то место, где
«не знаю» не должно молча превратиться в правдоподобный ответ, поэтому
он проверяется без участия модели.
"""

from __future__ import annotations

from grounded_rag.domain.plain import PlainProfile
from grounded_rag.domain.tenders import TendersProfile
from grounded_rag.generate.gigachat_llm import REFUSAL, answer, build_context
from grounded_rag.store.postgres import SearchHit

PROFILE = TendersProfile()


def _hit(index: int = 0, text: str = "Работа с 8:00 до 18:00.") -> SearchHit:
    return SearchHit(
        doc_id="0312100006326000036",
        title="Оказание услуг по гардеробному обслуживанию",
        part_name="Описание объекта закупки",
        chunk_index=index,
        text=text,
        distance=0.2,
    )


def test_no_hits_gives_honest_refusal_without_calling_model():
    # Пустых credentials хватило бы для падения, если бы функция полезла в API.
    assert answer("какой график работы?", [], PROFILE, "", "", "") == REFUSAL


def test_context_numbering_starts_at_one():
    context = build_context(PROFILE, [_hit(0), _hit(1)])
    assert context.startswith("[1] ")
    assert "\n\n[2] " in context


def test_context_carries_source_coordinates():
    context = build_context(PROFILE, [_hit(7)])
    assert "0312100006326000036" in context
    assert "Описание объекта закупки#7" in context


def test_context_keeps_chunk_text():
    context = build_context(PROFILE, [_hit(text="Круглосуточно, включая выходные.")])
    assert "Круглосуточно, включая выходные." in context


def test_citation_names_the_entity_of_the_profile():
    # Одна и та же выдача подписывается по-разному: «тендер 0312...» уместно
    # там, где корпус состоит из закупок, и вводит в заблуждение там, где нет.
    assert "тендер" in build_context(TendersProfile(), [_hit()])
    assert "тендер" not in build_context(PlainProfile(), [_hit()])


def test_system_prompt_demands_citations_and_refusal():
    system = PROFILE.answer_system
    assert "в найденных документах" in system
    assert "квадратных скобках" in system


def test_refusal_wording_matches_the_prompt():
    # Отказ на пустой выдаче и отказ модели должны выглядеть одинаково: иначе
    # по тексту ответа не понять, отказался движок или отказалась модель.
    assert REFUSAL in PROFILE.answer_system
